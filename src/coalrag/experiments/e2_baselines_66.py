import os
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

PROJ = Path(os.environ["PROJ"]); PROC = PROJ / "data" / "processed"; STORE = PROJ / "results" / "store"
P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]; LAB = ["Y", "N", "A"]; W = ["FRA", "GBR", "USA"]
ckg = pd.read_parquet(PROC / "ckg_v1_1.parquet").sort_values("date").reset_index(drop=True)
ckg = ckg[~ckg.symbol.str.contains("amendment", case=False, na=False)]
Q = ckg[ckg.in_choi_test].to_dict("records")
pool = ckg[ckg.in_choi_test | ckg.in_choi_adopted].to_dict("records")
idx = pd.read_parquet(PROC / "emb_configA_index.parquet").res_id.tolist(); pos = {r: i for i, r in enumerate(idx)}
E = np.load(PROC / "emb_configA_bge-small.npy")

def mode(vals, default="Y"):
    vals = [v for v in vals if isinstance(v, str)]
    if not vals: return default
    c = Counter(vals); top = max(c.values()); return next(v for v in ["Y", "A", "N"] if c.get(v) == top)

rows = []
for q in Q:
    before = ckg[ckg.date < q["date"]]
    na_full = before[(~before.adopted) & (before.n_rec == 5)]
    el = [c for c in pool if c["date"] < q["date"]]
    sims = np.array([E[pos[c["res_id"]]] @ E[pos[q["res_id"]]] for c in el]) if el else np.array([])
    order = np.argsort(-sims, kind="stable") if el else []
    el_na = [c for c in el if not c["adopted"]]
    sims_na = np.array([E[pos[c["res_id"]]] @ E[pos[q["res_id"]]] for c in el_na]) if el_na else np.array([])
    order_na = np.argsort(-sims_na, kind="stable") if el_na else []
    for n in P5:
        hist = mode(before[n].tolist())
        pred = {"always_favour": "Y", "hist_majority": hist,
                "nonadopted_prior": mode(na_full[n].tolist(), default=hist)}
        for k in (1, 5, 10):
            pred[f"kNN{k}_combined"] = mode([el[i][n] for i in order[:k]], default=hist)
            pred[f"kNN{k}_nonadopted"] = mode([el_na[i][n] for i in order_na[:k]], default=hist)
        for m, p in pred.items():
            rows.append(dict(qid=q["res_id"], qset="vetoed" if q["vetoed"] else "failed", nation=n, method=m, y=q[n], yhat=p))
df = pd.DataFrame(rows); df.to_parquet(STORE / "e2_baselines_66.parquet", index=False)

def f1s(yt, yp):
    yt, yp = np.asarray(yt), np.asarray(yp); labs = sorted(set(yt) | set(yp)); f, s = [], []
    for l in labs:
        tp = np.sum((yt == l) & (yp == l)); fp = np.sum((yt != l) & (yp == l)); fn = np.sum((yt == l) & (yp != l))
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        f.append(2 * p * r / (p + r) if p + r else 0.0); s.append(np.sum(yt == l))
    return float(np.mean(f)), float(np.dot(f, s) / np.sum(s)), float(np.mean(yt == yp))
rng = np.random.default_rng(0)
def boot_macro(frame, B=1000):
    q = frame.qid.unique(); g = {k: v for k, v in frame.groupby("qid")}; bs = []
    for _ in range(B):
        s = pd.concat([g[k] for k in rng.choice(q, len(q))]); bs.append(f1s(s.y, s.yhat)[0])
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)

METHODS = list(dict.fromkeys(df.method))
print("== weighted F1 by nation (comparable to v1 Tables 4/6 and Supplement Table 2) ==")
t = df.groupby(["method", "nation"]).apply(lambda g: f1s(g.y, g.yhat)[1]).unstack()[P5].reindex(METHODS)
print(t.round(2).to_string())
print("\n== macro F1 by nation ==")
t = df.groupby(["method", "nation"]).apply(lambda g: f1s(g.y, g.yhat)[0]).unstack()[P5].reindex(METHODS)
print(t.round(2).to_string())
print("\n== pooled over 5 nations / Western only: macro F1 [95% CI], weighted F1, accuracy ==")
for m in METHODS:
    a = df[df.method == m]; w = a[a.nation.isin(W)]
    ma, wf, acc = f1s(a.y, a.yhat); lo, hi = boot_macro(a); mw, ww, aw = f1s(w.y, w.yhat)
    print(f"{m:22s} all: macro={ma:.3f} [{lo:.3f},{hi:.3f}] weighted={wf:.3f} acc={acc:.3f} | West: macro={mw:.3f} weighted={ww:.3f} acc={aw:.3f}")
print("\n== reference: v1 LLM weighted F1 (Supplement Table 2) ==")
print(pd.DataFrame({"GPT4omini_B0": [.50, .60, .63, .36, .66], "GPT4omini_B3": [.69, .61, .61, .46, .81],
                    "Llama70B_B0": [.52, .54, .52, .47, .63], "Llama70B_B3": [.58, .59, .59, .55, .80],
                    "Mistral24B_B3": [.49, .50, .62, .47, .59]}, index=P5).T.to_string())
