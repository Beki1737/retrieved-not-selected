import os, re
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from coalrag.retrieval.lexical import tokenize

PROJ = Path(os.environ["PROJ"]); PROC = PROJ / "data" / "processed"; STORE = PROJ / "results" / "store"
P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]; CODE = {"Y": 1, "N": 2, "A": 3}; W = [1, 2, 4]
K = pd.read_parquet(PROC / "ckg_v1_1.parquet")
K = K[~K.symbol.str.contains("amendment", case=False, na=False)].sort_values(["date", "symbol"]).reset_index(drop=True)
def text(r):
    if r["adopted"]:
        d = str(r.get("description") or ""); m = re.search(r"\[(.*)\]", d)
        return ((m.group(1) if m else d) + " " + str(r.get("agenda") or "")).strip()
    if not r["vetoed"]: return str(r.get("choi_title") or r.get("description") or "")
    return str(r.get("agenda") or r.get("description") or "")
R = K.to_dict("records"); T = [text(r) for r in R]; TOK = [tokenize(t) for t in T]
K["ttype"] = np.where(K.adopted, "adopted", np.where(K.vetoed, "vetoed", "failed"))
K["tlen"] = [len(t) for t in T]
print("[text] mean chars by type:", K.groupby("ttype").tlen.mean().round(0).to_dict(), "| empty:", K.groupby("ttype").tlen.apply(lambda s: int((s == 0).sum())).to_dict())
V = np.array([[CODE.get(r[n], 0) if isinstance(r[n], str) else 0 for n in P5] for r in R], dtype=np.int8)
ADOPT = K.adopted.values
cnt = np.stack([(V == c).sum(1) for c in (1, 2, 3)], 1); tot = cnt.sum(1, keepdims=True)
with np.errstate(divide="ignore", invalid="ignore"):
    p = np.where(tot > 0, cnt / np.maximum(tot, 1), 0); H = -(np.where(p > 0, p * np.log2(p), 0)).sum(1) / np.log2(3)
EMB = {}
for name, path in {"bge-small": "BAAI/bge-small-en-v1.5", "mxbai-large": "$HF_MODELS/MixedBread-AI/mxbai-embed-large-v1"}.items():
    EMB[name] = SentenceTransformer(path, device="cuda").encode(T, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
dates = K.date.values
qmask = (K.date >= "2013-01-01") & K.regime.isin(["consensus", "split", "vetoed", "failed"]) & (K.n_rec == 5)
print(f"[queries] n={int(qmask.sum())} by regime={K[qmask].regime.value_counts().to_dict()} | excluded (incomplete votes, 2013+)="
      f"{int(((K.date >= '2013-01-01') & (K.n_rec < 5) & ~K.adopted).sum())}")

def match_vec(Vp, q, t, rule):
    o = [j for j in range(5) if j != t]; Vo, qo = Vp[:, o], q[o]
    comp = (Vo > 0) & (qo > 0)[None, :]; m = comp.sum(1); agree = ((Vo == qo[None, :]) & comp).sum(1)
    if rule == "legacy":
        d = m >= 1; return d & (agree >= 0.75 * m), d
    d = m >= 3; mt = d & (agree == m)
    if rule == "answer":
        td = (Vp[:, t] > 0) & (q[t] > 0); d = d & (~mt | td); mt = mt & td & (Vp[:, t] == q[t])
    return mt, d

rows = []
for qi in np.where(qmask)[0]:
    cut = int(np.searchsorted(dates, dates[qi], side="left"))        # strictly earlier
    if cut == 0: continue
    q = V[qi]; bm = BM25Okapi(TOK[:cut]); s_bm = bm.get_scores(TOK[qi]); o_bm = np.argsort(-s_bm, kind="stable")
    top = o_bm[:50]; dt = (dates[qi] - dates[top]).astype("timedelta64[D]").astype(float)
    tnorm = s_bm[top] / (s_bm[top].max() + 1e-9)
    o_csr = top[np.argsort(-(1.0 * H[top] + 0.3 * 0.5 ** (dt / 1825) + 0.2 * tnorm), kind="stable")]
    ranked = {"BM25": o_bm, "CSR3-bm25": o_csr}
    for name, E in EMB.items(): ranked[name] = np.argsort(-(E[:cut] @ E[qi]), kind="stable")
    for t in range(5):
        for rule in ("exact", "answer", "legacy"):
            mt, d = match_vec(V[:cut], q, t, rule)
            for pol, h in (("lower", mt), ("upper", mt | ~d)):
                base = dict(qid=K.res_id[qi], regime=K.regime[qi], nation=P5[t], rule=rule, policy=pol, n_pool=cut,
                            sup=bool(h.any()), sup_adopted=bool((h & ADOPT[:cut]).any()), sup_nonadopted=bool((h & ~ADOPT[:cut]).any()),
                            undefined_share=float((~d).mean()))
                for mname, o in ranked.items():
                    rows.append({**base, "method": mname, "r50": bool(h[o[:50]].any()), "r10": bool(h[o[:10]].any()), "h1": bool(h[o[0]])})
df = pd.DataFrame(rows)
for c in ["sup", "sup_adopted", "sup_nonadopted", "r50", "r10", "h1"]: df[c] = df[c].astype(int)
df.to_parquet(STORE / "e1b_funnel_configB.parquet", index=False)

REG = ["consensus", "split", "vetoed", "failed"]; M = list(ranked)
pc = lambda x: f"{100 * x:.0f}"
for label, sel in [("ALL 5 nations", P5), ("WESTERN (FRA, GBR, USA)", ["FRA", "GBR", "USA"])]:
    print(f"\n==== {label} | cell = Sup/R@50/R@10/Hit@1 (%) ====")
    for rule, pol in [("exact", "lower"), ("exact", "upper"), ("answer", "lower"), ("legacy", "lower")]:
        a = df[(df.rule == rule) & (df.policy == pol) & df.nation.isin(sel)]
        g = a.groupby(["method", "regime"])[["sup", "r50", "r10", "h1"]].mean()
        tab = g.apply(lambda r: f"{pc(r.sup)}/{pc(r.r50)}/{pc(r.r10)}/{pc(r.h1)}", axis=1).unstack("regime")[REG].reindex(M)
        print(f"-- rule={rule} undefined->{'non-match' if pol == 'lower' else 'match'}\n{tab.to_string()}")
    a = df[(df.rule == "exact") & (df.policy == "lower") & df.nation.isin(sel) & (df.method == "BM25")]
    print("-- support source, exact rule (%): any / from adopted / from non-adopted | mean undefined share of pool")
    print((a.groupby("regime")[["sup", "sup_adopted", "sup_nonadopted", "undefined_share"]].mean() * 100).round(1).reindex(REG).to_string())
print("\n[reference] v1 Table 2 (legacy rule, top-1, all nations): text 97.3/79.4/26.2, CSR-v3 62.7/84.9/3.3 for consensus/split/contested")
