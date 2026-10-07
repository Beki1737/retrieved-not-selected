import os
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from coalrag.data.configB import load, match_vec, P5

PROJ = Path(os.environ["PROJ"]); STORE = PROJ / "results" / "store"
K, T, TOK, V, H, EMB = load(partial_veto_coding=True, with_emb=True)
dates = K.date.values; REG = ["consensus", "split", "vetoed", "failed"]
qidx = np.where((K.date >= "2013-01-01") & K.regime.isin(REG) & (K.n_rec == 5))[0]
def strat(order, depth=50):
    seen, out = set(), []
    for j in order[:depth]:
        key = tuple(V[j])
        if key not in seen: seen.add(key); out.append(j)
    return np.array(out, dtype=int)
rows = []
for qi in qidx:
    cut = int(np.searchsorted(dates, dates[qi], side="left"))
    if cut == 0: continue
    s = BM25Okapi(TOK[:cut]).get_scores(TOK[qi]); o_bm = np.argsort(-s, kind="stable"); top = o_bm[:50]
    dt = (dates[qi] - dates[top]).astype("timedelta64[D]").astype(float)
    o_csr = top[np.argsort(-(H[top] + 0.3 * 0.5 ** (dt / 1825) + 0.2 * s[top] / (s[top].max() + 1e-9)), kind="stable")]
    E = EMB["bge-small"]; o_bge = np.argsort(-(E[:cut] @ E[qi]), kind="stable")
    L = {"BM25": o_bm, "bge": o_bge, "CSR3": o_csr, "strat-BM25": strat(o_bm), "strat-bge": strat(o_bge)}
    for t in range(5):
        for rule in ("exact", "answer"):
            h, _ = match_vec(V[:cut], V[qi], t, rule)
            for m, o in L.items():
                rows.append(dict(qid=K.res_id[qi], regime=K.regime[qi], nation=P5[t], rule=rule, method=m, n_list=len(o),
                                 r1=bool(h[o[:1]].any()), r3=bool(h[o[:3]].any()), r5=bool(h[o[:5]].any()), r10=bool(h[o[:10]].any())))
df = pd.DataFrame(rows); df.to_parquet(STORE / "e1d_stratified.parquet", index=False)
M = ["BM25", "bge", "CSR3", "strat-BM25", "strat-bge"]; pc = lambda x: f"{100*x:.0f}"
for rule, sel, lab in [("exact", ["FRA", "GBR", "USA"], "WESTERN, coalition match"), ("answer", P5, "ALL, exact configuration")]:
    a = df[(df.rule == rule) & df.nation.isin(sel)]
    g = a.groupby(["regime", "method"])[["r1", "r3", "r5", "r10"]].mean()
    tab = g.apply(lambda r: f"{pc(r.r1)}/{pc(r.r3)}/{pc(r.r5)}/{pc(r.r10)}", axis=1).unstack("method")[M].reindex(REG)
    print(f"\n==== {lab} | cell = R@1/R@3/R@5/R@10 (%) ====\n{tab.to_string()}")
    print("mean list length (distinct configurations in top-50):", a[a.method.str.startswith("strat")].groupby("method").n_list.mean().round(1).to_dict())
    n_q = a.drop_duplicates("qid").groupby("regime").size()
    for k in ["r1", "r3"]:
        per = a.groupby(["regime", "method"])[k].mean().unstack("method")[["BM25", "bge", "CSR3"]].reindex(REG)
        best_single = (per.mul(n_q, axis=0).sum() / n_q.sum()); routed = (per.max(axis=1) * n_q).sum() / n_q.sum()
        print(f"[oracle routing, {k}] best single method = {best_single.idxmax()} {100*best_single.max():.1f}% | "
              f"route by known regime = {100*routed:.1f}% | per-regime winner = {per.idxmax(axis=1).to_dict()}")
