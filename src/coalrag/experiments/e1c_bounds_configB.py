import os
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from coalrag.data.configB import load, match_vec, P5

PROJ = Path(os.environ["PROJ"]); STORE = PROJ / "results" / "store"
K, T, TOK, V4, H, EMB = load(partial_veto_coding=True)
V0 = V4.copy(); V0[V0 == 4] = 0
dates = K.date.values; ADOPT = K.adopted.values; PROCN = (K.regime == "procedural_N").values; NREC = K.n_rec.values
qidx = np.where((K.date >= "2013-01-01") & K.regime.isin(["consensus", "split", "vetoed", "failed"]) & (K.n_rec == 5))[0]
W = [1, 2, 4]; REG = ["consensus", "split", "vetoed", "failed"]; METH = ["BM25", "bge-small", "mxbai-large", "CSR3-bm25"]
rows, cands = [], []
for qi in qidx:
    cut = int(np.searchsorted(dates, dates[qi], side="left"))
    if cut == 0: continue
    s = BM25Okapi(TOK[:cut]).get_scores(TOK[qi]); o_bm = np.argsort(-s, kind="stable"); top = o_bm[:50]
    dt = (dates[qi] - dates[top]).astype("timedelta64[D]").astype(float)
    o_csr = top[np.argsort(-(1.0 * H[top] + 0.3 * 0.5 ** (dt / 1825) + 0.2 * s[top] / (s[top].max() + 1e-9)), kind="stable")]
    rk = {"BM25": o_bm, "CSR3-bm25": o_csr, **{n: np.argsort(-(E[:cut] @ E[qi]), kind="stable") for n, E in EMB.items()}}
    q = V0[qi]; reg = K.regime[qi]
    for t in range(5):
        for coding, VV in (("orig", V0), ("partial", V4)):
            for rule in ("exact", "answer"):
                mt, d = match_vec(VV[:cut], q, t, rule)
                for pol, h in (("lower", mt), ("upper", mt | ~d)):
                    ha = h & ADOPT[:cut]
                    base = dict(qid=K.res_id[qi], regime=reg, nation=P5[t], coding=coding, rule=rule, policy=pol,
                                sup=h.any(), sup_adopted=ha.any(), sup_ad_proc=(ha & PROCN[:cut]).any(),
                                sup_ad_unrec=(ha & ~PROCN[:cut] & (NREC[:cut] < 5)).any(),
                                sup_ad_clean=(ha & ~PROCN[:cut] & (NREC[:cut] == 5)).any(), undef=float((~d).mean()))
                    for m in METH:
                        o = rk[m]; rows.append({**base, "method": m, "r50": h[o[:50]].any(), "r10": h[o[:10]].any(), "h1": bool(h[o[0]])})
        if reg in ("vetoed", "failed") and t in W:
            mt, d = match_vec(V4[:cut], q, t, "exact")
            for m in ("BM25", "bge-small", "mxbai-large"):
                for j in rk[m][:10]:
                    if not d[j]:
                        cands.append(dict(cand=K.res_id[j], symbol=K.symbol[j], date=K.date[j].date(), agenda=str(K.agenda[j])[:60],
                                          vetoers=",".join(n for n in P5 if K[n][j] == "N"), qid=K.res_id[qi], method=m))
df = pd.DataFrame(rows)
for c in [c for c in df.columns if c.startswith(("sup", "r", "h1"))]:
    if df[c].dtype == bool: df[c] = df[c].astype(int)
df.to_parquet(STORE / f"e1c_bounds_configB_{os.environ.get('CKG_VERSION', 'v1_1')}.parquet", index=False)
pc = lambda x: f"{100 * x:.0f}"
w = df[df.nation.isin(["FRA", "GBR", "USA"]) & (df.rule == "exact")]
print("==== WESTERN, exact rule | cell = Sup/R@10/Hit@1 (%) under coding x undefined-policy ====")
g = w.groupby(["regime", "method", "coding", "policy"])[["sup", "r10", "h1"]].mean()
tab = g.apply(lambda r: f"{pc(r.sup)}/{pc(r.r10)}/{pc(r.h1)}", axis=1).unstack(["coding", "policy"])
print(tab.reindex(pd.MultiIndex.from_product([REG, METH])).to_string())
print("\n==== WESTERN, exact, partial coding, lower: where adopted support comes from (% of queries) ====")
a = w[(w.coding == "partial") & (w.policy == "lower") & (w.method == "BM25")]
print((a.groupby("regime")[["sup", "sup_adopted", "sup_ad_proc", "sup_ad_unrec", "sup_ad_clean"]].mean() * 100).round(1).reindex(REG).to_string())
print("\n==== undefined candidates in top-10 (vetoed/failed Western queries, partial coding) ====")
c = pd.DataFrame(cands)
if len(c):
    c["decade"] = [f"{d.year // 10 * 10}s" for d in c.date]
    print("by decade:", c.drop_duplicates(["cand"]).decade.value_counts().sort_index().to_dict(), "| distinct drafts:", c.cand.nunique())
    top = c.groupby(["symbol", "date", "agenda", "vetoers"]).size().sort_values(ascending=False).head(40)
    print(top.to_string())
    c.drop_duplicates("cand")[["cand", "symbol", "date", "agenda", "vetoers"]].sort_values("date").to_csv(STORE / "recovery_candidates.csv", index=False)
rng = np.random.default_rng(0)
def paired(fr, m1, m2, col, B=5000):
    a = fr[fr.method == m1].set_index(["qid", "nation"])[col]; b = fr[fr.method == m2].set_index(["qid", "nation"])[col]
    dq = (a - b).groupby(level=0).agg(["sum", "count"]); s, n = dq["sum"].values, dq["count"].values
    ix = rng.integers(0, len(s), (B, len(s))); bs = s[ix].sum(1) / n[ix].sum(1)
    return f"{100 * s.sum() / n.sum():+.1f} [{100 * np.percentile(bs, 2.5):+.1f},{100 * np.percentile(bs, 97.5):+.1f}]"
print("\n==== paired differences, WESTERN, exact, partial coding, lower (pp, resolution-cluster bootstrap) ====")
b = w[(w.coding == "partial") & (w.policy == "lower")]
for reg in REG:
    r = b[b.regime == reg]
    print(f"{reg:10s} n_q={r.qid.nunique():3d} | Hit@1 CSR3-BM25 {paired(r,'CSR3-bm25','BM25','h1')} | CSR3-bge {paired(r,'CSR3-bm25','bge-small','h1')} "
          f"| R@10 CSR3-BM25 {paired(r,'CSR3-bm25','BM25','r10')} | mxbai-bge {paired(r,'mxbai-large','bge-small','r10')}")
