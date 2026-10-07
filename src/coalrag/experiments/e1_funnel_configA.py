import os
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from coalrag.core.match import P5, RULES, resolve, norm_entropy
from coalrag.retrieval.lexical import lt_point, tokenize, doc_text
from coalrag.legacy_io import load_pickle

PROJ = Path(os.environ["PROJ"]); PROC = PROJ / "data" / "processed"; STORE = PROJ / "results" / "store"; STORE.mkdir(parents=True, exist_ok=True)
ckg = pd.read_parquet(PROC / "ckg_v1_1.parquet")
A = ckg[ckg.in_choi_adopted].to_dict("records"); N = ckg[ckg.in_choi_test].to_dict("records"); Q = N
idx = pd.read_parquet(PROC / "emb_configA_index.parquet").res_id.tolist(); pos = {r: i for i, r in enumerate(idx)}
EMB = {m: np.load(PROC / f"emb_configA_{m}.npy") for m in ["bge-small", "mxbai-large"]}
TOK = {r["res_id"]: tokenize(doc_text(r)) for r in A + N}
VOT = {r["res_id"]: {n: (r[n] if isinstance(r[n], str) else None) for n in P5} for r in A + N}
METHODS = ["LT-orig", "LT", "BM25", "bge-small", "mxbai-large", "CSR3-orig", "CSR3"]

def rank(q, el, method, pool):
    if not el: return []
    if method.startswith(("LT", "CSR3")):
        fixed = not method.endswith("orig"); sc = []
        for c in el:
            p, reg = lt_point(q, c, fixed)
            if not fixed and pool == "N" and not reg: continue      # Choi region gate, non-adopted pool only
            sc.append((p, c["date"], c))
        sc.sort(key=lambda t: (t[0], t[1]), reverse=True)           # Choi: (point, date) descending
        if method.startswith("CSR3"):                               # Eq. 2 over the Stage-1 top-50
            f = lambda t: 1.0 * norm_entropy(VOT[t[2]["res_id"]]) + 0.3 * 0.5 ** ((q["date"] - t[2]["date"]).days / 1825) + 0.2 * min(t[0] / 5, 1.0)
            sc = sorted(sc[:50], key=f, reverse=True)
        return [c for _, _, c in sc]
    if method == "BM25":                                            # IDF from the eligible pool only (no future term stats)
        s = BM25Okapi([TOK[c["res_id"]] for c in el]).get_scores(TOK[q["res_id"]])
    else:
        E = EMB[method]; qv = E[pos[q["res_id"]]]; s = np.array([E[pos[c["res_id"]]] @ qv for c in el])
    return [el[i] for i in np.argsort(-np.asarray(s), kind="stable")]

rows = []
for q in Q:
    M = VOT[q["res_id"]]
    for pool, P in [("A", A), ("N", N)]:
        el_all = [c for c in P if c["symbol_norm"] != q["symbol_norm"] and c["date"] <= q["date"]]   # Choi: same-day allowed
        el_strict = [c for c in el_all if c["date"] < q["date"]]                                     # ours: strictly earlier
        ranked = {m: rank(q, el_all if m.endswith("orig") else el_strict, m, pool) for m in METHODS}
        for n in P5:
            for rule, fn in RULES.items():
                raw = {c["res_id"]: fn(M, VOT[c["res_id"]], n) for c in el_all}
                for pol in ("lower", "upper"):
                    h = {k: resolve(v, pol) for k, v in raw.items()}
                    anyh = lambda cs: any(h[c["res_id"]] for c in cs)
                    for m in METHODS:
                        el = el_all if m.endswith("orig") else el_strict; rk = ranked[m]
                        rows.append(dict(qid=q["res_id"], symbol=q["symbol"], qset="vetoed" if q["vetoed"] else "failed",
                                         nation=n, pool=pool, method=m, rule=rule, policy=pol, n_el=len(el), n_ranked=len(rk),
                                         same_day=len(el_all) - len(el_strict), sup=anyh(el), r50=anyh(rk[:50]),
                                         r10=anyh(rk[:10]), h1=anyh(rk[:1])))
df = pd.DataFrame(rows)
for c in ["sup", "r50", "r10", "h1"]: df[c] = df[c].astype(int)
df.to_parquet(STORE / "e1_funnel_configA.parquet", index=False)
print(f"rows={len(df)} | queries={df.qid.nunique()} | same-day candidates available to *-orig: "
      f"{int(df[(df.method=='LT-orig')&(df.rule=='exact')&(df.policy=='lower')&(df.nation=='CHN')].same_day.sum())} "
      f"(over pools A+N, summed over queries)")

d = df[df.policy == "lower"]; pct = lambda x: f"{100 * x:.0f}"; W = ["FRA", "GBR", "USA"]
rng = np.random.default_rng(0)
def boot(frame, col, B=2000):
    g = frame.groupby("qid")[col].agg(["sum", "count"]); s, c = g["sum"].values, g["count"].values
    ix = rng.integers(0, len(s), (B, len(s))); bs = s[ix].sum(1) / c[ix].sum(1)
    return 100 * s.sum() / c.sum(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)
DV = {}
for rule in ["exact", "answer", "legacy"]:
    for pool in ["A", "N"]:
        g = d[(d.rule == rule) & (d.pool == pool)].groupby(["method", "nation"])[["sup", "r50", "r10", "h1"]].mean()
        tab = g.apply(lambda r: f"{pct(r.sup)}/{pct(r.r50)}/{pct(r.r10)}/{pct(r.h1)}", axis=1).unstack("nation")[list(P5)].reindex(METHODS)
        print(f"\n== rule={rule} pool={'adopted(515)' if pool=='A' else 'non-adopted(66)'}   cell = Sup/R@50/R@10/Hit@1 (%) ==\n{tab.to_string()}")
    dv = d[d.rule == rule].pivot_table(index=["method", "qid", "nation"], columns="pool", values="h1", aggfunc="max").fillna(0)
    dv["deliv"] = ((dv["A"] > 0) | (dv["N"] > 0)).astype(int); dv = dv.reset_index(); DV[rule] = dv
    tab = dv.groupby(["method", "nation"]).deliv.mean().unstack("nation")[list(P5)].reindex(METHODS)
    print(f"\n== rule={rule} DELIVERED = top-1 of either pool matches (%) ==\n{(100 * tab).round(1).to_string()}")

print("\n== Western targets (FRA, GBR, USA) pooled, DELIVERED %, resolution-cluster bootstrap 95% CI ==")
for m in METHODS:
    cells = []
    for rule in ["legacy", "exact", "answer"]:
        x = DV[rule]; est, lo, hi = boot(x[(x.method == m) & (x.nation.isin(W))], "deliv"); cells.append(f"{rule}={est:.1f} [{lo:.1f},{hi:.1f}]")
    print(f"{m:12s} " + "  ".join(cells))

st = load_pickle(next((PROJ / "legacy").rglob("canonical_state_v3.pkl")))
print("\n== LEGACY retrieval_results (Colab v1), for replication check ==")
for k, v in st["retrieval_results"].items():
    try: print(f"-- {k}\n{pd.DataFrame(v).round(3).to_string()}")
    except Exception: print(f"-- {k}: {str(v)[:600]}")
