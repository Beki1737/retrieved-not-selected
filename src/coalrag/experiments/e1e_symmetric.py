"""E1e: Config A' = Choi's 581 documents (all with rich summaries) as one pool and as queries across all regimes.
(i) regime crossover under symmetric text; (ii) composition law: mode-blind (hypergeometric) prediction vs observed;
(iii) xQuAD-style configuration diversification frontier; (iv) paired tests."""
import os
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from scipy.special import comb
from coalrag.retrieval.configA import ConfigA
from coalrag.core.match import P5
from coalrag.data.configB import match_vec

PROJ = Path(os.environ["PROJ"]); STORE = PROJ / "results" / "store"
cfg = ConfigA()
docs = sorted(cfg.A + cfg.N, key=lambda r: (r["date"], r["res_id"]))
ids = [d["res_id"] for d in docs]; dates = np.array([np.datetime64(d["date"]) for d in docs])
CODE = {"Y": 1, "N": 2, "A": 3}
V = np.array([[CODE.get(cfg.vot[i][n], 0) for n in P5] for i in ids], dtype=np.int8)
cnt = np.stack([(V == c).sum(1) for c in (1, 2, 3)], 1); tot = cnt.sum(1, keepdims=True)
p = np.where(tot > 0, cnt / np.maximum(tot, 1), 0.0); H = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1) / np.log2(3)
EB = np.stack([cfg.E["bge-small"][cfg.pos[i]] for i in ids]); EM = np.stack([cfg.E["mxbai-large"][cfg.pos[i]] for i in ids])
TOK = [cfg.tok[i] for i in ids]
REG = np.array([(("vetoed" if d["vetoed"] else "failed") if not d["adopted"] else d["regime"]) for d in docs])
LAMS = [0.25, 0.5, 0.75, 1.0]

def strat(order):
    seen, out = set(), []
    for j in order:
        key = tuple(V[j])
        if key not in seen: seen.add(key); out.append(j)
    return np.array(out, dtype=int)

def xquad(order, rel, lam, mode, k=10):
    cf = [tuple(V[j]) for j in order]
    if mode == "rw":
        w = {}
        for c, r in zip(cf, rel): w[c] = w.get(c, 0.0) + r
        z = sum(w.values()) or 1.0; pc = {c: v / z for c, v in w.items()}
    else:
        u = 1.0 / len(set(cf)); pc = {c: u for c in cf}
    chosen, covered, rem = [], set(), list(range(len(order)))
    while rem and len(chosen) < k:
        b = max(rem, key=lambda i: (1 - lam) * rel[i] + lam * (0.0 if cf[i] in covered else pc[cf[i]]))
        chosen.append(order[b]); covered.add(cf[b]); rem.remove(b)
    return np.array(chosen, dtype=int)

def pred_hit(n, m, k):
    if m <= 0: return 0.0
    if n - m < k: return 1.0
    return 1.0 - comb(n - m, k, exact=True) / comb(n, k, exact=True)

def csr(o, s, qi):
    top = o[:50]; dt = (dates[qi] - dates[top]).astype("timedelta64[D]").astype(float)
    tn = (s[top] - s[top].min()) / (s[top].max() - s[top].min() + 1e-9)
    return top[np.argsort(-(H[top] + 0.3 * 0.5 ** (dt / 1825) + 0.2 * tn), kind="stable")]

rows = []
for qi in [i for i in range(len(docs)) if (V[i] > 0).all()]:
    cut = int(np.searchsorted(dates, dates[qi], side="left"))
    if cut < 5: continue
    s_bm = BM25Okapi(TOK[:cut]).get_scores(TOK[qi]); o_bm = np.argsort(-s_bm, kind="stable")
    s_b = EB[:cut] @ EB[qi]; o_b = np.argsort(-s_b, kind="stable")
    s_m = EM[:cut] @ EM[qi]; o_m = np.argsort(-s_m, kind="stable")
    top_b = o_b[:50]; rel_b = (s_b[top_b] - s_b[top_b].min()) / (s_b[top_b].max() - s_b[top_b].min() + 1e-9)
    L = {"BM25": o_bm, "bge": o_b, "mxbai": o_m, "CSR3-bm25": csr(o_bm, s_bm, qi), "CSR3-bge": csr(o_b, s_b, qi), "strat-bge": strat(top_b)}
    for lam in LAMS:
        for mode in ("uni", "rw"): L[f"xq-{mode}-{lam}"] = xquad(top_b, rel_b, lam, mode)
    n50 = len(top_b)
    for t in range(5):
        for rule in ("answer", "exact"):
            h, _ = match_vec(V[:cut], V[qi], t, rule); m = int(h[top_b].sum())
            base = dict(qid=ids[qi], regime=REG[qi], nation=P5[t], rule=rule, n_pool=cut, n50=n50, m50=m, sup=bool(h.any()),
                        pred1=pred_hit(n50, m, 1), pred3=pred_hit(n50, m, 3), pred10=pred_hit(n50, m, 10))
            for name, o in L.items():
                rows.append({**base, "method": name, "r1": bool(h[o[:1]].any()), "r3": bool(h[o[:3]].any()),
                             "r5": bool(h[o[:5]].any()), "r10": bool(h[o[:10]].any())})
df = pd.DataFrame(rows); df.to_parquet(STORE / "e1e_symmetric_configA.parquet", index=False)

REGS = ["consensus", "split", "vetoed", "failed"]; pc = lambda x: f"{100 * x:.0f}"
print(f"queries={df.qid.nunique()} | by regime={df.drop_duplicates('qid').regime.value_counts().to_dict()}")
SHOW = ["BM25", "bge", "mxbai", "CSR3-bm25", "CSR3-bge", "strat-bge", "xq-uni-0.5", "xq-rw-0.5"]
for rule, sel, lab in [("answer", P5, "ALL nations, exact configuration"), ("exact", ["FRA", "GBR", "USA"], "WESTERN, coalition match")]:
    a = df[(df.rule == rule) & df.nation.isin(sel)]
    g = a.groupby(["regime", "method"])[["r1", "r3", "r10"]].mean()
    tab = g.apply(lambda r: f"{pc(r.r1)}/{pc(r.r3)}/{pc(r.r10)}", axis=1).unstack("method")[SHOW].reindex(REGS)
    print(f"\n==== SYMMETRIC TEXT (Config A') | {lab} | R@1/R@3/R@10 (%) ====\n{tab.to_string()}")
a = df[df.rule == "answer"]
comp = a[a.method == "bge"].groupby("regime")[["m50", "n50", "pred1", "pred3", "pred10", "sup"]].mean()
obs = a.groupby(["regime", "method"])[["r1", "r3", "r10"]].mean()
o = lambda k, m: (100 * obs.xs(m, level="method")[k]).round(1)
print("\n==== COMPOSITION LAW: mode-blind prediction from bge top-50 composition vs observed (exact configuration, %) ====")
print(pd.DataFrame({"matches/top50": comp.m50.round(2), "pred R@1": (100 * comp.pred1).round(1), "bge R@1": o("r1", "bge"), "BM25 R@1": o("r1", "BM25"),
                    "CSR3 R@1": o("r1", "CSR3-bge"), "pred R@10": (100 * comp.pred10).round(1), "bge R@10": o("r10", "bge"),
                    "CSR3 R@10": o("r10", "CSR3-bge"), "strat R@10": o("r10", "strat-bge"), "in pool": (100 * comp.sup).round(1)}).reindex(REGS).to_string())
b = a[a.method == "bge"].copy(); b["matches_in_top50"] = pd.cut(b.m50, [-1, 0, 1, 2, 5, 10, 25, 50], labels=["0", "1", "2", "3-5", "6-10", "11-25", "26-50"])
print("\n-- calibration of the law (bge, all regimes) --")
print(b.groupby("matches_in_top50", observed=True).agg(n=("r10", "size"), pred1=("pred1", "mean"), obs1=("r1", "mean"),
      pred10=("pred10", "mean"), obs10=("r10", "mean")).round(3).to_string())
fr = a[a.method.str.startswith("xq") | a.method.isin(["bge", "strat-bge", "CSR3-bge"])].groupby(["method", "regime"]).r3.mean().unstack("regime")[REGS]
print("\n==== CONFIGURATION DIVERSIFICATION FRONTIER (xQuAD-style; R@3 %, exact configuration) ====\n" + (100 * fr).round(1).to_string())
rng = np.random.default_rng(0)
def paired(fr_, m1, m2, col, B=5000):
    x = fr_[fr_.method == m1].set_index(["qid", "nation"])[col].astype(int); y = fr_[fr_.method == m2].set_index(["qid", "nation"])[col].astype(int)
    dq = (x - y).groupby(level=0).agg(["sum", "count"]); s, n = dq["sum"].values, dq["count"].values
    ix = rng.integers(0, len(s), (B, len(s))); bs = s[ix].sum(1) / n[ix].sum(1)
    return f"{100 * s.sum() / n.sum():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f},{100 * np.percentile(bs, 97.5):+5.1f}]"
print("\n==== paired tests, exact configuration (pp, resolution-cluster bootstrap 95% CI) ====")
for reg in REGS:
    r = a[a.regime == reg]
    print(f"{reg:9s} n_q={r.qid.nunique():3d} | R@3 strat vs CSR3-bge {paired(r, 'strat-bge', 'CSR3-bge', 'r3')} | R@3 strat vs bge {paired(r, 'strat-bge', 'bge', 'r3')} "
          f"| R@1 CSR3-bge vs bge {paired(r, 'CSR3-bge', 'bge', 'r1')}")
