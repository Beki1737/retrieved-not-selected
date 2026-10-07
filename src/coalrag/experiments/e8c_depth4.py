"""E8c (exploratory follow-up to D86): coverage of the k stratified slots by stratification depth.
Pools and similarity exactly as E8/E8b (CKG texts, bge-small; all Choi documents dated before the query)."""
import os
os.environ.setdefault("CKG_VERSION", "v1_2")
import numpy as np, pandas as pd
from coalrag.data.configB import load, match_vec
from coalrag.retrieval.configA import ConfigA

Kdf, T, TOK, V, H, EMB = load(partial_veto_coding=True, with_emb=True)
E = EMB["bge-small"]; rid = Kdf.res_id.tolist(); pos = {r: i for i, r in enumerate(rid)}
dates = Kdf.date.values; REG = Kdf.regime.values
cfg = ConfigA(); A = [pos[r["res_id"]] for r in cfg.A if r["res_id"] in pos]; N = [pos[r["res_id"]] for r in cfg.N if r["res_id"] in pos]
choi = sorted(set(A) | set(N)); complete = lambda i: bool(((V[i] >= 1) & (V[i] <= 3)).all())
DEPTHS, KS = (50, 200, None), (4, 10)


def strat(order, Vp, depth, k):
    seen, out = set(), []
    for j in (order if depth is None else order[:depth]):
        key = tuple(Vp[j])
        if key not in seen:
            seen.add(key); out.append(j)
        if len(out) == k: break
    return np.array(out, dtype=int)


rows = []
for qset, queries in [("nonadopted66", N), ("contested143", [q for q in choi if REG[q] in ("split", "vetoed", "failed")])]:
    for q in queries:
        if not complete(q): continue
        pool = np.array([j for j in choi if dates[j] < dates[q]], dtype=int)
        if len(pool) < 10: continue
        h, _ = match_vec(V[pool], V[q], 0, "answer"); o = np.argsort(-(E[pool] @ E[q]), kind="stable")
        r = dict(qset=qset, qid=rid[q], regime=REG[q], sup=bool(h.any()))
        for k in KS:
            for d in DEPTHS: r[f"k{k}_d{d or 'all'}"] = bool(h[strat(o, V[pool], d, k)].any())
        rows.append(r)
df = pd.DataFrame(rows); rng = np.random.default_rng(0); cols = ["sup"] + [f"k{k}_d{d or 'all'}" for k in KS for d in DEPTHS]


def paired(sub, a, b, B=5000):
    d = (sub[a].astype(float) - sub[b].astype(float)).values; bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return f"{100 * d.mean():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f},{100 * np.percentile(bs, 97.5):+5.1f}] (drafts changed: {int((d != 0).sum())})"


print("==== E8c: true configuration among the first k stratified configurations (%), by depth ====")
print((100 * df.groupby("qset")[cols].mean()).round(1).to_string())
print("\n-- by regime --")
print((100 * df.groupby(["qset", "regime"])[["sup", "k4_d50", "k4_dall", "k10_d50", "k10_dall"]].mean()).round(1).to_string())
print("\n-- whole pool minus depth 50 (pp, 95% CI, paired bootstrap over drafts) --")
for qs in ["nonadopted66", "contested143"]:
    s = df[df.qset == qs]
    print(f"{qs:13s} k=4:  {paired(s, 'k4_dall', 'k4_d50')} | k=10: {paired(s, 'k10_dall', 'k10_d50')}")
print("\nanchor (must match E8b 'none'): contested143 k10_d50 = 76.2, k10_dall = 83.2")
print("compare nonadopted66 k4_d50 with E7 slot coverage 45.5% (different text space for the LLM pool, so close, not equal)")
