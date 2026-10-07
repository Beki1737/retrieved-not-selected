"""E1f on Config A' (581 Choi documents, symmetric text; exact five-vote configuration):
(i) R@k curves; (ii) stage-1 depth for Coalition-Stratified Retrieval (does stratifying deeper recover the R@50 loss?);
(iii) temporal embargo (exclude precedents within delta days: near-duplicate redrafts / leakage, VD1Y-4);
(iv) a training-free regime router: share of contested drafts among the 10 nearest earlier precedents; threshold tuned on
queries before 2013, evaluated on 2013 onwards; realized vs oracle routing between bge (R@1) and CSR3-bge (R@1)."""
import os
from pathlib import Path
import numpy as np, pandas as pd
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
EB = np.stack([cfg.E["bge-small"][cfg.pos[i]] for i in ids])
REG = np.array([(("vetoed" if d["vetoed"] else "failed") if not d["adopted"] else d["regime"]) for d in docs])
CONT = REG != "consensus"
DEPTHS, EMB, KS = [10, 25, 50, 100, 200, 0], [0, 30, 90, 365], [1, 2, 3, 5, 10]

def strat(order):
    seen, out = set(), []
    for j in order:
        key = tuple(V[j])
        if key not in seen: seen.add(key); out.append(j)
    return np.array(out, dtype=int)

def csr(o, s, qi):
    top = o[:50]; dt = (dates[qi] - dates[top]).astype("timedelta64[D]").astype(float)
    tn = (s[top] - s[top].min()) / (s[top].max() - s[top].min() + 1e-9)
    return top[np.argsort(-(H[top] + 0.3 * 0.5 ** (dt / 1825) + 0.2 * tn), kind="stable")]

rows = []
for qi in [i for i in range(len(docs)) if (V[i] > 0).all()]:
    for delta in EMB:
        cut = int(np.searchsorted(dates, dates[qi] - np.timedelta64(delta, "D"), side="left"))
        if cut < 5: continue
        h, _ = match_vec(V[:cut], V[qi], 0, "answer")
        s = EB[:cut] @ EB[qi]; o = np.argsort(-s, kind="stable")
        L = {"bge": o, "CSR3-bge": csr(o, s, qi), "strat-50": strat(o[:50])}
        if delta == 0:
            for d in DEPTHS:
                if d != 50: L[f"strat-{d or 'all'}"] = strat(o[:d] if d else o)
        base = dict(qid=ids[qi], regime=REG[qi], year=int(str(dates[qi])[:4]), delta=delta, n_pool=cut, sup=bool(h.any()),
                    r50=bool(h[o[:50]].any()), p_cont=float(CONT[o[:10]].mean()))
        for name, oo in L.items():
            rows.append({**base, "method": name, "n_cfg": len(oo) if name.startswith("strat") else np.nan,
                         **{f"r{k}": bool(h[oo[:k]].any()) for k in KS}})
df = pd.DataFrame(rows); df.to_parquet(STORE / "e1f_depth_embargo.parquet", index=False)
REGS = ["consensus", "split", "vetoed", "failed"]; pc = lambda x: (100 * x).round(1)
z = df[df.delta == 0]
print(f"queries={z.qid.nunique()} | by regime={z.drop_duplicates('qid').regime.value_counts().reindex(REGS).to_dict()}")

print("\n==== (i) R@k curves, delta=0 (%): bge / CSR3-bge / strat-50 ====")
for m in ["bge", "CSR3-bge", "strat-50"]:
    print(f"-- {m} --\n" + pc(z[z.method == m].groupby("regime")[[f"r{k}" for k in KS]].mean().reindex(REGS)).to_string())

print("\n==== (ii) STAGE-1 DEPTH for stratification, delta=0 (R@3 / R@5 / R@10 %; mean distinct configurations) ====")
D = [f"strat-{d or 'all'}" for d in DEPTHS]; t = z[z.method.isin(D)]
tab = t.groupby(["regime", "method"]).apply(lambda g: f"{100*g.r3.mean():.0f}/{100*g.r5.mean():.0f}/{100*g.r10.mean():.0f} ({g.n_cfg.mean():.1f})").unstack("method")[D].reindex(REGS)
tab["Sup"] = pc(z[z.method == "bge"].groupby("regime").sup.mean()); tab["R50"] = pc(z[z.method == "bge"].groupby("regime").r50.mean())
print(tab.to_string())

rng = np.random.default_rng(0)
def paired(fr, m1, m2, col, B=5000):
    x = fr[fr.method == m1].set_index("qid")[col].astype(int); y = fr[fr.method == m2].set_index("qid")[col].astype(int)
    d = (x - y).dropna().values; ix = rng.integers(0, len(d), (B, len(d))); bs = d[ix].mean(1)
    return f"{100 * d.mean():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f},{100 * np.percentile(bs, 97.5):+5.1f}]"
print("\n-- paired (pp, bootstrap over queries) --")
for reg in REGS[1:]:
    r = z[z.regime == reg]
    print(f"{reg:9s} R@10 strat-all vs strat-50 {paired(r, 'strat-all', 'strat-50', 'r10')} | R@5 strat-100 vs strat-50 {paired(r, 'strat-100', 'strat-50', 'r5')} "
          f"| R@3 strat-25 vs strat-50 {paired(r, 'strat-25', 'strat-50', 'r3')}")

print("\n==== (iii) TEMPORAL EMBARGO (exclude precedents within delta days), %: Sup | bge R@1 | bge R@10 | CSR3 R@3 | strat-50 R@3 | strat-50 R@10 ====")
E = []
for (dl, reg), g in df.groupby(["delta", "regime"]):
    b, c, s_ = g[g.method == "bge"], g[g.method == "CSR3-bge"], g[g.method == "strat-50"]
    E.append(dict(delta=dl, regime=reg, n=b.qid.nunique(), Sup=b.sup.mean(), bge_R1=b.r1.mean(), bge_R10=b.r10.mean(),
                  CSR3_R3=c.r3.mean(), strat_R3=s_.r3.mean(), strat_R10=s_.r10.mean()))
E = pd.DataFrame(E).set_index(["regime", "delta"]).reindex(REGS, level=0)
print(pd.concat([E[["n"]], pc(E.drop(columns="n"))], axis=1).to_string())
for reg in REGS:
    a = df[(df.regime == reg) & (df.method == "bge")]
    x = a[a.delta == 0].set_index("qid").r1.astype(int); y = a[a.delta == 30].set_index("qid").r1.astype(int)
    d = (y - x).dropna()
    print(f"{reg:9s} bge R@1, delta 30 minus 0: {100 * d.mean():+5.1f} pp (n={len(d)}; queries whose R@1 hit disappears: {int((d < 0).sum())})")

print("\n==== (iv) TRAINING-FREE REGIME ROUTER (contested share among 10 nearest earlier precedents) ====")
w = z[z.method.isin(["bge", "CSR3-bge"])].pivot_table(index=["qid", "regime", "year", "p_cont"], columns="method", values="r1").reset_index()
w["cont"] = w.regime != "consensus"
def auroc(s, y):
    y = np.asarray(y, bool); r = pd.Series(s).rank().values; n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
cy = int(w.year.quantile(0.4)); dev, test = w[w.year <= cy], w[w.year > cy]
grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
route = lambda f, tau: np.where(f.p_cont >= tau, f["CSR3-bge"], f["bge"]).mean()
tau = max(grid, key=lambda t: route(dev, t))
print(f"dev n={len(dev)} test n={len(test)} | AUROC(p_cont -> contested): dev {auroc(dev.p_cont, dev.cont):.3f}, test {auroc(test.p_cont, test.cont):.3f} | tau*={tau}")
for lab, f in [("dev (<2013)", dev), ("test (>=2013)", test)]:
    reg_or = f.groupby("regime")[["bge", "CSR3-bge"]].mean().max(axis=1).mul(f.regime.value_counts()).sum() / len(f)
    print(f"{lab:13s} R@1 %: bge {100*f.bge.mean():.1f} | CSR3 {100*f['CSR3-bge'].mean():.1f} | routed(tau*) {100*route(f, tau):.1f} | "
          f"oracle regime routing {100*reg_or:.1f}")
