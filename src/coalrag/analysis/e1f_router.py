"""E1f follow-up (D67): training-free regime router with a time split inside Choi's 2013+ documents
(dev = queries up to the 40th-percentile year; test = later). Reads results/store/e1f_depth_embargo.parquet."""
import os, numpy as np, pandas as pd
df = pd.read_parquet(os.path.expandvars("$PROJ/results/store/e1f_depth_embargo.parquet")); z = df[df.delta == 0]
w = z[z.method.isin(["bge", "CSR3-bge"])].pivot_table(index=["qid", "regime", "year", "p_cont"], columns="method", values="r1").reset_index()
w["cont"] = w.regime != "consensus"
print("queries per year:", w.year.value_counts().sort_index().to_dict())
cy = int(w.year.quantile(0.4)); dev, test = w[w.year <= cy], w[w.year > cy]
def auroc(s, y):
    y = np.asarray(y, bool); r = pd.Series(np.asarray(s)).rank().values; n1, n0 = y.sum(), (~y).sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan
route = lambda f, t: np.where(f.p_cont >= t, f["CSR3-bge"], f["bge"])
grid = np.round(np.arange(0.1, 1.01, 0.1), 1); tau = max(grid, key=lambda t: route(dev, t).mean())
print(f"dev = years <= {cy} (n={len(dev)}), test = later (n={len(test)}) | tau* = {tau} (tuned on dev only)")
for lab, f in [("dev", dev), ("test", test)]:
    r = route(f, tau); c = f.cont.values
    reg_or = f.groupby("regime")[["bge", "CSR3-bge"]].mean().max(axis=1).mul(f.regime.value_counts()).sum() / len(f)
    print(f"{lab:4s} AUROC(p_cont -> contested) {auroc(f.p_cont, f.cont):.3f} | R@1 all: bge {100*f.bge.mean():.1f} CSR3 {100*f['CSR3-bge'].mean():.1f} "
          f"routed {100*r.mean():.1f} oracle-regime {100*reg_or:.1f} | contested only: bge {100*f.bge[c].mean():.1f} CSR3 {100*f['CSR3-bge'][c].mean():.1f} "
          f"routed {100*r[c].mean():.1f} | consensus routed away {100*(f.p_cont[~c] >= tau).mean():.1f}%")
s = z[z.method == "strat-50"].groupby("regime").r3.mean().mul(100).round(1).to_dict()
print("reference, strat-50 R@3 by regime (no routing needed):", s)
