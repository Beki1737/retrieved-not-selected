"""E11 exploratory decomposition (no support claims; after H18 results). Questions:
  1. Are the drafts whose A1 four slots contain a match simply easy? -> B0 and S4-A0 on that subset.
  2. Does augmentation change anything on drafts where the match is NOT shown? -> complement subset.
  3. What moves BC-F1 / P(true) without moving AUROC? -> mean P(against), P(non-favour) per arm.
Resolution-cluster bootstrap, B = 2000, 95% percentile CIs; two-sided reading, exploratory."""
import json, warnings
import numpy as np, pandas as pd
from coalrag.analysis.e11_analysis import load, m3, STORE
warnings.filterwarnings("ignore")
B = 2000
D0 = json.load(open(STORE / "e11_design.json")); design = {d["qid"]: d for d in D0["drafts"]}
S = {q for q, d in design.items() if d["arms"]["A1"]["match"]}
print(f"drafts with an A1 match in the four slots: {len(S)} of {len(design)}")
print("  match slot position (A1):", pd.Series([design[q]["arms"]["A1"]["match_slot"] for q in S]).value_counts().sort_index().to_dict())
CONDS = ["B0", "S4-A0", "S4-A1", "S4-A3", "A1-match"]
for m in sorted(p.name[4:-6] for p in STORE.glob("e11_*.jsonl")):
    d = load(m); g = {c: x.set_index(["qid", "nation"]) for c, x in d.groupby("cond")}
    keys = sorted(set.intersection(*[set(g[c].index) for c in ["B0", "S4-A0", "S4-A1", "S4-A3"]]))
    y = np.array([g["B0"].loc[k, "true"] for k in keys]); qid = np.array([k[0] for k in keys]); qs = np.array([g["B0"].loc[k, "qset"] for k in keys])
    P = {c: g[c].loc[keys, ["pY", "pN", "pA"]].values.astype(float) for c in ["B0", "S4-A0", "S4-A1", "S4-A3"]}
    X = np.full((len(keys), 3), np.nan)
    for i, k in enumerate(keys):
        if k in g["A1-match"].index: X[i] = g["A1-match"].loc[k, ["pY", "pN", "pA"]].values.astype(float)
    P["A1-match"] = X
    inS = np.array([q in S for q in qid])
    print(f"\n==== {m} ====")
    print(f"  subset S (A1 match shown): rows {inS.sum()}, against rows {int((y[inS] == 'N').sum())}, regimes {pd.Series(qs[inS]).value_counts().to_dict()}")
    print(f"  complement C:             rows {(~inS).sum()}, against rows {int((y[~inS] == 'N').sum())}, regimes {pd.Series(qs[~inS]).value_counts().to_dict()}")
    rng = np.random.default_rng(0)
    for lab, mask, conds, pairs in [("S", inS, CONDS[:3] + ["A1-match"], [("S4-A1", "B0"), ("A1-match", "B0"), ("S4-A1", "S4-A0"), ("A1-match", "S4-A1")]),
                                    ("C", ~inS, CONDS[:4], [("S4-A1", "B0"), ("S4-A1", "S4-A0"), ("S4-A1", "S4-A3")])]:
        rows = np.where(mask)[0]; uq = np.unique(qid[rows]); pos = {u: rows[qid[rows] == u] for u in uq}
        def st(ix): return {c: m3(y[ix], P[c][ix]) for c in conds}
        pt = st(rows); bs = [st(np.concatenate([pos[u] for u in rng.choice(uq, len(uq))])) for _ in range(B)]
        for j, met in enumerate(["AUROC(N)", "P(true)"]):
            jj = 0 if j == 0 else 2
            print(f"  [{lab}] {met}: " + " | ".join(f"{c} {pt[c][jj]:.3f} [{np.nanpercentile([b[c][jj] for b in bs], 2.5):.3f}, {np.nanpercentile([b[c][jj] for b in bs], 97.5):.3f}]" for c in conds))
            for a, b in pairs:
                dd = np.array([x[a][jj] - x[b][jj] for x in bs]); dd = dd[~np.isnan(dd)]
                ci = f"[{np.percentile(dd, 2.5):+.3f}, {np.percentile(dd, 97.5):+.3f}]" if len(dd) else "[n/a]"
                print(f"      {a} - {b}: {pt[a][jj] - pt[b][jj]:+.3f} {ci}")
    print("  mean P(against) / P(non-favour) / share of argmax 'against' over all rows; truth: against share %.3f, non-favour %.3f" % ((y == "N").mean(), (y != "Y").mean()))
    for c in ["B0", "S4-A0", "S4-A1", "S4-A3"]:
        print(f"    {c:6s} {P[c][:, 1].mean():.3f} / {(1 - P[c][:, 0]).mean():.3f} / {(P[c].argmax(1) == 1).mean():.3f}")
