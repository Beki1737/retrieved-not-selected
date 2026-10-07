"""E11 analysis (pre-registered H18, wave 6; fixed before any E11 output).
  H18a  AUROC(N), four stratified: A1 (+100 similar dissent-bearing) > A3 (+100 similar unanimous)
  H18b  AUROC(N), four stratified: A1 > A0 (outcome-filtered pool, no augmentation)
  H18c  drafts whose A1 four slots contain a match: AUROC(N) of the match alone > the four together
Paired resolution-cluster bootstrap (B = 2000), one-sided, Holm within model over H18a-c.
Exploratory (no support claim): A1 vs A2 (mean over the three random seeds), A1 vs B0, A0 vs B0, A3 vs A0, BC-F1 and P(true)
contrasts, match alone vs four together in A0 and A3, coverage by arm (model-independent manipulation check).
Usage: python -m coalrag.analysis.e11_analysis [--B 2000]"""
import argparse, json, warnings
import numpy as np, pandas as pd
from coalrag.analysis.e4_metrics import metrics
from coalrag.analysis.e7_analysis import STORE, holm

ARMS = ["A0", "A1", "A2s0", "A2s1", "A2s2", "A3"]; MATCH_ARMS = ["A0", "A1", "A3"]
warnings.filterwarnings("ignore")
LI = {"Y": 0, "N": 1, "A": 2}; MINSUB = 10


def load(m):
    rows = {}
    for r in map(json.loads, open(STORE / f"e11_{m}.jsonl")):
        if r.get("error") is None and r.get("pY") is not None: rows[(r["qid"], r["nation"], r["cond"])] = r
    return pd.DataFrame(list(rows.values()))


def build(m, design):
    d = load(m)
    if not len(d): return None
    g = {c: x.set_index(["qid", "nation"]) for c, x in d.groupby("cond")}
    need = ["B0"] + [f"S4-{a}" for a in ARMS]
    miss = [c for c in need if c not in g]
    if miss: print(f"  {m}: missing conditions {miss}"); return None
    keys = sorted(set.intersection(*[set(g[c].index) for c in need]))
    y = np.array([g["B0"].loc[k, "true"] for k in keys]); qid = np.array([k[0] for k in keys])
    P = {c: g[c].loc[keys, ["pY", "pN", "pA"]].values.astype(float) for c in need}
    M = {}
    for a in MATCH_ARMS:
        X = np.full((len(keys), 3), np.nan); c = f"{a}-match"
        if c in g:
            have = set(g[c].index)
            for i, k in enumerate(keys):
                if design[k[0]]["arms"][a]["match"] is not None and k in have: X[i] = g[c].loc[k, ["pY", "pN", "pA"]].values.astype(float)
        M[a] = (~np.isnan(X[:, 0]), X)
    return dict(keys=keys, y=y, qid=qid, P=P, M=M)


def m3(y, X):
    if len(y) < MINSUB or len(np.unique(y)) < 2: return np.nan, np.nan, np.nan
    r = metrics(y, X); return r["aucN"], r["macroF1_BC"], float(X[np.arange(len(y)), [LI[v] for v in y]].mean())


def stats(D, ix):
    y = D["y"][ix]; o = {}
    for c, X in D["P"].items():
        o[f"AUROC {c}"], o[f"BC-F1 {c}"], o[f"P(true) {c}"] = m3(y, X[ix])
    for met in ("AUROC", "BC-F1", "P(true)"):
        o[f"{met} S4-A2"] = float(np.mean([o[f"{met} S4-A2s{s}"] for s in (0, 1, 2)]))
    for a, (mask, X) in D["M"].items():
        s = ix[mask[ix]]; ys = D["y"][s]
        o[f"AUROC {a}-match [sub]"], o[f"BC-F1 {a}-match [sub]"], o[f"P(true) {a}-match [sub]"] = m3(ys, X[s])
        o[f"AUROC S4-{a} [sub {a}]"], o[f"BC-F1 S4-{a} [sub {a}]"], o[f"P(true) S4-{a} [sub {a}]"] = m3(ys, D["P"][f"S4-{a}"][s])
        o[f"rows [sub {a}]"] = float(len(s))
    return o


TESTS = [("H18a", "AUROC S4-A1", "AUROC S4-A3"), ("H18b", "AUROC S4-A1", "AUROC S4-A0"), ("H18c", "AUROC A1-match [sub]", "AUROC S4-A1 [sub A1]"),
         ("E", "AUROC S4-A1", "AUROC S4-A2"), ("E", "AUROC S4-A1", "AUROC B0"), ("E", "AUROC S4-A0", "AUROC B0"), ("E", "AUROC S4-A3", "AUROC S4-A0"),
         ("E", "BC-F1 S4-A1", "BC-F1 S4-A3"), ("E", "BC-F1 S4-A1", "BC-F1 S4-A0"), ("E", "P(true) S4-A1", "P(true) S4-A3"),
         ("E", "P(true) S4-A1", "P(true) S4-A0"), ("E", "P(true) A1-match [sub]", "P(true) S4-A1 [sub A1]"),
         ("E", "AUROC A0-match [sub]", "AUROC S4-A0 [sub A0]"), ("E", "AUROC A3-match [sub]", "AUROC S4-A3 [sub A3]")]


def coverage(drafts, B, rng):
    C = {a: np.array([d["arms"][a]["match"] is not None for d in drafts], float) for a in ARMS}
    S = {a: np.array([d["arms"][a]["sup"] for d in drafts], float) for a in ARMS}
    C["A2"] = np.mean([C[f"A2s{s}"] for s in (0, 1, 2)], 0); S["A2"] = np.mean([S[f"A2s{s}"] for s in (0, 1, 2)], 0)
    n = len(drafts); ix = rng.integers(0, n, (B, n))
    print(f"\n==== Manipulation check (model-independent; drafts n = {n}) ====")
    for a in ["A0", "A1", "A2", "A3"]: print(f"  {a}: four slots contain a match {C[a].mean():.3f} | match anywhere in pool (Sup) {S[a].mean():.3f}")
    for a, b in [("A1", "A0"), ("A1", "A3"), ("A1", "A2")]:
        dd = C[a] - C[b]; bs = dd[ix].mean(1)
        print(f"  coverage {a} - {b}: {dd.mean():+.3f} [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]")


def main(B):
    D0 = json.load(open(STORE / "e11_design.json")); design = {d["qid"]: d for d in D0["drafts"]}
    coverage(D0["drafts"], B, np.random.default_rng(0))
    models = sorted(p.name[4:-6] for p in STORE.glob("e11_*.jsonl")); rows = []
    for m in models:
        D = build(m, design)
        if D is None or not len(D["keys"]): continue
        uq = np.unique(D["qid"]); pos = {u: np.where(D["qid"] == u)[0] for u in uq}; rng = np.random.default_rng(0)
        full = np.arange(len(D["y"])); pt = stats(D, full)
        bs = [stats(D, np.concatenate([pos[u] for u in rng.choice(uq, len(uq))])) for _ in range(B)]
        print(f"\n==== {m} | drafts {len(uq)} | rows {len(full)} | against rows {int((D['y'] == 'N').sum())} | point estimate [95% CI] ====")
        for k, v in pt.items():
            if k.startswith("rows"): print(f"  {k:34s} {int(v)}"); continue
            col = np.array([b[k] for b in bs], float)
            print(f"  {k:34s} {v:.3f}  [{np.nanpercentile(col, 2.5):.3f}, {np.nanpercentile(col, 97.5):.3f}]")
        for h, a, b in TESTS:
            d = np.array([x[a] - x[b] for x in bs], float); d = d[~np.isnan(d)]
            if not len(d) or np.isnan(pt[a] - pt[b]):
                rows.append(dict(model=m, hyp=h, a=a, b=b, delta=np.nan, ci="n/a", p=np.nan)); continue
            rows.append(dict(model=m, hyp=h, a=a, b=b, delta=pt[a] - pt[b], ci=f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]",
                             p=(1 + np.sum(d <= 0)) / (len(d) + 1)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for m in t.model.unique():
            i = t.index[(t.model == m) & t.hyp.str.startswith("H18") & t.p.notna()]
            if len(i): t.loc[i, "p_holm"] = holm(t.loc[i, "p"])
        t["delta"] = t.delta.map(lambda x: f"{x:+.3f}" if pd.notna(x) else "n/a"); t["p"] = t.p.map(lambda x: f"{x:.4f}" if pd.notna(x) else "n/a")
        t["p_holm"] = t.p_holm.map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        with pd.option_context("display.width", 250, "display.max_colwidth", 40, "display.max_rows", 500):
            print("\n==== E11 tests (one-sided; Holm within model over H18a-c; E = exploratory) ====\n" + t.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); main(ap.parse_args().B)
