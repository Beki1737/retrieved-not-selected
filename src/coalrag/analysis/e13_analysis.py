"""E13 analysis (pre-registered H20, wave 8). Query-cluster bootstrap, B = 2000, one-sided; Holm within model over H20a-d.
  H20a match alone > four together (mean over positions)      H20b match alone > own pick alone (mean over orderings)
  H20c four together > none                                   H20d four together, match first > match last
Exploratory: own pick vs first-ranked and random; selection hit vs 0.25 and by position; selection appropriateness (judge J1,
votes hidden): own pick vs random pick, match vs distractors, judge's pick; BC-F1 and P(true) for the main conditions.
Usage: python -m coalrag.analysis.e13_analysis [--B 2000] [--judge llama-3.3-70b-fp8]"""
import argparse, json, warnings
import numpy as np, pandas as pd
from coalrag.analysis.e4_metrics import metrics
from coalrag.analysis.e7_analysis import STORE, holm
warnings.filterwarnings("ignore")
POS = (1, 2, 3, 4); LI = {"Y": 0, "N": 1, "A": 2}
NEED = ["G-B0", "G-orc", "G-d1", "G-d2", "G-d3"] + [f"G4-p{p}" for p in POS]


def build(m, D):
    f = STORE / f"e13_{m}_gen.jsonl"
    if not f.exists(): return None
    R, S = {}, {}
    for r in map(json.loads, open(f)):
        if r.get("error") is not None: continue
        if "pos" in r: S[(r["qid"], r["pos"])] = (r["ids"], np.array(r["w"], float))
        else: R[(r["qid"], r["nation"], r["cond"])] = r
    Qd = {q["rid"]: q for q in D["queries"]}
    keys = sorted({(q, n) for (q, n, c) in R if all((q, n, cc) in R for cc in NEED) and all((q, p) in S for p in POS)})
    if not keys: return None
    y = np.array([R[(q, n, "G-B0")]["true"] for q, n in keys]); qid = np.array([k[0] for k in keys])
    P = {c: np.array([[R[(q, n, c)][x] for x in ("pY", "pN", "pA")] for q, n in keys], float) for c in NEED}
    single = np.stack([P["G-orc"], P["G-d1"], P["G-d2"], P["G-d3"]], 1)
    pick = np.zeros((len(keys), 4), int)
    for i, (q, n) in enumerate(keys):
        idx = {c["rid"]: j for j, c in enumerate(Qd[q]["cands"])}
        for p in POS: ids, w = S[(q, p)]; pick[i, p - 1] = idx[ids[int(np.argmax(w))]]
    first = np.array([int(np.argmin([c["rank"] for c in Qd[q]["cands"]])) for q, n in keys])
    return dict(keys=keys, y=y, qid=qid, P=P, single=single, pick=pick, first=first)


def m3(y, X):
    if len(np.unique(y)) < 2: return np.nan, np.nan, np.nan
    r = metrics(y, X); return r["aucN"], r["macroF1_BC"], float(X[np.arange(len(y)), [LI[v] for v in y]].mean())


def stats(D, ix, rng):
    y, Sg, pick = D["y"][ix], D["single"][ix], D["pick"][ix]; r = np.arange(len(ix)); o = {}
    for lab, X in (("none", D["P"]["G-B0"][ix]), ("match alone", D["P"]["G-orc"][ix])):
        o[f"AUROC {lab}"], o[f"BC-F1 {lab}"], o[f"P(true) {lab}"] = m3(y, X)
    four = [m3(y, D["P"][f"G4-p{p}"][ix]) for p in POS]
    for j, met in enumerate(("AUROC", "BC-F1", "P(true)")): o[f"{met} four together (mean)"] = float(np.mean([f[j] for f in four]))
    for p in POS: o[f"AUROC four, match at {p}"] = four[p - 1][0]
    own = [m3(y, Sg[r, pick[:, p - 1]]) for p in POS]
    for j, met in enumerate(("AUROC", "BC-F1", "P(true)")): o[f"{met} own pick alone (mean)"] = float(np.mean([f[j] for f in own]))
    o["AUROC first-ranked alone"] = m3(y, Sg[r, D["first"][ix]])[0]
    o["AUROC random pick alone"] = m3(y, Sg[r, rng.integers(0, 4, len(ix))])[0]
    o["hit mean"] = float(np.mean(pick == 0))
    for p in POS: o[f"hit, match at {p}"] = float(np.mean(pick[:, p - 1] == 0))
    return o


TESTS = [("H20a", "AUROC match alone", "AUROC four together (mean)"), ("H20b", "AUROC match alone", "AUROC own pick alone (mean)"),
         ("H20c", "AUROC four together (mean)", "AUROC none"), ("H20d", "AUROC four, match at 1", "AUROC four, match at 4"),
         ("E", "AUROC own pick alone (mean)", "AUROC first-ranked alone"), ("E", "AUROC own pick alone (mean)", "AUROC random pick alone"),
         ("E", "AUROC match alone", "AUROC none"), ("E", "hit mean", None), ("E", "hit, match at 1", "hit, match at 4")]


def appropriateness(D, judge, models, B, rng):
    f = STORE / f"e13_{judge}_judge.jsonl"
    if not f.exists(): print("\n(no E13 judge output yet)"); return
    O = {}
    for r in map(json.loads, open(f)):
        if r.get("error") is None and r["q"] == "O" and r.get("p"): O[(r["qid"], r["cid"])] = float(r["p"][1])
    A = {q["rid"]: np.array([O.get((q["rid"], c["rid"]), np.nan) for c in q["cands"]]) for q in D["queries"]}
    A = {k: v for k, v in A.items() if not np.isnan(v).any()}; qs = sorted(A); App = {k: (v >= .5).astype(float) for k, v in A.items()}
    bs = lambda f: np.array([f(rng.choice(qs, len(qs))) for _ in range(B)])
    ci = lambda x: f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"
    f1 = lambda s: np.mean([App[q][0] - App[q][1:].mean() for q in s])
    print(f"\n==== Selection appropriateness (judge {judge}, votes hidden; {len(qs)} queries) ====")
    print(f"match appropriate {np.mean([App[q][0] for q in qs]):.3f} | distractors {np.mean([App[q][1:].mean() for q in qs]):.3f} | "
          f"match - distractors {f1(qs):+.3f} {ci(bs(f1))} | queries with 2+ appropriate {np.mean([App[q].sum() >= 2 for q in qs]):.3f}")
    jp = {q: int(np.argmax(A[q])) for q in qs}; print(f"judge's pick hits the match in {np.mean([jp[q] == 0 for q in qs]):.3f} (chance 0.250)")
    for m in models:
        f = STORE / f"e13_{m}_gen.jsonl"
        if not f.exists(): continue
        S = {}
        for r in map(json.loads, open(f)):
            if r.get("error") is None and "pos" in r: S[(r["qid"], r["pos"])] = (r["ids"], np.array(r["w"], float))
        Qd = {q["rid"]: q for q in D["queries"]}; pk = {}
        for q in qs:
            if all((q, p) in S for p in POS):
                idx = {c["rid"]: j for j, c in enumerate(Qd[q]["cands"])}; pk[q] = [idx[S[(q, p)][0][int(np.argmax(S[(q, p)][1]))]] for p in POS]
        s2 = [q for q in qs if q in pk]
        if not s2: continue
        g = lambda s: np.mean([np.mean([App[q][j] for j in pk[q]]) - App[q].mean() for q in s])
        b = np.array([g(rng.choice(s2, len(s2))) for _ in range(B)])
        ma = [q for q in s2 if App[q][0] == 1]; mn = [q for q in s2 if App[q][0] == 0]
        print(f"  {m:28s} own pick appropriate {np.mean([np.mean([App[q][j] for j in pk[q]]) for q in s2]):.3f} vs random {np.mean([App[q].mean() for q in s2]):.3f}: "
              f"{g(s2):+.3f} {ci(b)} | hit when match appropriate {np.mean([np.mean(np.array(pk[q]) == 0) for q in ma]) if ma else float('nan'):.3f} (n {len(ma)}), "
              f"when not {np.mean([np.mean(np.array(pk[q]) == 0) for q in mn]) if mn else float('nan'):.3f} (n {len(mn)})")


def main(B, judge):
    D = json.load(open(STORE / "e13_design.json")); rows = []
    models = sorted(p.name[4:-10] for p in STORE.glob("e13_*_gen.jsonl"))
    for m in models:
        X = build(m, D)
        if X is None: print(f"{m}: incomplete"); continue
        uq = np.unique(X["qid"]); pos = {u: np.where(X["qid"] == u)[0] for u in uq}; rng = np.random.default_rng(0)
        full = np.arange(len(X["y"])); pt = stats(X, full, rng)
        pt["AUROC random pick alone"] = np.nanmean([stats(X, full, np.random.default_rng(s))["AUROC random pick alone"] for s in range(200)])
        bs = [stats(X, np.concatenate([pos[u] for u in rng.choice(uq, len(uq))]), rng) for _ in range(B)]
        print(f"\n==== {m} | queries {len(uq)} | rows {len(full)} | against rows {int((X['y'] == 'N').sum())} | point [95% CI] ====")
        for k, v in pt.items():
            col = np.array([b[k] for b in bs], float); print(f"  {k:36s} {v:.3f}  [{np.nanpercentile(col, 2.5):.3f}, {np.nanpercentile(col, 97.5):.3f}]")
        for h, a, b in TESTS:
            d = np.array([x[a] - (x[b] if b else 0.25) for x in bs], float); d = d[~np.isnan(d)]
            rows.append(dict(model=m, hyp=h, a=a, b=b or "chance 0.25", delta=pt[a] - (pt[b] if b else .25),
                             ci=f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]", p=(1 + np.sum(d <= 0)) / (len(d) + 1)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for m in t.model.unique():
            i = t.index[(t.model == m) & t.hyp.str.startswith("H20")]; t.loc[i, "p_holm"] = holm(t.loc[i, "p"])
        t["delta"] = t.delta.map("{:+.3f}".format); t["p"] = t.p.map("{:.4f}".format)
        with pd.option_context("display.width", 250, "display.max_colwidth", 40, "display.max_rows", 200):
            print("\n==== E13 tests (one-sided; Holm within model over H20a-d; E = exploratory) ====\n" + t.to_string(index=False))
    appropriateness(D, judge, models, B, np.random.default_rng(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--judge", default="llama-3.3-70b-fp8")
    a = ap.parse_args(); main(a.B, a.judge)
