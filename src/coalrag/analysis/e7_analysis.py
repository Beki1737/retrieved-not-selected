"""E7 analysis (pre-registered H13-H14, D81; aggregation rules fixed before any E7 output exists).
From the four single-precedent predictions P_i (S-slot1..4, i = stratified rank) and the selector probabilities w_i:
  S-top1    P_1 alone (the most similar stratified precedent)
  S-mean    uniform mean of P_i
  S-rank    weights proportional to 1/i
  S-llmsel  weights w_i from the model's own choice of the closest precedent (texts only, no votes)   <- H13
  S-perfect the P_i of the slot whose five-vote configuration equals the draft's; S-mean if no slot matches  <- bound, H14
Decomposition: D-oracle - D-strat4 = (S-perfect - D-strat4) [selection] + (D-oracle - S-perfect) [coverage + presentation].
Tests: one-sided resolution-cluster paired bootstrap; Holm across models for the H13 AUROC test.
Usage: python -m coalrag.analysis.e7_analysis [--B 2000]"""
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
from coalrag.analysis.e4_metrics import metrics
from coalrag.analysis.e5_analysis import load, stat, align, table

STORE = Path(os.environ["PROJ"]) / "results" / "store"; K = ["qid", "nation"]; LAB = np.array(["Y", "N", "A"]); P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]


def rows_from(base, P, cond):
    d = base.copy(); d["cond"] = cond; d["pY"], d["pN"], d["pA"] = P[:, 0], P[:, 1], P[:, 2]
    d["ptrue"] = P[np.arange(len(d)), d.true.map({"Y": 0, "N": 1, "A": 2}).values]
    d["pred"] = LAB[P.argmax(1)]; d["hit"] = (d.pred == d.true).astype(float); return d


def pboot(a, b, name, B, seed=0):
    a, b = align([a, b]); f = stat(name); q = a.qid.values; uq = np.unique(q)
    pos = {u: np.where(q == u)[0] for u in uq}; rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        ix = np.concatenate([pos[u] for u in rng.choice(uq, len(uq))]); bs.append(f(a.iloc[ix]) - f(b.iloc[ix]))
    bs = np.array(bs, float); bs = bs[~np.isnan(bs)]
    return f(a) - f(b), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (1 + np.sum(bs <= 0)) / (len(bs) + 1)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o): run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


def build(m):
    S = load(m, "select"); main = load(m, "main")
    sel = {}
    p = STORE / f"e7_{m}_selector.jsonl"
    for r in map(json.loads, open(p)):
        if not r.get("error"): sel[r["qid"]] = (r["ids"], np.array(r["w"], float))
    from coalrag.retrieval.configA import ConfigA
    vot = ConfigA().vot
    out = {c: [] for c in ("S-top1", "S-mean", "S-rank", "S-llmsel", "S-perfect")}; meta = []
    for (q, n), g in S.groupby(K):
        g = g.sort_values("slot"); P = g[["pY", "pN", "pA"]].values.astype(float); k = len(g); base = g.iloc[[0]][["qid", "symbol", "qset", "nation", "true"]]
        ids = [r[0] for r in g.retrieved]; cfgq = tuple(vot[q][x] for x in P5)
        match = [i for i, d in enumerate(ids) if tuple(vot[d][x] for x in P5) == cfgq]
        wr = 1 / np.arange(1, k + 1); wr = wr / wr.sum()
        add = {"S-top1": P[0], "S-mean": P.mean(0), "S-rank": wr @ P, "S-perfect": P[match[0]] if match else P.mean(0)}
        if q in sel and len(sel[q][1]) == k: add["S-llmsel"] = (sel[q][1] / sel[q][1].sum()) @ P
        for c, v in add.items(): out[c].append((base, v))
        meta.append(dict(qid=q, nation=n, k=k, covered=bool(match), match_slot=(match[0] + 1) if match else None,
                         sel_argmax=int(np.argmax(sel[q][1])) + 1 if q in sel else None, w1=float(sel[q][1][0]) if q in sel else np.nan))
    agg = [rows_from(pd.concat([b for b, _ in v], ignore_index=True), np.vstack([x for _, x in v]), c) for c, v in out.items() if v]
    ref = main[main.cond.isin(["B0-std", "D-random-coal", "D-strat4-coal", "D-oracle-coal"])]
    return pd.concat([ref] + agg, ignore_index=True), pd.DataFrame(meta)


def main(B):
    models = sorted(p.name[3:-13] for p in STORE.glob("e4_*_select.jsonl"))
    tests, H13 = [], []
    for m in models:
        d, meta = build(m); d = pd.concat(align([g for _, g in d.groupby("cond")]))
        conds = ["B0-std", "D-random-coal", "D-strat4-coal", "S-top1", "S-mean", "S-rank", "S-llmsel", "S-perfect", "D-oracle-coal"]
        print(f"\n==== {m} | keys {d.groupby('cond').size().min()} ====")
        with pd.option_context("display.width", 200): print(table(d, [c for c in conds if c in set(d.cond)]).to_string())
        cov = meta.drop_duplicates("qid"); hit = cov[cov.covered]
        print(f"coverage: a stratified slot matches the draft's configuration for {cov.covered.mean():.1%} of drafts | "
              f"selector picks that slot in {np.mean(hit.sel_argmax == hit.match_slot):.1%} (slot-1 rule: {np.mean(hit.match_slot == 1):.1%}; "
              f"chance {np.mean(1 / hit.k):.1%}) | mean selector weight on slot 1 {cov.w1.mean():.2f}")
        g = {c: x for c, x in d.groupby("cond")}
        for h, k, a, b in [("H13", "aucN", "S-llmsel", "D-strat4-coal"), ("H13", "macroF1_BC", "S-llmsel", "D-strat4-coal"),
                           ("E", "aucN", "S-llmsel", "S-rank"), ("E", "aucN", "S-mean", "D-strat4-coal"),
                           ("H14", "aucN", "S-perfect", "D-strat4-coal"), ("E", "aucN", "D-oracle-coal", "S-perfect")]:
            if a in g and b in g:
                dl, lo, hi, p = pboot(g[a], g[b], k, B)
                tests.append(dict(model=m, hyp=h, metric=k, a=a, b=b, delta=dl, ci=f"[{lo:+.3f}, {hi:+.3f}]", p=p))
        if {"D-oracle-coal", "D-strat4-coal", "S-perfect"} <= set(g):
            f = stat("aucN"); o, s4, sp = f(g["D-oracle-coal"]), f(g["D-strat4-coal"]), f(g["S-perfect"])
            print(f"decomposition of the oracle gap in AUROC(N): total {o - s4:+.3f} = selection {sp - s4:+.3f} + coverage/presentation {o - sp:+.3f}")
    t = pd.DataFrame(tests)
    if len(t):
        i = t.index[(t.hyp == "H13") & (t.metric == "aucN")]; t["p_holm"] = np.nan; t.loc[i, "p_holm"] = holm(t.loc[i, "p"])
        t["delta"] = t.delta.map("{:+.3f}".format); t["p"] = t.p.map("{:.4f}".format)
        print("\n==== E7 tests (one-sided; Holm across models for H13 AUROC) ====\n" + t.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); main(ap.parse_args().B)
