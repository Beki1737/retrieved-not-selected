"""E7c (exploratory re-analysis of the pre-registered E7 outputs, D99; design review item). Every selection strategy shows ONE
stratified precedent alone (hard pick, no averaging), so selection is not mixed with an ensemble effect:
  ground truth   the slot whose configuration equals the draft's
  model          argmax of the model's own selector weights (texts only)
  first-ranked   slot 1 (most similar stratified precedent)
  random         uniform pick (point = mean of 200 draws; interval = bootstrap with a fresh draw per resample)
  vote-lift      the H16 structure-aware rule
Primary pool: drafts where a matching slot exists (so no fallback is needed). Secondary: all drafts, with the SAME fallback (slot 1)
for the ground-truth rule whenever no slot matches. Reference rows on the same keys: no evidence, four shown together, oracle.
Usage: python -m coalrag.analysis.e7c_harmonized [--B 2000]"""
import argparse, json
import numpy as np, pandas as pd
from coalrag.analysis.e4_metrics import metrics
from coalrag.analysis.e5_analysis import load
from coalrag.analysis.e7_analysis import STORE, P5
from coalrag.analysis.e7b_structure import lift, conf

S_ORDER = ["no evidence", "four shown together", "ground truth", "model", "first-ranked", "random", "vote-lift", "oracle precedent"]


def build(m):
    S = load(m, "select"); main = load(m, "main"); sel = {}
    for r in map(json.loads, open(STORE / f"e7_{m}_selector.jsonl")):
        if not r.get("error"): sel[r["qid"]] = np.array(r["w"], float)
    ref = {c: x.set_index(["qid", "nation"]) for c, x in main.groupby("cond") if c in ("B0-std", "D-strat4-coal", "D-oracle-coal")}
    R, cache = [], {}
    for (q, n), g in S.groupby(["qid", "nation"]):
        g = g.sort_values("slot")
        if len(g) != 4 or q not in sel or len(sel[q]) != 4 or any((q, n) not in ref[c].index for c in ref): continue
        ids = [r[0] for r in g.retrieved]; cq = conf(q); match = [i for i, x in enumerate(ids) if conf(x) == cq]
        if q not in cache: cache[q] = int(np.argmax(lift(q, ids)))
        R.append(dict(q=q, y=g.true.iloc[0], P=g[["pY", "pN", "pA"]].values.astype(float), match=match[0] if match else -1,
                      model=int(np.argmax(sel[q])), lift=cache[q], **{c: ref[c].loc[(q, n), ["pY", "pN", "pA"]].values.astype(float) for c in ref}))
    return R


def stat(R, rng, fallback):
    y = np.array([r["y"] for r in R]); P = np.stack([r["P"] for r in R]); i = np.arange(len(R))
    gt = np.array([r["match"] if r["match"] >= 0 else (0 if fallback else -1) for r in R])
    a = lambda X: metrics(y, X)["aucN"]
    return {"no evidence": a(np.stack([r["B0-std"] for r in R])), "four shown together": a(np.stack([r["D-strat4-coal"] for r in R])),
            "ground truth": a(P[i, gt]), "model": a(P[i, [r["model"] for r in R]]), "first-ranked": a(P[i, 0]),
            "random": a(P[i, rng.integers(0, 4, len(R))]), "vote-lift": a(P[i, [r["lift"] for r in R]]),
            "oracle precedent": a(np.stack([r["D-oracle-coal"] for r in R])),
            "selection hit: model": float(np.mean([r["model"] == r["match"] for r in R])),
            "selection hit: first-ranked": float(np.mean([r["match"] == 0 for r in R])),
            "selection hit: vote-lift": float(np.mean([r["lift"] == r["match"] for r in R]))}


def report(name, R, B, fallback):
    if not R: return
    qs = np.array([r["q"] for r in R]); uq = np.unique(qs); pos = {u: np.where(qs == u)[0] for u in uq}; rng = np.random.default_rng(0)
    pt = stat(R, rng, fallback); pt["random"] = np.mean([stat(R, np.random.default_rng(s), fallback)["random"] for s in range(200)])
    bs = [stat([R[j] for j in np.concatenate([pos[u] for u in rng.choice(uq, len(uq))])], rng, fallback) for _ in range(B)]
    print(f"\n-- {name}: {len(uq)} drafts, {len(R)} rows | AUROC(N) [95% CI]; difference from ground truth [95% CI] --")
    for k in S_ORDER + [k for k in pt if k.startswith("selection")]:
        lo, hi = np.nanpercentile([b[k] for b in bs], [2.5, 97.5]); line = f"  {k:30s} {pt[k]:.3f} [{lo:.3f}, {hi:.3f}]"
        if k in ("model", "first-ranked", "random", "vote-lift", "four shown together"):
            d = np.array([b[k] - b["ground truth"] for b in bs]); line += f" | vs ground truth {pt[k] - pt['ground truth']:+.3f} [{np.nanpercentile(d, 2.5):+.3f}, {np.nanpercentile(d, 97.5):+.3f}]"
        print(line)


def main(B):
    for m in sorted(p.name[3:-13] for p in STORE.glob("e4_*_select.jsonl")):
        R = build(m)
        print(f"\n==== {m} ====")
        report("drafts with a matching slot (no fallback needed)", [r for r in R if r["match"] >= 0], B, False)
        report("all drafts, slot-1 fallback for the ground-truth rule", R, B, True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); main(ap.parse_args().B)
