"""Cross-model LLM summary (pre-registered H5-H7, D76). For every model with a complete main run:
  table 1: condition x model, macro-F1 after Batch Calibration (BC) / AUROC of P(against) / Yes-rate gap / No-recall
  table 2: pre-registered tests, one-sided resolution-cluster paired bootstrap (B draws), Holm within each model
Writes results/tables/llm_crossmodel.md and results/tables/llm_tests.csv.
Usage: python -m coalrag.analysis.summary [--B 2000] [--models m1,m2]"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from coalrag.analysis.e4_metrics import metrics, add_cad

PROJ = Path(os.environ["PROJ"]); STORE = PROJ / "results" / "store"; OUT = PROJ / "results" / "tables"; K = ["qid", "nation"]
BL = ["BL-majority", "BL-nation-all", "BL-nation-nonadopted", "BL-copy-bge1", "BL-knn10", "BL-strat4-vote"]
CONDS = BL + ["B0-choi", "B0-std", "CF-LTorig", "CF-LT", "CF-bge", "CF-CSR3", "D-bge-text", "D-bge-target", "D-bge-coal",
         "D-random-coal", "D-oracle-coal", "D-top4-coal", "D-strat4-target", "D-strat4-coal",
         "D-random-coal+CAD1", "D-oracle-coal+CAD1", "D-strat4-coal+CAD1"]
TESTS = [  # (hypothesis, metric, condition a, condition b): one-sided H1: metric(a) > metric(b)
    ("H5", "aucN", "D-strat4-coal", "B0-std"), ("H5", "aucN", "D-oracle-coal", "B0-std"),
    ("H5", "macroF1_BC", "D-oracle-coal", "D-random-coal"),
    ("H6", "macroF1_BC", "D-strat4-coal", "B0-std"), ("H6", "macroF1_BC", "D-oracle-coal", "B0-std"),
    ("H7", "macroF1_BC", "D-strat4-coal+CAD1", "D-strat4-coal"), ("H7", "macroF1_BC", "D-oracle-coal+CAD1", "D-oracle-coal"),
    ("H7-control", "macroF1_BC", "D-random-coal+CAD1", "D-random-coal"),
    ("E-baseline", "macroF1_BC", "B0-std", "BL-nation-nonadopted"), ("E-baseline", "aucN", "B0-std", "BL-nation-nonadopted"),
    ("E-baseline", "macroF1_BC", "D-strat4-coal", "BL-strat4-vote"), ("E-baseline", "aucN", "D-strat4-coal", "BL-knn10"),
    ("E-baseline", "macroF1_BC", "D-oracle-coal", "BL-knn10")]


def load(m):
    p = STORE / f"e4_{m}_main.jsonl"
    d = pd.read_json(p, lines=True); d = d[d.error.isna() & d.true.isin(["Y", "N", "A"])].drop_duplicates(K + ["cond"], keep="last")
    d = add_cad(d, "B0-std")
    bl = STORE / "e4_baselines_main.jsonl"
    if bl.exists():  # D79: non-LLM baselines on the same (draft, nation) rows
        b = pd.read_json(bl, lines=True); d = pd.concat([d, b[b.true.isin(["Y", "N", "A"])]], ignore_index=True)
    keys = set.intersection(*[set(zip(g.qid, g.nation)) for _, g in d.groupby("cond")])
    return d[[k in keys for k in zip(d.qid, d.nation)]].sort_values(["cond"] + K).reset_index(drop=True), len(keys)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i])); adj[i] = run
    return adj


def main(B, models):
    models = models or sorted(m for m in (p.name[3:-11] for p in STORE.glob("e4_*_main.jsonl")) if m != "baselines")
    OUT.mkdir(parents=True, exist_ok=True); T1, T2 = [], []
    for m in models:
        d, nk = load(m); g = {c: x for c, x in d.groupby("cond")}
        qids = np.array(sorted(d.qid.unique())); rng = np.random.default_rng(0)
        pos = {c: {q: np.where(g[c].qid.values == q)[0] for q in qids} for c in g}
        pt = {c: metrics(g[c].true.values, g[c][["pY", "pN", "pA"]].values.astype(float)) for c in g}
        for c in CONDS:
            if c in pt: T1.append(dict(model=m, cond=c, **{k: pt[c][k] for k in ("macroF1", "macroF1_BC", "aucN", "yes_gap", "no_recall", "acc")}))
        need = sorted({c for _, _, a, b in TESTS for c in (a, b) if c in g}); boot = {c: [] for c in need}
        for _ in range(B):
            s = rng.choice(qids, len(qids))
            for c in need:
                ix = np.concatenate([pos[c][q] for q in s])
                boot[c].append(metrics(g[c].true.values[ix], g[c][["pY", "pN", "pA"]].values[ix].astype(float)))
        rows = []
        for h, k, a, b in TESTS:
            if a not in g or b not in g: continue
            db = np.array([x[k] - y[k] for x, y in zip(boot[a], boot[b])], float); db = db[~np.isnan(db)]
            rows.append(dict(model=m, hyp=h, metric=k, a=a, b=b, delta=pt[a][k] - pt[b][k], lo=np.percentile(db, 2.5), hi=np.percentile(db, 97.5),
                             p_one_sided=(1 + np.sum(db <= 0)) / (len(db) + 1)))
        fam = [i for i, r in enumerate(rows) if r["hyp"].startswith("H") and r["hyp"] != "H7-control"]
        adj = holm([rows[i]["p_one_sided"] for i in fam])
        for i, r in enumerate(rows): r["p_holm"] = adj[fam.index(i)] if i in fam else np.nan
        T2 += rows
        bc_up = sum(pt[c]["macroF1_BC"] >= pt[c]["macroF1"] for c in g if "+" not in c and not c.startswith("BL-"))
        print(f"{m}: {nk} (draft, nation) pairs in every condition | H6 part 1: BC >= raw macro-F1 in {bc_up}/{sum('+' not in c and not c.startswith('BL-') for c in g)} conditions")
    t1, t2 = pd.DataFrame(T1), pd.DataFrame(T2)
    cell = lambda r: f"{r.macroF1_BC:.3f} / {r.aucN:.3f} / {r.yes_gap:+.2f} / {r.no_recall:.2f}"
    tab = t1.assign(v=t1.apply(cell, axis=1)).pivot(index="cond", columns="model", values="v").reindex([c for c in CONDS if c in set(t1.cond)])
    t2.to_csv(OUT / "llm_tests.csv", index=False); t1.to_csv(OUT / "llm_conditions.csv", index=False)
    fmt = t2.assign(delta=t2.delta.map("{:+.3f}".format), ci=t2.apply(lambda r: f"[{r.lo:+.3f}, {r.hi:+.3f}]", axis=1),
                    p=t2.p_one_sided.map("{:.4f}".format), p_holm=t2.p_holm.map(lambda x: "" if pd.isna(x) else f"{x:.4f}"))[
        ["model", "hyp", "metric", "a", "b", "delta", "ci", "p", "p_holm"]]
    def mdtab(x, index=True):
        try: return x.to_markdown(index=index)
        except ImportError: return "```\n" + x.to_string(index=index) + "\n```"
    md = ["# LLM results across models", "", "Cell = macro-F1 after Batch Calibration / AUROC of P(against) / Yes-rate gap / No-recall", "",
          mdtab(tab), "", "## Pre-registered tests (one-sided, resolution-cluster bootstrap; Holm within model over H5-H7; H7-control and E-baseline rows are exploratory)", "",
          mdtab(fmt, False)]
    (OUT / "llm_crossmodel.md").write_text("\n".join(md))
    with pd.option_context("display.width", 260, "display.max_columns", 20, "display.max_colwidth", 40):
        print("\n==== Table 1: macro-F1(BC) / AUROC(N) / Yes-gap / No-recall ====\n" + tab.to_string())
        print("\n==== Table 2: pre-registered tests ====\n" + fmt.to_string(index=False))
    print(f"\nwritten: {OUT / 'llm_crossmodel.md'}, llm_tests.csv, llm_conditions.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); ap.add_argument("--models", default="")
    a = ap.parse_args(); main(a.B, [x for x in a.models.split(",") if x])
