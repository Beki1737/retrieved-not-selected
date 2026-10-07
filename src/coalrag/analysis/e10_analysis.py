"""E10 analysis (pre-registered H17, D99; fixed before any E10 output). Rows: (draft, nation) pairs with a guaranteed match.
  H17a  AUROC(N): matching precedent alone (G-orc) > four precedents shown together (mean over the four match positions)
  H17b  AUROC(N): matching precedent alone > the precedent the model itself selects, shown alone (mean over the four orderings)
  H17c  the model's selection hits the matching precedent more often when it is first than when it is last (p1 - p4 > 0)
  H17d  AUROC(N): four shown together with the match first > with the match last (G4-p1 - G4-p4 > 0)
One-sided resolution-cluster bootstrap (B = 2000); Holm within model over H17a-d.
Exploratory (no support claim): four together vs no evidence; first-ranked, random (uniform pick) and model-choice strategies against
the matching precedent, each shown alone with no fallback (a match always exists); selection accuracy by position vs chance (25%).
Usage: python -m coalrag.analysis.e10_analysis [--B 2000]"""
import argparse, json
import numpy as np, pandas as pd
from coalrag.analysis.e4_metrics import metrics
from coalrag.analysis.e5_analysis import load
from coalrag.analysis.e7_analysis import STORE, holm

POS = (1, 2, 3, 4)


def build(m):
    d = load(m, "guar")
    if d is None: return None
    S = {}
    for r in map(json.loads, open(STORE / f"e10_{m}_selector.jsonl")):
        if not r.get("error"): S[(r["qid"], r["pos"])] = (r["ids"], np.array(r["w"], float))
    need = ["G-B0", "G-orc", "G-d1", "G-d2", "G-d3"] + [f"G4-p{p}" for p in POS]
    g = {c: x.set_index(["qid", "nation"]) for c, x in d.groupby("cond")}
    keys = sorted(set.intersection(*[set(g[c].index) for c in need if c in g]))
    keys = [k for k in keys if all((k[0], p) in S for p in POS)]
    y = np.array([g["G-orc"].loc[k, "true"] for k in keys]); qid = np.array([k[0] for k in keys])
    P = {c: g[c].loc[keys, ["pY", "pN", "pA"]].values.astype(float) for c in need}
    single = {}; rank = np.zeros((len(keys), 4))
    for c, j in (("G-orc", 0), ("G-d1", 1), ("G-d2", 2), ("G-d3", 3)):
        single[j] = P[c]; rank[:, j] = g[c].loc[keys, "rank"].fillna(1e9).values if "rank" in g[c] else j
    ids = {k: {g["G-orc"].loc[k, "retrieved"][0]: 0, **{g[f"G-d{i}"].loc[k, "retrieved"][0]: i for i in (1, 2, 3)}} for k in keys}
    pick = np.zeros((len(keys), 4), int)  # chosen single (0 = match) per ordering p
    for i, k in enumerate(keys):
        for p in POS:
            sid, w = S[(k[0], p)]; pick[i, p - 1] = ids[k][sid[int(np.argmax(w))]]
    first = rank.argmin(1)
    return dict(keys=keys, y=y, qid=qid, P=P, single=np.stack([single[j] for j in range(4)], 1), pick=pick, first=first)


def auc(y, P): return metrics(y, P)["aucN"]


def stats(D, ix, rng):
    y, P, Sg, pick = D["y"][ix], {c: v[ix] for c, v in D["P"].items()}, D["single"][ix], D["pick"][ix]
    r = np.arange(len(ix)); out = {}
    out["B0"] = auc(y, P["G-B0"]); out["match alone"] = auc(y, P["G-orc"])
    out["four together (mean over positions)"] = np.mean([auc(y, P[f"G4-p{p}"]) for p in POS])
    for p in POS: out[f"four together, match at {p}"] = auc(y, P[f"G4-p{p}"])
    out["model's choice alone (mean over orderings)"] = np.mean([auc(y, Sg[r, pick[:, p - 1]]) for p in POS])
    out["first-ranked alone"] = auc(y, Sg[r, D["first"][ix]])
    out["random pick alone"] = auc(y, Sg[r, rng.integers(0, 4, len(ix))])
    out["sel hit p1"] = float(np.mean(pick[:, 0] == 0)); out["sel hit p4"] = float(np.mean(pick[:, 3] == 0))
    out["sel hit mean"] = float(np.mean(pick == 0))
    return out


TESTS = [("H17a", "match alone", "four together (mean over positions)"), ("H17b", "match alone", "model's choice alone (mean over orderings)"),
         ("H17c", "sel hit p1", "sel hit p4"), ("H17d", "four together, match at 1", "four together, match at 4"),
         ("E", "four together (mean over positions)", "B0"), ("E", "match alone", "first-ranked alone"), ("E", "match alone", "random pick alone"),
         ("E", "model's choice alone (mean over orderings)", "first-ranked alone"), ("E", "model's choice alone (mean over orderings)", "random pick alone")]


def main(B):
    models = sorted(p.name[3:-11] for p in STORE.glob("e4_*_guar.jsonl")); rows = []
    for m in models:
        D = build(m)
        if D is None or not len(D["keys"]): continue
        uq = np.unique(D["qid"]); pos = {u: np.where(D["qid"] == u)[0] for u in uq}; rng = np.random.default_rng(0)
        full = np.arange(len(D["y"])); pt = stats(D, full, rng)
        pt["random pick alone"] = np.mean([stats(D, full, np.random.default_rng(s))["random pick alone"] for s in range(200)])
        bs = [stats(D, np.concatenate([pos[u] for u in rng.choice(uq, len(uq))]), rng) for _ in range(B)]
        print(f"\n==== {m} | drafts {len(uq)} | rows {len(full)} | AUROC(N) and selection hit ====")
        for k, v in pt.items(): print(f"  {k:45s} {v:.3f}  [{np.nanpercentile([b[k] for b in bs], 2.5):.3f}, {np.nanpercentile([b[k] for b in bs], 97.5):.3f}]")
        print(f"  selection hit by match position: " + ", ".join(f"p{p} {np.mean(D['pick'][:, p - 1] == 0):.3f}" for p in POS) + " | chance 0.250")
        for h, a, b in TESTS:
            d = np.array([x[a] - x[b] for x in bs]); d = d[~np.isnan(d)]
            rows.append(dict(model=m, hyp=h, a=a, b=b, delta=pt[a] - pt[b], ci=f"[{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]",
                             p=(1 + np.sum(d <= 0)) / (len(d) + 1), drafts=len(uq), rows=len(full)))
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for m in t.model.unique():
            i = t.index[(t.model == m) & t.hyp.str.startswith("H17")]; t.loc[i, "p_holm"] = holm(t.loc[i, "p"])
        t["delta"] = t.delta.map("{:+.3f}".format); t["p"] = t.p.map("{:.4f}".format)
        with pd.option_context("display.width", 250, "display.max_colwidth", 48):
            print("\n==== E10 tests (one-sided; Holm within model over H17a-d; E = exploratory) ====\n" + t.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); main(ap.parse_args().B)
