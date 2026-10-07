"""LLM metrics (D49): threshold-free discrimination, Batch Calibration (Zhou et al., ICLR 2024), context-aware decoding
(Shi et al., NAACL 2024; alpha=1 fixed a priori) and AdaCAD (Wang et al., NAACL 2025; alpha = JSD, parameter-free),
computed post hoc from paired with/without-context option probabilities. Resolution-cluster bootstrap and paired deltas."""
import argparse, re
from pathlib import Path
import numpy as np, pandas as pd
L = np.array(["Y", "N", "A"])

def macro_f1(y, yp):
    fs = []
    for l in L:
        tp = np.sum((y == l) & (yp == l)); fp = np.sum((y != l) & (yp == l)); fn = np.sum((y == l) & (yp != l))
        p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(fs))

def auroc(s, pos):
    pos = np.asarray(pos, bool); n1, n0 = pos.sum(), (~pos).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values; return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def metrics(y, P):
    yp = L[P.argmax(1)]; Pc = P / P.mean(0, keepdims=True); ypc = L[Pc.argmax(1)]
    oh = np.stack([(y == l) for l in L], 1).astype(float)
    return dict(macroF1=macro_f1(y, yp), macroF1_BC=macro_f1(y, ypc), aucN=auroc(P[:, 1], y == "N"),
                aucMacro=np.nanmean([auroc(P[:, i], y == l) for i, l in enumerate(L)]),
                yes_gap=float((yp == "Y").mean() - (y == "Y").mean()), no_recall=float(((y == "N") & (yp == "N")).sum() / max((y == "N").sum(), 1)),
                brier=float(((P - oh) ** 2).sum(1).mean()), logloss=float(-np.log(np.clip((P * oh).sum(1), 1e-12, 1)).mean()), acc=float((y == yp).mean()))

def add_cad(df, base="B0-std"):
    """For every direct-evidence condition D-*, amplify the shift caused by the evidence relative to the identical
    prompt without the evidence block (B0-std): logit = (1+a) log p(y|ctx,q) - a log p(y|q)."""
    if base not in set(df.cond): return df
    b = df[df.cond == base].set_index(["qid", "nation"])[["pY", "pN", "pA"]]; out = []
    for c in sorted(x for x in df.cond.unique() if x.startswith("D-")):
        j = df[df.cond == c].set_index(["qid", "nation"])[["pY", "pN", "pA", "true"]].join(b, rsuffix="_0", how="inner")
        Pc = np.clip(j[["pY", "pN", "pA"]].values.astype(float), 1e-12, 1); P0 = np.clip(j[["pY_0", "pN_0", "pA_0"]].values.astype(float), 1e-12, 1)
        M = 0.5 * (Pc + P0); jsd = 0.5 * (Pc * np.log2(Pc / M)).sum(1) + 0.5 * (P0 * np.log2(P0 / M)).sum(1)
        for tag, alpha in [("CAD1", np.ones(len(j))), ("AdaCAD", jsd)]:
            z = (1 + alpha)[:, None] * np.log(Pc) - alpha[:, None] * np.log(P0); z -= z.max(1, keepdims=True)
            Pn = np.exp(z); Pn /= Pn.sum(1, keepdims=True)
            r = j.reset_index()[["qid", "nation", "true"]].copy(); r["cond"] = f"{c}+{tag}"
            r["pY"], r["pN"], r["pA"] = Pn[:, 0], Pn[:, 1], Pn[:, 2]; r["error"] = None; out.append(r)
    return pd.concat([df] + out, ignore_index=True)

def relabel(df, ver):
    """D61: replace stored truth with the votes of CKG <ver> (e.g. v1_2); rows without a recorded vote are dropped."""
    import os
    P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]
    ck = pd.read_parquet(Path(os.environ["PROJ"]) / "data" / "processed" / f"ckg_{ver}.parquet")
    m = {(r["res_id"], n): (r[n] if isinstance(r[n], str) else None) for r in ck[["res_id"] + P5].to_dict("records") for n in P5}
    new = [m.get((q, n), t) for q, n, t in zip(df.qid, df.nation, df.true)]
    ch = sum((a or "") != (b or "") for a, b in zip(new, df.true)); df = df.assign(true=new); keep = df.true.isin(list(L))
    print(f"[truth={ver}] relabelled {ch} rows | dropped {int((~keep).sum())} rows without a recorded vote"); return df[keep]

def main(path, ref, B, by, cad, show, truth=""):
    df = pd.concat([pd.read_json(x, lines=True) for x in path.split(",")], ignore_index=True)
    df = df[df.error.isna()].drop_duplicates(["qid", "nation", "cond"], keep="last")
    if truth: df = relabel(df, truth)
    if cad: df = add_cad(df, ref)
    conds = list(dict.fromkeys(df.cond)); keyset = df.groupby("cond").apply(lambda g: frozenset(zip(g.qid, g.nation)))
    common = frozenset.intersection(*keyset.values); df = df[[k in common for k in zip(df.qid, df.nation)]]
    df = df.sort_values(["cond", "qid", "nation"]); qids = np.array(sorted(df.qid.unique()))
    A = {c: (g.true.values, g[["pY", "pN", "pA"]].values.astype(float), g.qid.values) for c, g in df.groupby("cond")}
    qpos = {c: {q: np.where(A[c][2] == q)[0] for q in qids} for c in conds}
    point = {c: metrics(A[c][0], A[c][1]) for c in conds}
    K = ["macroF1", "macroF1_BC", "aucN"]; rng = np.random.default_rng(0); bd = {c: {k: [] for k in K} for c in conds}
    for _ in range(B):
        s = rng.choice(qids, len(qids)); m = {}
        for c in conds:
            ix = np.concatenate([qpos[c][q] for q in s]); m[c] = metrics(A[c][0][ix], A[c][1][ix])
        for c in conds:
            for k in K: bd[c][k].append(m[c][k] - m[ref][k])
    ci = lambda v: f"[{np.nanpercentile(v, 2.5):+.3f},{np.nanpercentile(v, 97.5):+.3f}]"
    print(f"file={Path(path).name} | rows/cond={len(A[conds[0]][0])} | resolutions={len(qids)} | ref={ref} | B={B} | CAD={cad}")
    rows = []
    for c in conds:
        if show and not re.search(show, c): continue
        p = point[c]
        rows.append({"cond": c, **{k: round(p[k], 3) for k in ["acc", "macroF1", "macroF1_BC", "aucN", "yes_gap", "no_recall", "brier", "logloss"]},
                     "dF1": f"{p['macroF1'] - point[ref]['macroF1']:+.3f} {ci(bd[c]['macroF1'])}",
                     "dF1_BC": f"{p['macroF1_BC'] - point[ref]['macroF1_BC']:+.3f} {ci(bd[c]['macroF1_BC'])}",
                     "dAUC_N": f"{p['aucN'] - point[ref]['aucN']:+.3f} {ci(bd[c]['aucN'])}"})
    with pd.option_context("display.width", 260, "display.max_columns", 30, "display.max_rows", 200):
        print(pd.DataFrame(rows).set_index("cond").to_string())
    if by:
        f = lambda g: metrics(g.true.values, g[["pY", "pN", "pA"]].values.astype(float))
        print(f"\n-- by {by}: macroF1_BC / aucN --")
        print(df.groupby(["cond", by]).apply(lambda g: f"{f(g)['macroF1_BC']:.2f} / {f(g)['aucN']:.2f}").unstack(by).to_string())

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--ref", default="B0-std")
    ap.add_argument("--B", type=int, default=1000); ap.add_argument("--by", default=""); ap.add_argument("--cad", action="store_true")
    ap.add_argument("--show", default=""); ap.add_argument("--truth", default="")
    a = ap.parse_args(); main(a.path, a.ref, a.B, a.by, a.cad, a.show, a.truth)
