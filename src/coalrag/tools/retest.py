"""Test-retest of one model's main conditions (D92): e4_<m>_main.jsonl vs e4_<m>_mainrep.jsonl, same serving standard, same code.
Prints row-level agreement per condition and every key contrast estimated in each run separately.
Usage: python -m coalrag.tools.retest <served> [--B 2000]"""
import argparse
import numpy as np, pandas as pd
from coalrag.analysis.e5_analysis import load
from coalrag.analysis.e7_analysis import pboot

CONTRASTS = [("H5", "aucN", "D-oracle-coal", "B0-std"), ("H5", "aucN", "D-strat4-coal", "B0-std"),
             ("H5", "macroF1_BC", "D-oracle-coal", "D-random-coal"), ("H6", "macroF1_BC", "D-strat4-coal", "B0-std"),
             ("H6", "macroF1_BC", "D-oracle-coal", "B0-std"), ("raw", "macroF1", "D-random-coal", "B0-std"),
             ("BC", "macroF1_BC", "D-random-coal", "B0-std"), ("E", "aucN", "D-strat4-coal", "D-top4-coal")]


def main(m, B):
    a, b = load(m, "main"), load(m, "mainrep")
    if b is None: raise SystemExit(f"no e4_{m}_mainrep.jsonl")
    k = ["qid", "nation", "cond"]; j = a[k + ["pY", "pN", "pA"]].merge(b[k + ["pY", "pN", "pA"]], on=k, suffixes=("_1", "_2"))
    P1 = j[["pY_1", "pN_1", "pA_1"]].values.astype(float); P2 = j[["pY_2", "pN_2", "pA_2"]].values.astype(float)
    j["dp"] = np.abs(P1 - P2).max(1); j["same"] = P1.argmax(1) == P2.argmax(1)
    print(f"==== {m}: {len(j)} rows scored in both runs | max |dp| {j.dp.max():.4f} | mean {j.dp.mean():.5f} | argmax agreement {j.same.mean():.3f} ====")
    print(j.groupby("cond").agg(rows=("dp", "size"), mean_dp=("dp", "mean"), max_dp=("dp", "max"), argmax_agree=("same", "mean")).round(4).to_string())
    out = []
    for h, met, x, y in CONTRASTS:
        r = dict(hyp=h, metric=met, a=x, b=y)
        for tag, d in (("run1", a), ("run2", b)):
            g = {c: v for c, v in d.groupby("cond")}
            if x in g and y in g:
                dl, lo, hi, p = pboot(g[x], g[y], met, B); r[tag] = f"{dl:+.3f} [{lo:+.3f}, {hi:+.3f}] p={p:.4f}"
        out.append(r)
    with pd.option_context("display.width", 220, "display.max_colwidth", 60): print("\n" + pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("served"); ap.add_argument("--B", type=int, default=2000); a = ap.parse_args(); main(a.served, a.B)
