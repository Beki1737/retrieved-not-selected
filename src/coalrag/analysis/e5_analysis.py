"""Wave-2 analysis (D60 plan, fixed before results). Usage: python -m coalrag.analysis.e5_analysis <served> [--B 1000]
Reads e4_<served>_{main,wave2,allreg,think-off,think-on}.jsonl when present. Resolution-cluster paired bootstrap.
A evidence factorial and flip test | B order/k | C paraphrases | D memorization | E all regimes | F reasoning mode"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
from coalrag.analysis.e4_metrics import metrics

STORE = Path(os.environ["PROJ"]) / "results" / "store"; K = ["qid", "nation"]; LAB = ["Y", "N", "A"]


def load(m, tag):
    p = STORE / f"e4_{m}_{tag}.jsonl"
    if not p.exists(): return None
    d = pd.read_json(p, lines=True)
    d = d[d.error.isna() & d.true.isin(LAB)].drop_duplicates(K + ["cond"], keep="last").copy()
    P = d[["pY", "pN", "pA"]].values.astype(float)
    d["ptrue"] = P[np.arange(len(d)), d.true.map({"Y": 0, "N": 1, "A": 2}).values]
    d["pred"] = np.array(LAB)[P.argmax(1)]; d["hit"] = (d.pred == d.true).astype(float)
    return d


def stat(name):
    if name in ("ptrue", "hit"): return lambda g: float(g[name].mean())
    if name == "false_against": return lambda g: float(((g.pred == "N") & (g.true != "N")).sum() / max((g.true != "N").sum(), 1))
    return lambda g: metrics(g.true.values, g[["pY", "pN", "pA"]].values.astype(float))[name]


def align(frames):
    keys = set.intersection(*[set(zip(f.qid, f.nation)) for f in frames])
    return [f[[k in keys for k in zip(f.qid, f.nation)]].sort_values(K).reset_index(drop=True) for f in frames]


def paired(a, b, name, B, seed=0):
    a, b = align([a, b]); f = stat(name); q = a.qid.values; uq = np.unique(q)
    pos = {u: np.where(q == u)[0] for u in uq}; rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        ix = np.concatenate([pos[u] for u in rng.choice(uq, len(uq))]); bs.append(f(a.iloc[ix]) - f(b.iloc[ix]))
    d = f(a) - f(b); lo, hi = np.nanpercentile(bs, [2.5, 97.5]); star = "*" if lo > 0 or hi < 0 else " "
    return f"{d:+.3f} [{lo:+.3f},{hi:+.3f}]{star}"


def table(d, conds, cols=("hit", "ptrue", "macroF1_BC", "aucN", "yes_gap", "no_recall")):
    rows = []
    for c in conds:
        g = d[d.cond == c]
        if len(g): rows.append(dict(cond=c, n=len(g), **{k: round(stat(k)(g), 3) for k in cols}))
    return pd.DataFrame(rows).set_index("cond") if rows else pd.DataFrame()


def main(m, B):
    main_, w2, ar = load(m, "main"), load(m, "wave2"), load(m, "allreg")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    if w2 is not None and main_ is not None:
        # ---- A. evidence factorial on the oracle precedent ----
        F = ["F-orc-text", "F-orc-coal", "F-orc-others", "F-orc-flip", "F-rnd-coal", "F-rnd-orcv"]
        base = main_[main_.cond == "B0-std"]; fr = [base] + [w2[w2.cond == c] for c in F if c in set(w2.cond)]
        A = pd.concat(align(fr)); print(f"\n==== A. EVIDENCE FACTORIAL (one oracle precedent; keys={A.groupby('cond').size().min()}) ====")
        print(table(A, ["B0-std"] + F).to_string())
        g = {c: A[A.cond == c] for c in A.cond.unique()}
        C = [("votes | relevant text", "F-orc-coal", "F-orc-text"), ("relevant text | no votes", "F-orc-text", "B0-std"),
             ("correct vs own votes | random text", "F-rnd-orcv", "F-rnd-coal"), ("text relevance | correct votes", "F-orc-coal", "F-rnd-orcv"),
             ("own vote shown vs allies only", "F-orc-coal", "F-orc-others"), ("allies only vs text only", "F-orc-others", "F-orc-text"),
             ("flipped target vote", "F-orc-flip", "F-orc-coal")]
        print("\n-- paired contrasts (cluster bootstrap 95% CI; * = excludes 0) --")
        for lab, x, y in C:
            if x in g and y in g:
                print(f"{lab:38s} {x} - {y}: P(true) {paired(g[x], g[y], 'ptrue', B)} | acc {paired(g[x], g[y], 'hit', B)} | AUROC(N) {paired(g[x], g[y], 'aucN', B)}")
        if "F-orc-flip" in g and "F-orc-text" in g:
            fl, tx = align([g["F-orc-flip"], g["F-orc-text"]])
            col = {"Y": "pY", "N": "pN", "A": "pA"}
            fl["p_shown"] = [r[col[s]] for s, r in zip(fl.shown, fl.to_dict("records"))]
            tx["p_shown"] = [r[col[s]] for s, r in zip(fl.shown, tx.to_dict("records"))]
            fl["uptake"] = fl.p_shown - tx.p_shown; fl["follow"] = (fl.pred == fl.shown).astype(float)
            fl["case"] = fl.true + "->" + fl.shown
            print("\n-- flip test: uptake = P(shown label | flipped evidence) - P(same label | text only); follow = argmax equals shown --")
            print(fl.groupby("case").agg(n=("uptake", "size"), uptake=("uptake", "mean"), follow=("follow", "mean")).round(3).to_string())
            co, tx2 = align([g["F-orc-coal"], g["F-orc-text"]]); co["uptake_true"] = co.ptrue.values - tx2.ptrue.values
            print("-- correct evidence: uptake of the true label by true label --")
            print(co.groupby("true").uptake_true.agg(["size", "mean"]).round(3).to_string())
            a = fl[fl.case == "Y->N"].uptake; b = fl[fl.case == "N->Y"].uptake
            if len(a) and len(b):
                rng = np.random.default_rng(0); bs = [rng.choice(a.values, len(a)).mean() - rng.choice(b.values, len(b)).mean() for _ in range(B)]
                print(f"asymmetry: uptake(shown against | true favour) - uptake(shown favour | true against) = {a.mean() - b.mean():+.3f} "
                      f"[{np.percentile(bs, 2.5):+.3f},{np.percentile(bs, 97.5):+.3f}]")
        # ---- B. order and k ----
        R = ["D-strat4-coal", "R-strat4-rev", "R-strat4-shuf", "R-strat2-coal", "R-strat6-coal", "D-top4-coal"]
        fr = [main_[main_.cond == c] for c in ("B0-std", "D-strat4-coal", "D-top4-coal")] + [w2[w2.cond == c] for c in R[1:5]]
        Rd = pd.concat(align([f for f in fr if len(f)])); gg = {c: Rd[Rd.cond == c] for c in Rd.cond.unique()}
        print("\n==== B. ORDER AND k (stratified evidence, coalition view) ====")
        print(table(Rd, ["B0-std"] + R).to_string())
        for c in R[1:]:
            if c in gg: print(f"{c:15s} - D-strat4-coal: AUROC(N) {paired(gg[c], gg['D-strat4-coal'], 'aucN', B)} | P(true) {paired(gg[c], gg['D-strat4-coal'], 'ptrue', B)}")
        # ---- C. paraphrases ----
        P = sorted({c.split("-")[0] for c in w2.cond if c[0] == "P" and c[1].isdigit()})
        if P:
            rows = []
            for p, (b0, s4) in [("P0 (main)", ("B0-std", "D-strat4-coal"))] + [(p, (f"{p}-B0", f"{p}-strat4")) for p in P]:
                src = main_ if p.startswith("P0") else w2; x, y = src[src.cond == b0], src[src.cond == s4]
                if not len(x) or not len(y): continue
                tpl = "" if p.startswith("P0") else str(x.template.iloc[0])
                rows.append(dict(template=p, name=tpl, yes_rate_B0=(x.pred == "Y").mean(), aucN_B0=stat("aucN")(x), F1BC_B0=stat("macroF1_BC")(x),
                                 aucN_strat4=stat("aucN")(y), F1BC_strat4=stat("macroF1_BC")(y), d_aucN=paired(y, x, "aucN", B)))
            T = pd.DataFrame(rows).set_index("template")
            print("\n==== C. PARAPHRASE ROBUSTNESS (Choi templates) ====\n" + T.round(3).to_string())
            num = T[["yes_rate_B0", "aucN_B0", "aucN_strat4", "F1BC_B0", "F1BC_strat4"]]
            print("mean +- sd across templates:", {k: f"{v:.3f} +- {s:.3f}" for k, v, s in zip(num.columns, num.mean(), num.std())})
        # ---- D. memorization ----
        if "M-symbol" in set(w2.cond):
            M = w2[w2.cond == "M-symbol"]; print("\n==== D. MEMORIZATION PROBE (symbol + date only) ====")
            print(table(M, ["M-symbol"]).to_string(), "| majority-class acc:", round(M.true.value_counts(normalize=True).max(), 3))
            Mi = M.set_index(K); mem = (Mi.pred.eq(Mi.true) & Mi.true.ne("Y")).rename("memorized")  # D79: probe right on a non-favour vote
            print(f"memorized (probe correct on an against/abstention vote): {int(mem.sum())} of {(Mi.true != 'Y').sum()} non-favour rows")
            for c in ("D-strat4-coal", "D-oracle-coal", "D-bge-coal"):
                x, y = align([main_[main_.cond == c], base])
                j = x.set_index(K).ptrue.sub(y.set_index(K).ptrue).rename("gain").to_frame().join(mem, how="inner")
                print(f"{c:14s} gain in P(true) over B0-std: memorized {j[j.memorized].gain.mean():+.3f} (n={j.memorized.sum()}) | "
                      f"not memorized {j[~j.memorized].gain.mean():+.3f} (n={(~j.memorized).sum()})")
    # ---- E. all regimes ----
    if ar is not None and main_ is not None:
        E = pd.concat([ar, main_[main_.cond.isin(set(ar.cond))]]); print("\n==== E. ALL REGIMES (adopted queries from allreg + non-adopted from main) ====")
        rows = []
        for (c, s), g in E.groupby(["cond", "qset"]):
            notN = g.true != "N"
            rows.append(dict(cond=c, qset=s, n=len(g), yes_rate=(g.pred == "Y").mean(), true_yes=(g.true == "Y").mean(),
                             false_against=((g.pred == "N") & notN).sum() / max(notN.sum(), 1), no_recall=((g.pred == "N") & ~notN).sum() / max((~notN).sum(), 1),
                             F1BC=stat("macroF1_BC")(g)))
        print(pd.DataFrame(rows).set_index(["cond", "qset"]).round(3).to_string())
        print("-- pooled over all queries: AUROC(N) / macro-F1 (BC) --")
        for c in sorted(set(ar.cond)):
            g = E[E.cond == c]; print(f"{c:15s} n={len(g)} AUROC(N)={stat('aucN')(g):.3f} F1BC={stat('macroF1_BC')(g):.3f}")
        for c in sorted(set(ar.cond) - {"B0-std"}):
            x, y = E[E.cond == c], E[E.cond == "B0-std"]
            xc, yc = x[x.qset == "consensus"], y[y.qset == "consensus"]
            if len(xc) and len(yc): print(f"  H12 {c} - B0-std | consensus false-against rate {paired(xc, yc, 'false_against', B)}")
            for s in ["consensus", "split", "vetoed", "failed"]:
                xs, ys = x[x.qset == s], y[y.qset == s]
                if len(xs) and len(ys): print(f"  {c} - B0-std | {s:9s} P(true) {paired(xs, ys, 'ptrue', B)}")
    # ---- F. reasoning mode ----
    th = {t: load(m, t) for t in ("think-off", "think-on")}
    if all(v is not None for v in th.values()):
        print("\n==== F. REASONING MODE (thinking off vs on) ====")
        for t, d in th.items(): print(f"-- {t} --\n" + table(d, list(dict.fromkeys(d.cond))).to_string())
        on, off = th["think-on"], th["think-off"]
        for c in dict.fromkeys(on.cond):
            print(f"{c:15s} on - off: AUROC(N) {paired(on[on.cond == c], off[off.cond == c], 'aucN', B)} | P(true) {paired(on[on.cond == c], off[off.cond == c], 'ptrue', B)}")
        if "reason_tokens" in on: print("reasoning tokens: mean", round(on.reason_tokens.mean()), "| truncated share", round(on.truncated.mean(), 3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("served"); ap.add_argument("--B", type=int, default=1000)
    a = ap.parse_args(); main(a.served, a.B)
