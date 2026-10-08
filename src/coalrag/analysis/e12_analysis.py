"""E12 analysis (pre-registered H19, wave 7; fixed before any E12 output; human-label and among-appropriate blocks added after the first
E12 output and reported as exploratory).
Primary label: appropriate = P(O = yes) >= 0.5 from judge J1 (Llama-3.3-70B) under information P (what the predicting LLM sees,
votes and outcome removed). Robustness: judge J2 (Qwen2.5-32B), information R, same-agenda metadata rule, human annotation.
  H19a  E10 sets: the matching precedent is judged appropriate more often than the distractors (draft-level, one-sided)
  H19b  per model: the model's own pick (E10, four orderings) is appropriate more often than a random pick from the same set
  H19c  decision analysis on U = E10 drafts whose match is the only appropriate candidate: hit rate of the model's pick
        (pooled over models and per model, 95% draft-cluster CI). Fixed reading: upper bound < 0.5 -> selection failure among
        content-appropriate precedents; lower bound > 0.5 -> claim restricted to vote-matching precedents; else inconclusive;
        fewer than 10 drafts in U -> inconclusive by design
  H19d  per model: AUROC(N) of the candidate J1-P rates most appropriate, shown alone > the model's own pick shown alone
Bootstrap B = 2000 over drafts; Holm across models for H19b and H19d.
Usage: python -m coalrag.analysis.e12_analysis [--B 2000] [--j1 llama-3.3-70b-fp8] [--j2 qwen2.5-32b-gptq-int8]"""
import argparse, json, warnings
import numpy as np, pandas as pd
from coalrag.core.match import P5
from coalrag.retrieval.configA import ConfigA
from coalrag.analysis.e4_metrics import metrics
from coalrag.analysis.e7_analysis import STORE, holm
from coalrag.analysis import e10_analysis as E10A
from coalrag.experiments.e10_guaranteed import design
warnings.filterwarnings("ignore")
MODELS = ["llama-3.3-70b-fp8", "mistral-small-24b-2501-fp8", "qwen2.5-32b-gptq-int8", "qwen2.5-7b"]


def judged(j):
    rows = {}
    for f in sorted(STORE.glob(f"e12_{j}_s*.jsonl")):
        for r in map(json.loads, open(f)):
            if r.get("error") is None and r.get("p"): rows[(r["qid"], r["cid"], r["cond"], r["q"])] = r["p"]
    if not rows: return None
    out = {}
    for (q, c, cond, qq), p in rows.items():
        d = out.setdefault((q, c, cond), {})
        d[qq] = float(np.dot(p, np.arange(1, len(p) + 1))) if qq != "O" else float(p[1])
    t = pd.DataFrame([dict(qid=k[0], cid=k[1], cond=k[2], **v) for k, v in out.items()])
    t["app"] = t.O >= 0.5
    return t


def _yn(x):
    x = str(x).strip().lower()
    return 1.0 if x in ("yes", "y", "1", "true") else 0.0 if x in ("no", "n", "0", "false") else np.nan


def human_labels(L):
    """Annotator 1 (120 pairs) and annotator 2 (annotator2 subset): agreement with each other and with J1-P; returns
    {draft: array over [match, d1, d2, d3]} from annotator 1 for drafts with all four candidates labelled."""
    logs = STORE.parent.parent / "logs"; f1, f2 = logs / "e12_human_sheet_annotated.csv", logs / "e12_human_sheet_annotator2.csv"
    if not f1.exists(): print("human annotation: logs/e12_human_sheet_annotated.csv not found"); return None
    key = pd.read_csv(STORE / "e12_human_key.csv")
    def read(f):
        h = pd.read_csv(f).merge(key, on="pair_id", how="inner"); h["y"] = h.appropriate_yes_no.map(_yn)
        for c in ("same_situation_1to3", "same_countries_1to3", "same_action_1to3"): h[c] = pd.to_numeric(h.get(c), errors="coerce")
        return h.dropna(subset=["y"])
    h1 = read(f1); j = np.array([float(L[(a, b)].app) if (a, b) in L else np.nan for a, b in zip(h1.qid, h1.cid)]); m = ~np.isnan(j)
    print(f"human 1: {len(h1)} pairs labelled | appropriate {h1.y.mean():.3f} (matches {h1[h1.match].y.mean():.3f}, non-matches {h1[~h1.match].y.mean():.3f}) | "
          f"vs J1-P agreement {np.mean(h1.y[m] == j[m]):.3f}, kappa {kappa(h1.y[m] > .5, j[m] > .5):.3f}")
    if f2.exists():
        h2 = read(f2); x = h1.merge(h2[["pair_id", "y", "same_situation_1to3", "same_action_1to3"]], on="pair_id", suffixes=("", "_2"))
        j2 = np.array([float(L[(a, b)].app) if (a, b) in L else np.nan for a, b in zip(h2.qid, h2.cid)]); m2 = ~np.isnan(j2)
        print(f"human 2: {len(h2)} pairs | human 1 vs human 2 on {len(x)} shared pairs: agreement {np.mean(x.y == x.y_2):.3f}, kappa {kappa(x.y > .5, x.y_2 > .5):.3f} | "
              f"situation score exact agreement {np.mean(x.same_situation_1to3 == x.same_situation_1to3_2):.3f} | "
              f"human 2 vs J1-P kappa {kappa(h2.y[m2] > .5, j2[m2] > .5):.3f}")
    else:
        print("human 2: logs/e12_human_sheet_annotator2.csv not found")
    cfg = ConfigA(); H = {}
    for q, g in h1.groupby("qid"):
        dz = design(cfg, next(r for r in cfg.N if r["res_id"] == q))
        if not dz: continue
        o, dis, _ = dz; lab = dict(zip(g.cid, g.y)); cs = [o["res_id"]] + [d["res_id"] for d in dis]
        if all(c in lab for c in cs): H[q] = np.array([lab[c] for c in cs], float)
    print(f"human labels cover {len(H)} complete E10 drafts")
    return H or None


def kappa(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool); po = np.mean(a == b); pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def boot(units, f, B, rng):
    u = np.array(sorted(units)); return np.array([f(rng.choice(u, len(u))) for _ in range(B)])


def ci(x): x = x[~np.isnan(x)]; return f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]" if len(x) else "[n/a]"


def main(B, j1, j2):
    rng = np.random.default_rng(0); cfg = ConfigA(); rec = {r["res_id"]: r for r in cfg.A + cfg.N}
    pairs = pd.read_parquet(STORE / "e12_pairs.parquet"); J1 = judged(j1); J2 = judged(j2)
    if J1 is None: print("no J1 output"); return
    L = {(r.qid, r.cid): r for r in J1[J1.cond == "P"].itertuples()}
    print(f"==== E12 | pairs {len(pairs)} | judged by J1 (P) {len(L)} ====")
    # ---------- label quality and agreement
    norm = lambda s: " ".join(str(s or "").lower().split())
    pairs["meta_same_agenda"] = [norm(rec[a].get("agenda")) == norm(rec[b].get("agenda")) and norm(rec[a].get("agenda")) != "" for a, b in zip(pairs.qid, pairs.cid)]
    pairs["app"] = [L[(a, b)].app if (a, b) in L else np.nan for a, b in zip(pairs.qid, pairs.cid)]
    pairs["O"] = [L[(a, b)].O if (a, b) in L else np.nan for a, b in zip(pairs.qid, pairs.cid)]
    for c in ["T", "C", "A"]: pairs[c] = [getattr(L[(a, b)], c) if (a, b) in L else np.nan for a, b in zip(pairs.qid, pairs.cid)]
    ok = pairs.dropna(subset=["app"]); okb = ok.app.astype(bool)
    print(f"appropriate (J1-P): {okb.mean():.3f} of pairs | among vote matches {okb[ok.match].mean():.3f} (n={int(ok.match.sum())}) | "
          f"among non-matches {okb[~ok.match].mean():.3f} (n={int((~ok.match).sum())})")
    print(f"same agenda item (metadata): {ok.meta_same_agenda.mean():.3f} | P(app | same agenda) {okb[ok.meta_same_agenda].mean():.3f} | "
          f"P(app | other agenda) {okb[~ok.meta_same_agenda].mean():.3f} | kappa(app, same agenda) {kappa(okb, ok.meta_same_agenda):.3f}")
    R = {(r.qid, r.cid): r.app for r in J1[J1.cond == "R"].itertuples()}
    k = [(L[x].app, R[x]) for x in L if x in R]
    if k: print(f"J1 information P vs R: agreement {np.mean([a == b for a, b in k]):.3f}, kappa {kappa(*zip(*k)):.3f} (n={len(k)}) | "
                f"appropriate under R {np.mean([b for _, b in k]):.3f}")
    if J2 is not None:
        L2 = {(r.qid, r.cid): r.app for r in J2[J2.cond == "P"].itertuples()}; k = [(L[x].app, L2[x]) for x in L if x in L2]
        if k: print(f"J1 vs J2 (P): agreement {np.mean([a == b for a, b in k]):.3f}, kappa {kappa(*zip(*k)):.3f} (n={len(k)}) | J2 appropriate {np.mean([b for _, b in k]):.3f}")
    H = human_labels(L)
    # ---------- E10 sets
    E = {}
    for q in sorted(cfg.N, key=lambda r: r["date"]):
        dz = design(cfg, q)
        if dz: o, dis, _ = dz; E[q["res_id"]] = [o["res_id"]] + [d["res_id"] for d in dis]
    A = {q: np.array([L[(q, c)].app if (q, c) in L else np.nan for c in cs], float) for q, cs in E.items()}
    O = {q: np.array([L[(q, c)].O if (q, c) in L else np.nan for c in cs], float) for q, cs in E.items()}
    A = {q: v for q, v in A.items() if not np.isnan(v).any()}; drafts = sorted(A)
    d19a = lambda s: np.mean([A[q][0] - A[q][1:].mean() for q in s])
    pt = d19a(drafts); bs = boot(drafts, d19a, B, rng)
    print(f"\n==== E10 sets: {len(drafts)} drafts ====")
    print(f"match appropriate {np.mean([A[q][0] for q in drafts]):.3f} | distractors appropriate {np.mean([A[q][1:].mean() for q in drafts]):.3f} | "
          f"drafts with no appropriate candidate {np.mean([A[q].sum() == 0 for q in drafts]):.3f} | with 2+ appropriate {np.mean([A[q].sum() >= 2 for q in drafts]):.3f}")
    print(f"H19a  match - distractors (appropriate share): {pt:+.3f} {ci(bs)} one-sided p={(1 + np.sum(bs <= 0)) / (B + 1):.4f}")
    if H is not None:
        hq = sorted(H); fh = lambda s: np.mean([H[q][0] - H[q][1:].mean() for q in s]); bh = boot(hq, fh, B, rng)
        print(f"H19a with human labels ({len(hq)} drafts): match appropriate {np.mean([H[q][0] for q in hq]):.3f} | distractors "
              f"{np.mean([H[q][1:].mean() for q in hq]):.3f} | difference {fh(hq):+.3f} {ci(bh)} | U (only the match appropriate) "
              f"{sum(1 for q in hq if H[q][0] == 1 and H[q][1:].sum() == 0)} drafts")
    U = [q for q in drafts if A[q][0] == 1 and A[q][1:].sum() == 0]; MA = [q for q in drafts if A[q][0] == 1]; MN = [q for q in drafts if A[q][0] == 0]
    jpick = {q: int(np.argmax(O[q])) for q in drafts}  # continuous P(yes); exact ties do not occur in practice
    print(f"U (match is the only appropriate candidate): {len(U)} drafts | match appropriate, others mixed: {len(MA) - len(U)} | match not appropriate: {len(MN)}")
    print(f"judge's own pick (argmax P(O=yes), information P) hits the match in {np.mean([jpick[q] == 0 for q in drafts]):.3f} of drafts (chance 0.250)")
    rows = []; pooled = {}
    for m in MODELS:
        try: D = E10A.build(m)
        except Exception as e: print(f"  {m}: E10 outputs not loadable ({type(e).__name__})"); continue
        if D is None: continue
        dq = [q for q in dict.fromkeys(D["qid"]) if q in A]
        first = {q: np.where(D["qid"] == q)[0][0] for q in dq}
        pick = {q: D["pick"][first[q]] for q in dq}  # 4 orderings, index into [match, d1, d2, d3]
        pooled[m] = pick
        f19b = lambda s: np.mean([np.mean([A[q][p] for p in pick[q]]) - A[q].mean() for q in s])
        p19b = f19b(dq); b19b = boot(dq, f19b, B, rng)
        rows.append(dict(model=m, hyp="H19b", est=p19b, ci=ci(b19b), p=(1 + np.sum(b19b <= 0)) / (B + 1)))
        for lab, S in (("U", U), ("match appropriate", MA), ("match not appropriate", MN), ("all", dq)):
            S = [q for q in S if q in pick]
            if not S: continue
            fh = lambda s: np.mean([np.mean(pick[q] == 0) for q in s]); bh = boot(S, fh, B, rng)
            print(f"  {m:28s} hit rate on {lab:22s} {fh(S):.3f} {ci(bh)} (n drafts {len(S)})")
        # exploratory (post hoc): among drafts whose match is appropriate, is the match preferred over other appropriate candidates?
        S = [q for q in MA if q in pick]
        if S:
            pa = np.mean([np.mean([A[q][p] for p in pick[q]]) for q in S])
            hit_given_app = np.mean([np.mean([p == 0 for p in pick[q] if A[q][p] == 1]) if any(A[q][p] == 1 for p in pick[q]) else np.nan for q in S])
            print(f"  {m:28s} [match appropriate] pick appropriate {pa:.3f} | hit given an appropriate pick {np.nanmean(hit_given_app):.3f} "
                  f"vs uniform over appropriate candidates {np.mean([1 / A[q].sum() for q in S]):.3f}")
        if H is not None:
            hq = [q for q in H if q in pick]
            if hq:
                fb = lambda s: np.mean([np.mean([H[q][p] for p in pick[q]]) - H[q].mean() for q in s]); bb = boot(hq, fb, B, rng)
                hma = [q for q in hq if H[q][0] == 1]; hmn = [q for q in hq if H[q][0] == 0]
                print(f"  {m:28s} [human labels, {len(hq)} drafts] own pick appropriate minus random {fb(hq):+.3f} {ci(bb)} | "
                      f"hit when match appropriate {np.mean([np.mean(pick[q] == 0) for q in hma]) if hma else float('nan'):.3f} (n {len(hma)}), "
                      f"when not {np.mean([np.mean(pick[q] == 0) for q in hmn]) if hmn else float('nan'):.3f} (n {len(hmn)})")
        # H19d: AUROC(N) of judge-selected candidate shown alone vs own pick alone (mean over orderings)
        idx = {q: np.where(D["qid"] == q)[0] for q in dq}; y = D["y"]; Sg = D["single"]
        def auc_sel(s, which):
            ix = np.concatenate([idx[q] for q in s]); r = np.arange(len(ix))
            if which == "judge": P = Sg[ix][r, [jpick[D["qid"][i]] for i in ix]]; return metrics(y[ix], P)["aucN"]
            return np.mean([metrics(y[ix], Sg[ix][r, D["pick"][ix][:, o]])["aucN"] for o in range(4)])
        f19d = lambda s: auc_sel(s, "judge") - auc_sel(s, "own"); p19d = f19d(dq); b19d = boot(dq, f19d, B, rng)
        rows.append(dict(model=m, hyp="H19d", est=p19d, ci=ci(b19d), p=(1 + np.sum(b19d <= 0)) / (B + 1),
                         note=f"judge-pick alone {auc_sel(dq, 'judge'):.3f} vs own pick alone {auc_sel(dq, 'own'):.3f}"))
    if pooled:
        S = [q for q in U if all(q in pooled[m] for m in pooled)]
        fp = lambda s: np.mean([np.mean(pooled[m][q] == 0) for m in pooled for q in s])
        if len(S) >= 10:
            b = boot(S, fp, B, rng); lo, hi = np.percentile(b, [2.5, 97.5])
            verdict = ("selection failure among content-appropriate precedents" if hi < .5 else
                       "claim restricted to vote-matching precedents" if lo > .5 else "inconclusive")
            print(f"\nH19c pooled over {len(pooled)} models: hit rate on U {fp(S):.3f} [{lo:.3f}, {hi:.3f}] (n drafts {len(S)}) -> {verdict}")
        else:
            print(f"\nH19c: U has {len(S)} drafts (< 10) -> inconclusive by design; descriptive hit rate {fp(S) if S else float('nan'):.3f}")
    t = pd.DataFrame(rows)
    if len(t):
        t["p_holm"] = np.nan
        for h in t.hyp.unique(): i = t.index[t.hyp == h]; t.loc[i, "p_holm"] = holm(t.loc[i, "p"])
        with pd.option_context("display.width", 250, "display.max_colwidth", 70): print("\n==== H19b / H19d (one-sided; Holm across models) ====\n" + t.round(4).to_string(index=False))
    # ---------- E7 sets (exploratory)
    print("\n==== E7 stratified slots (exploratory) ====")
    for m in ["llama-3.3-70b-fp8", "mistral-small-24b-2501-fp8", "qwen2.5-32b-gptq-int8"]:
        f = STORE / f"e7_{m}_selector.jsonl"
        if not f.exists(): continue
        sel = {r["qid"]: np.array(r["w"], float) for r in map(json.loads, open(f)) if not r.get("error")}
        hit_app, rnd_app, cov = [], [], []
        for q in cfg.N:
            sl = cfg.strat_slots(q, 4, True)
            if q["res_id"] not in sel or len(sel[q["res_id"]]) != len(sl): continue
            a = [L[(q["res_id"], d["res_id"])].app if (q["res_id"], d["res_id"]) in L else np.nan for d in sl]
            if np.isnan(a).any(): continue
            hit_app.append(a[int(np.argmax(sel[q["res_id"]]))]); rnd_app.append(np.mean(a))
        print(f"  {m:28s} own pick appropriate {np.mean(hit_app):.3f} vs random pick {np.mean(rnd_app):.3f} (n drafts {len(hit_app)})")
    e7 = ok[ok.roles.str.contains("E7")]
    print(f"  E7 slots: appropriate {e7.app.astype(bool).mean():.3f}; matching slots appropriate {e7[e7.match].app.astype(bool).mean():.3f} (n={int(e7.match.sum())})")
    print("\n==== Rubric means (J1-P): matches vs non-matches ====")
    print(ok.groupby("match")[["T", "C", "A", "O"]].mean().round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--j1", default="llama-3.3-70b-fp8"); ap.add_argument("--j2", default="qwen2.5-32b-gptq-int8")
    a = ap.parse_args(); main(a.B, a.j1, a.j2)
