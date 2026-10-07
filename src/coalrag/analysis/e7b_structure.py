"""E7b structure-aware selection among the four stratified precedents (pre-registered H16, D91; rules fixed before any output).
S-lift shows alone the stratified precedent whose five-vote configuration is most over-represented among the draft's KN nearest
earlier documents (bge-small on UNSC-CKG text; Choi pool; strictly earlier) relative to its share of the whole eligible pool:
    score(c) = log[(n_c,KN + 0.5) / (KN + 0.5 k)] - log[(N_c + 0.5) / (N + 0.5 k)],  KN = 20,  ties to the more similar slot.
It uses neither the draft's votes nor its outcome.
  H16a  S-lift > D-strat4-coal in AUROC(N); one-sided resolution-cluster paired bootstrap; Holm across the models with E7 output.
  H16b  retrieval only, every draft of the unfiltered pool: selection hit (chosen slot carries the draft's configuration) of
        S-lift minus the slot-1 rule > 0 on split, vetoed and failed drafts and < 0 on consensus drafts (expected trade-off).
Exploratory (no support claim): soft weights exp(score) (S-liftsoft); KN = 10 and 50; S-lift vs S-llmsel and S-top1.
Also prints H13 with Holm over the pre-registered E7 family only (Mistral-24B, Qwen-32B).
Usage: python -m coalrag.analysis.e7b_structure [--B 2000]"""
import argparse, os
os.environ.setdefault("CKG_VERSION", "v1_2")
import numpy as np, pandas as pd
from coalrag.analysis.e5_analysis import stat, align, table
from coalrag.analysis.e7_analysis import STORE, P5, rows_from, pboot, holm, build
from coalrag.data.configB import load as ckg_load
from coalrag.retrieval.configA import ConfigA

KN, KN_EXPL = 20, (10, 50)
PREREG_E7 = {"mistral-small-24b-2501-fp8", "qwen2.5-32b-gptq-int8"}
Kdf, _, _, _, _, EMB = ckg_load(partial_veto_coding=True, with_emb=True)
E = EMB["bge-small"]; pos = {r: i for i, r in enumerate(Kdf.res_id.tolist())}; dates = Kdf.date.values; REG = Kdf.regime.values
cfg = ConfigA(); vot = cfg.vot
choi = [r["res_id"] for r in cfg.A + cfg.N if r["res_id"] in pos]
conf = lambda rid: tuple(vot[rid][x] for x in P5)
complete = lambda rid: all(vot[rid][x] in ("Y", "N", "A") for x in P5)


def lift(qid, slot_ids, kn=KN):
    qi = pos[qid]; pool = [r for r in choi if dates[pos[r]] < dates[qi] and r != qid]
    k = len(slot_ids)
    if not pool: return np.zeros(k)
    o = np.argsort(-(E[[pos[r] for r in pool]] @ E[qi]), kind="stable")[:kn]
    near = [conf(pool[i]) for i in o]; allc = [conf(r) for r in pool]
    s = []
    for d in slot_ids:
        c = conf(d)
        s.append(np.log((sum(x == c for x in near) + 0.5) / (len(near) + 0.5 * k)) - np.log((sum(x == c for x in allc) + 0.5) / (len(allc) + 0.5 * k)))
    return np.array(s)


def llm_part(B):
    from coalrag.analysis.e5_analysis import load
    models = sorted(p.name[3:-13] for p in STORE.glob("e4_*_select.jsonl"))
    tests, sel_rows = [], []
    for m in models:
        d0, meta0 = build(m); S = load(m, "select"); cache = {}
        out = {c: [] for c in ("S-lift", "S-liftsoft", "S-lift-k10", "S-lift-k50")}
        for (q, n), g in S.groupby(["qid", "nation"]):
            g = g.sort_values("slot"); P = g[["pY", "pN", "pA"]].values.astype(float)
            base = g.iloc[[0]][["qid", "symbol", "qset", "nation", "true"]]; ids = [r[0] for r in g.retrieved]
            if q not in cache: cache[q] = {kn: lift(q, ids, kn) for kn in (KN,) + KN_EXPL}
            s = cache[q][KN]; w = np.exp(s - s.max()); w = w / w.sum()
            out["S-lift"].append((base, P[int(np.argmax(s))])); out["S-liftsoft"].append((base, w @ P))
            out["S-lift-k10"].append((base, P[int(np.argmax(cache[q][10]))])); out["S-lift-k50"].append((base, P[int(np.argmax(cache[q][50]))]))
            cq = conf(q); match = [i for i, x in enumerate(ids) if conf(x) == cq]
            sel_rows.append(dict(model=m, qid=q, nation=n, covered=bool(match), match_slot=match[0] + 1 if match else None, lift_pick=int(np.argmax(s)) + 1))
        agg = [rows_from(pd.concat([b for b, _ in v], ignore_index=True), np.vstack([x for _, x in v]), c) for c, v in out.items()]
        d = pd.concat(align([g for _, g in pd.concat([d0] + agg, ignore_index=True).groupby("cond")]))
        conds = ["B0-std", "D-strat4-coal", "S-top1", "S-llmsel", "S-lift", "S-liftsoft", "S-lift-k10", "S-lift-k50", "S-perfect", "D-oracle-coal"]
        print(f"\n==== {m} | keys {d.groupby('cond').size().min()} ====")
        with pd.option_context("display.width", 200): print(table(d, [c for c in conds if c in set(d.cond)]).to_string())
        sm = pd.DataFrame([r for r in sel_rows if r["model"] == m]).drop_duplicates("qid"); h = sm[sm.covered]
        print(f"covered drafts {h.qid.nunique()} of {sm.qid.nunique()} | S-lift picks the matching slot in {np.mean(h.lift_pick == h.match_slot):.1%}"
              f" (slot-1 rule {np.mean(h.match_slot == 1):.1%}) | S-lift picks slot 1 for {np.mean(sm.lift_pick == 1):.1%} of drafts")
        g = {c: x for c, x in d.groupby("cond")}
        for hyp, k, a, b in [("H16a", "aucN", "S-lift", "D-strat4-coal"), ("E", "macroF1_BC", "S-lift", "D-strat4-coal"),
                             ("E", "aucN", "S-lift", "S-llmsel"), ("E", "aucN", "S-lift", "S-top1"), ("E", "aucN", "S-liftsoft", "D-strat4-coal"),
                             ("E", "aucN", "D-oracle-coal", "S-lift"), ("E", "false_against", "S-lift", "D-strat4-coal")]:
            if a in g and b in g:
                dl, lo, hi, p = pboot(g[a], g[b], k, B)
                tests.append(dict(model=m, hyp=hyp, metric=k, a=a, b=b, delta=dl, ci=f"[{lo:+.3f}, {hi:+.3f}]", p=p))
        if m in PREREG_E7 and "S-llmsel" in g:
            dl, lo, hi, p = pboot(g["S-llmsel"], g["D-strat4-coal"], "aucN", B)
            tests.append(dict(model=m, hyp="H13-prereg", metric="aucN", a="S-llmsel", b="D-strat4-coal", delta=dl, ci=f"[{lo:+.3f}, {hi:+.3f}]", p=p))
    t = pd.DataFrame(tests)
    if len(t):
        t["p_holm"] = np.nan
        for hyp in ("H16a", "H13-prereg"):
            i = t.index[(t.hyp == hyp) & (t.metric == "aucN")]
            if len(i): t.loc[i, "p_holm"] = holm(t.loc[i, "p"])
        t["delta"] = t.delta.map("{:+.3f}".format); t["p"] = t.p.map("{:.4f}".format)
        print("\n==== E7b tests (one-sided; Holm across models for H16a; H13-prereg = Holm over Mistral + Qwen-32B only) ====\n" + t.to_string(index=False))


def retrieval_part(B, seed=0):
    rows = []
    for r in cfg.A + cfg.N:
        q = r["res_id"]
        if q not in pos or not complete(q): continue
        slots = [x["res_id"] for x in cfg.strat_slots(r, 4, True)]
        if len(slots) < 2: continue
        cq = conf(q); hits = [conf(x) == cq for x in slots]; s = lift(q, slots)
        rows.append(dict(qid=q, regime=REG[pos[q]], covered=any(hits), slot1=hits[0], lift=hits[int(np.argmax(s))]))
    df = pd.DataFrame(rows); rng = np.random.default_rng(seed)
    print("\n==== H16b: selection hit among 4 stratified slots, all drafts of the unfiltered pool (%; S-lift minus slot-1, 95% CI) ====")
    for reg, g in df.groupby("regime"):
        dd = (g.lift.astype(float) - g.slot1.astype(float)).values; bs = dd[rng.integers(0, len(dd), (B, len(dd)))].mean(1)
        print(f"{reg:9s} n={len(g):3d} | covered {100 * g.covered.mean():5.1f} | slot-1 {100 * g.slot1.mean():5.1f} | S-lift {100 * g.lift.mean():5.1f}"
              f" | diff {100 * dd.mean():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f}, {100 * np.percentile(bs, 97.5):+5.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=2000); a = ap.parse_args()
    llm_part(a.B); retrieval_part(5000)
