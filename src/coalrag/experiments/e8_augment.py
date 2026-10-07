"""E8 corpus augmentation with quantity and topic similarity controlled (design review item; pre-registered H15, D82).
Candidate documents: UNSC-CKG v1.2 records outside Choi et al.'s 581 documents, dated before 2013 (every query is from
2013 on), with all five permanent-member votes recorded. Mode: unanimous (all five favour) vs dissent-bearing (any against
or abstention). Topic: similar (the K candidates of that mode nearest to the query) vs random (K drawn uniformly, 3 seeds).
Quantity: K in {100, 400}. All similarities use one text space (CKG texts, bge-small) for every document, so arms differ
only in which documents are added.
  Base F (outcome-filtered): pool = Choi's adopted documents earlier than the query; queries = the non-adopted drafts.
  Base U (unfiltered):       pool = all Choi documents earlier than the query; queries = split, vetoed and failed drafts.
Metrics (exact five-vote configuration): Sup = any match in the pool; bge R@1 / R@10; stratified (top-50) R@10;
m50 = matches in the bge top-50. Paired bootstrap over queries against the un-augmented base."""
import os, zlib
os.environ.setdefault("CKG_VERSION", "v1_2")
from pathlib import Path
import numpy as np, pandas as pd
from coalrag.data.configB import load, match_vec
from coalrag.retrieval.configA import ConfigA

STORE = Path(os.environ["PROJ"]) / "results" / "store"
Kdf, T, TOK, V, H, EMB = load(partial_veto_coding=True, with_emb=True)
E = EMB["bge-small"]; rid = Kdf.res_id.tolist(); pos = {r: i for i, r in enumerate(rid)}
dates = Kdf.date.values; REG = Kdf.regime.values
cfg = ConfigA(); A = [pos[r["res_id"]] for r in cfg.A if r["res_id"] in pos]; N = [pos[r["res_id"]] for r in cfg.N if r["res_id"] in pos]
choi = set(A) | set(N); complete = lambda i: bool(((V[i] >= 1) & (V[i] <= 3)).all())
cand = np.array([i for i in range(len(rid)) if i not in choi and dates[i] < np.datetime64("2013-01-01") and complete(i)])
MODE = {"unanimous": cand[(V[cand] == 1).all(1)], "dissent": cand[~(V[cand] == 1).all(1)]}
print(f"candidates: {len(cand)} (unanimous {len(MODE['unanimous'])}, dissent-bearing {len(MODE['dissent'])}) | Choi docs mapped {len(choi)}")
KS, SEEDS = [100, 400], [0, 1, 2]


def strat(order, Vp, k=10):
    seen, out = set(), []
    for j in order[:50]:
        key = tuple(Vp[j])
        if key not in seen:
            seen.add(key); out.append(j)
        if len(out) == k: break
    return np.array(out, dtype=int)


def measure(q, pool):
    pool = np.asarray(pool, dtype=int); h, _ = match_vec(V[pool], V[q], 0, "answer")
    s = E[pool] @ E[q]; o = np.argsort(-s, kind="stable"); st = strat(o, V[pool])
    return dict(sup=bool(h.any()), r1=bool(h[o[:1]].any()), r10=bool(h[o[:10]].any()), s_r10=bool(h[st].any()), m50=int(h[o[:50]].sum()), n_pool=len(pool))


rows = []
for base, queries, poolsrc in [("F", N, A), ("U", sorted(choi), sorted(choi))]:
    for q in queries:
        if not complete(q) or (base == "U" and REG[q] not in ("split", "vetoed", "failed")): continue
        bp = [j for j in poolsrc if dates[j] < dates[q]]
        if len(bp) < 10: continue
        rows.append(dict(base=base, qid=rid[q], regime=REG[q], arm="none", K=0, mode="", topic="", seed=-1, **measure(q, bp)))
        for mode, C in MODE.items():
            sims = E[C] @ E[q]
            for k in KS:
                k_ = min(k, len(C))
                rows.append(dict(base=base, qid=rid[q], regime=REG[q], arm=f"{mode}-similar-{k}", K=k, mode=mode, topic="similar", seed=-1,
                                 **measure(q, bp + C[np.argsort(-sims, kind="stable")[:k_]].tolist())))
                for sd in SEEDS:
                    rng = np.random.default_rng(zlib.crc32(f"{rid[q]}|{mode}|{k}|{sd}".encode()))
                    rows.append(dict(base=base, qid=rid[q], regime=REG[q], arm=f"{mode}-random-{k}", K=k, mode=mode, topic="random", seed=sd,
                                     **measure(q, bp + rng.choice(C, k_, replace=False).tolist())))
df = pd.DataFrame(rows); df.to_parquet(STORE / "e8_augment.parquet", index=False)
q = df.groupby(["base", "qid", "regime", "arm"], as_index=False)[["sup", "r1", "r10", "s_r10", "m50"]].mean()  # average random seeds per query
rng = np.random.default_rng(0)


def delta(sub, arm, col, B=5000):
    a = sub[sub.arm == arm].set_index("qid")[col]; b = sub[sub.arm == "none"].set_index("qid")[col]; d = (a - b).dropna().values
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return f"{100 * d.mean():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f},{100 * np.percentile(bs, 97.5):+5.1f}]"


ARMS = ["none"] + [f"{m}-{t}-{k}" for m in MODE for t in ("similar", "random") for k in KS]
for base, lab, regs in [("F", "OUTCOME-FILTERED POOL (Choi adopted only) | non-adopted queries", ["vetoed", "failed"]),
                        ("U", "UNFILTERED POOL (all Choi documents) | contested queries", ["split", "vetoed", "failed"])]:
    sub = q[(q.base == base) & q.regime.isin(regs)]
    print(f"\n==== {lab} | n queries = {sub[sub.arm == 'none'].qid.nunique()} | %, change vs no augmentation [95% CI] ====")
    out = []
    for arm in ARMS:
        s = sub[sub.arm == arm]
        r = dict(arm=arm, Sup=100 * s.sup.mean(), bge_R10=100 * s.r10.mean(), strat_R10=100 * s.s_r10.mean(), m50=s.m50.mean())
        if arm != "none": r.update(dSup=delta(sub, arm, "sup"), dbge_R10=delta(sub, arm, "r10"), dstrat_R10=delta(sub, arm, "s_r10"))
        out.append(r)
    with pd.option_context("display.width", 220): print(pd.DataFrame(out).set_index("arm").round(1).fillna("").to_string())
    for reg in regs:
        s = sub[sub.regime == reg]
        print(f"  {reg:8s} n={s[s.arm == 'none'].qid.nunique():3d} | dissent-similar-400: dSup {delta(s, 'dissent-similar-400', 'sup')} dbge_R10 {delta(s, 'dissent-similar-400', 'r10')}"
              f" | unanimous-similar-400: dbge_R10 {delta(s, 'unanimous-similar-400', 'r10')} dstrat_R10 {delta(s, 'unanimous-similar-400', 's_r10')}")
