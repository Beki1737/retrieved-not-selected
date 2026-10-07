"""E8b (exploratory follow-up, D84): E8 found that similar unanimous documents cut stratified-50 R@10 more than bge R@10
(H15c not supported). Hypothesis from D64: stratified recall beyond rank 10 lives at ranks 11-50, exactly where similar
dilution lands, so stratifying deeper should restore robustness. Same pools, candidates, arms and seeds as E8; adds
stratification depth 200 and the whole pool, and the paired similar-vs-random test at K=100 that H15b needs."""
import os, zlib
os.environ.setdefault("CKG_VERSION", "v1_2")
import numpy as np, pandas as pd
from coalrag.data.configB import load, match_vec
from coalrag.retrieval.configA import ConfigA

Kdf, T, TOK, V, H, EMB = load(partial_veto_coding=True, with_emb=True)
E = EMB["bge-small"]; rid = Kdf.res_id.tolist(); pos = {r: i for i, r in enumerate(rid)}
dates = Kdf.date.values; REG = Kdf.regime.values
cfg = ConfigA(); A = [pos[r["res_id"]] for r in cfg.A if r["res_id"] in pos]; N = [pos[r["res_id"]] for r in cfg.N if r["res_id"] in pos]
choi = set(A) | set(N); complete = lambda i: bool(((V[i] >= 1) & (V[i] <= 3)).all())
cand = np.array([i for i in range(len(rid)) if i not in choi and dates[i] < np.datetime64("2013-01-01") and complete(i)])
MODE = {"unanimous": cand[(V[cand] == 1).all(1)], "dissent": cand[~(V[cand] == 1).all(1)]}
SEEDS = [0, 1, 2]


def strat(order, Vp, depth, k=10):
    seen, out = set(), []
    for j in (order if depth is None else order[:depth]):
        key = tuple(Vp[j])
        if key not in seen:
            seen.add(key); out.append(j)
        if len(out) == k: break
    return np.array(out, dtype=int)


def measure(q, pool):
    pool = np.asarray(pool, dtype=int); h, _ = match_vec(V[pool], V[q], 0, "answer"); o = np.argsort(-(E[pool] @ E[q]), kind="stable")
    r = dict(sup=bool(h.any()), bge=bool(h[o[:10]].any()))
    for d in (50, 200, None): r[f"strat{d or 'all'}"] = bool(h[strat(o, V[pool], d)].any())
    return r


def added(q, mode, k, how, sd=0):
    C = MODE[mode]; k = min(k, len(C))
    if how == "similar": return C[np.argsort(-(E[C] @ E[q]), kind="stable")[:k]].tolist()
    return np.random.default_rng(zlib.crc32(f"{rid[q]}|{mode}|{k}|{sd}".encode())).choice(C, k, replace=False).tolist()


rows = []
for base, queries, src in [("F", N, A), ("U", sorted(choi), sorted(choi))]:
    for q in queries:
        if not complete(q) or (base == "U" and REG[q] not in ("split", "vetoed", "failed")): continue
        bp = [j for j in src if dates[j] < dates[q]]
        if len(bp) < 10: continue
        arms = [("none", [[]])]
        arms += [(f"{m}-similar-{k}", [added(q, m, k, "similar")]) for m in MODE for k in (100, 400)]
        arms += [(f"{m}-random-{k}", [added(q, m, k, "random", s) for s in SEEDS]) for m in MODE for k in (100, 400)]
        for arm, adds in arms:
            ms = pd.DataFrame([measure(q, bp + a) for a in adds]).mean()
            rows.append(dict(base=base, qid=rid[q], regime=REG[q], arm=arm, **ms.to_dict()))
df = pd.DataFrame(rows); rng = np.random.default_rng(0)


def pd_(sub, a, b, col, B=5000):
    x = sub[sub.arm == a].set_index("qid")[col]; y = sub[sub.arm == b].set_index("qid")[col]; d = (x - y).dropna().values
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return f"{100 * d.mean():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f},{100 * np.percentile(bs, 97.5):+5.1f}]"


U = df[df.base == "U"]; F = df[df.base == "F"]
print(f"==== E8b, unfiltered pool, contested drafts (n={U[U.arm == 'none'].qid.nunique()}): R@10 % by stratification depth ====")
cols = ["bge", "strat50", "strat200", "stratall"]
print((100 * U.groupby("arm")[cols].mean()).round(1).loc[["none", "unanimous-similar-100", "unanimous-similar-400", "unanimous-random-100", "unanimous-random-400"]].to_string())
print("-- change vs no augmentation (pp, 95% CI) --")
for arm in ["unanimous-similar-100", "unanimous-similar-400"]:
    print(f"{arm:22s} " + " | ".join(f"{c} {pd_(U, arm, 'none', c)}" for c in cols))
print(f"\n==== H15b, outcome-filtered pool (n={F[F.arm == 'none'].qid.nunique()}): dissent similar minus random at K=100 (pp, 95% CI) ====")
print(" | ".join(f"{c} {pd_(F, 'dissent-similar-100', 'dissent-random-100', c)}" for c in ["sup", "bge", "strat50", "stratall"]))
if len(MODE["dissent"]) < 400: print(f"note: only {len(MODE['dissent'])} dissent-bearing candidates exist, so K=400 adds all of them in both topic arms (not a valid quantity level)")
