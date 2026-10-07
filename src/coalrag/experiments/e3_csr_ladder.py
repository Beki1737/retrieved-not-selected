"""E3 (design review item): CSR-v3 -> CSR-v6 one factor at a time, crossed with three missing-vote conventions.
Config B on CKG v1.2; Stage 1 = BM25 top-50. Metrics: target-vote match at top-1 (v1 Table 3 metric), exact coalition
match at top-1/3, exact configuration at top-1. Informativeness profiles are leakage-safe (dev: pre-2008; test: pre-2013).
The v6 weight is tuned on 2008-2012 queries only."""
import os
os.environ.setdefault("CKG_VERSION", "v1_2")
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from coalrag.data.configB import load, match_vec, P5
from coalrag.legacy_io import load_pickle

PROJ = Path(os.environ["PROJ"]); STORE = PROJ / "results" / "store"
K, T, TOK, V, _, _ = load(partial_veto_coding=True, with_emb=False)
kg = load_pickle(next((PROJ / "legacy").rglob("canonical_state_v3.pkl")))["ckg_by_number"]
nz = lambda s: "".join(str(s).split()).upper()
TR = {nz(e["resolution_id"]): (e.get("theme") or "Other", e.get("geopolitical_region") or "Other") for e in kg.values()}
THEME = np.array([TR.get(s, ("Other", "Other"))[0] for s in K.symbol_norm]); REGN = np.array([TR.get(s, ("Other", "Other"))[1] for s in K.symbol_norm])
print(f"theme coverage {np.mean(THEME != 'Other'):.1%} | themes {len(set(THEME))} | regions {len(set(REGN))}")
dates = K.date.values; REGS = ["consensus", "split", "vetoed", "failed"]

def profile(cut):
    m = (K.date < cut).values; out = {}
    for ni in range(5):
        v = V[m, ni]; ok = (v >= 1) & (v <= 3); th, rg, vv = THEME[m][ok], REGN[m][ok], v[ok]
        dist = lambda mask: (int(mask.sum()), np.array([(vv[mask] == c).mean() for c in (1, 2, 3)])) if mask.sum() else (0, None)
        out[("nat", ni)] = dist(np.ones(len(vv), bool))
        for t in set(th): out[("th", t, ni)] = dist(th == t)
        for r in set(rg): out[("rg", r, ni)] = dist(rg == r)
    return out
def info_matrix(prof):
    I = np.zeros(V.shape)
    for j in range(len(V)):
        for ni in range(5):
            c = V[j, ni]
            if 1 <= c <= 3:
                e = next((d[1] for key in (("th", THEME[j], ni), ("rg", REGN[j], ni), ("nat", ni)) for d in [prof.get(key)] if d and d[0] >= 10), np.full(3, 1 / 3))
                I[j, ni] = 1.0 - e[c - 1]
    return I
I_dev, I_test = info_matrix(profile("2008-01-01")), info_matrix(profile("2013-01-01"))

def H_conv(rows, members, conv):
    Vm = rows[:, members]; c = np.stack([(Vm == k).sum(1) for k in (1, 2, 3)], 1).astype(float); rec = c.sum(1)
    if conv == "c":
        c += (Vm == 0).sum(1)[:, None] / 3.0; n4 = (Vm == 4).sum(1); c[:, 0] += n4 / 2; c[:, 2] += n4 / 2
    tot = c.sum(1, keepdims=True); p = np.where(tot > 0, c / np.maximum(tot, 1e-9), 0)
    H = -(np.where(p > 0, p * np.log2(np.where(p > 0, p, 1)), 0)).sum(1) / np.log2(3)
    return np.where(rec >= 3, H, 0.0) if conv == "b" else H

GRID = [round(x, 1) for x in np.arange(0, 1.01, 0.1)]
splits = {"dev": (K.date >= "2008-01-01") & (K.date < "2013-01-01"), "test": K.date >= "2013-01-01"}
rows = []
for split, dm in splits.items():
    I = I_dev if split == "dev" else I_test
    for qi in np.where(dm & K.regime.isin(REGS) & (K.n_rec == 5))[0]:
        cut = int(np.searchsorted(dates, dates[qi], side="left"))
        if cut == 0: continue
        s = BM25Okapi(TOK[:cut]).get_scores(TOK[qi]); top = np.argsort(-s, kind="stable")[:50]
        rec = 0.5 ** ((dates[qi] - dates[top]).astype("timedelta64[D]").astype(float) / 1825); txt = s[top] / (s[top].max() + 1e-9)
        for ni in range(5):
            tv = V[top, ni] == V[qi, ni]
            coal, _ = match_vec(V[top], V[qi], ni, "exact"); cfgm, _ = match_vec(V[top], V[qi], ni, "answer")
            oth = [j for j in range(5) if j != ni]
            for conv in ("a", "b", "c"):
                H5, Hn, Ii = H_conv(V[top], list(range(5)), conv), H_conv(V[top], oth, conv), I[top, ni]
                sc = {"L0 v3": H5 + 0.3 * rec + 0.2 * txt, "L1 -target entropy": Hn + 0.3 * rec + 0.2 * txt,
                      "L2 +informativeness": Hn + 0.3 * rec + 0.2 * txt + 0.3 * Ii, "L3 -recency": Hn + 0.2 * txt + 0.3 * Ii,
                      "L4 -text (=v6)": 0.7 * Hn + 0.3 * Ii, **{f"w={w}": (1 - w) * Hn + w * Ii for w in GRID}}
                orders = {"BM25": np.arange(len(top)), **{k: np.argsort(-v, kind="stable") for k, v in sc.items()}}
                for name, o in orders.items():
                    rows.append(dict(split=split, qid=K.res_id[qi], regime=K.regime[qi], nation=P5[ni], conv=conv, method=name,
                                     tv1=bool(tv[o[0]]), coal1=bool(coal[o[0]]), coal3=bool(coal[o[:3]].any()), cfg1=bool(cfgm[o[0]])))
df = pd.DataFrame(rows); df.to_parquet(STORE / "e3_csr_ladder.parquet", index=False)
dev = df[(df.split == "dev") & (df.conv == "a") & df.method.str.startswith("w=")]
obj = dev.groupby(["method", "regime"]).tv1.mean().groupby("method").mean()
wbest = obj.idxmax(); print(f"dev regimes: {df[df.split == 'dev'].drop_duplicates('qid').regime.value_counts().to_dict()} | tuned v6 weight: {wbest} (dev tv1 {obj.max():.3f})")
t = df[(df.split == "test")].copy(); t.loc[t.method == wbest, "method"] = f"L5 v6 tuned ({wbest})"
LAD = ["BM25", "L0 v3", "L1 -target entropy", "L2 +informativeness", "L3 -recency", "L4 -text (=v6)", f"L5 v6 tuned ({wbest})"]
pc = lambda x: f"{100 * x:.0f}"
for lab, sel in [("WESTERN", ["FRA", "GBR", "USA"]), ("ALL nations", P5)]:
    a = t[(t.conv == "a") & t.nation.isin(sel) & t.method.isin(LAD)]
    g = a.groupby(["method", "regime"])[["tv1", "coal1", "coal3"]].mean()
    tab = g.apply(lambda r: f"{pc(r.tv1)}/{pc(r.coal1)}/{pc(r.coal3)}", axis=1).unstack("regime")[REGS].reindex(LAD)
    print(f"\n==== LADDER (test 2013-2026, convention a) | {lab} | target-vote@1 / coalition@1 / coalition@3 (%) ====\n{tab.to_string()}")
rng = np.random.default_rng(0)
def paired(fr, m1, m2, col, B=5000):
    x = fr[fr.method == m1].set_index(["qid", "nation"])[col].astype(int); y = fr[fr.method == m2].set_index(["qid", "nation"])[col].astype(int)
    d = (x - y).groupby(level=0).agg(["sum", "count"]); s, n = d["sum"].values, d["count"].values
    ix = rng.integers(0, len(s), (B, len(s))); bs = s[ix].sum(1) / n[ix].sum(1)
    return f"{100 * s.sum() / n.sum():+5.1f} [{100 * np.percentile(bs, 2.5):+5.1f},{100 * np.percentile(bs, 97.5):+5.1f}]"
w = t[(t.conv == "a") & t.nation.isin(["FRA", "GBR", "USA"])]
print("\n==== one-factor steps, target-vote@1, WESTERN (pp, resolution-cluster bootstrap) ====")
for reg in REGS:
    r = w[w.regime == reg]
    print(f"{reg:9s} " + " | ".join(f"{LAD[i]}-{LAD[i-1].split()[0]} {paired(r, LAD[i], LAD[i-1], 'tv1')}" for i in range(2, len(LAD))))
print("\n==== missing-vote convention sensitivity, target-vote@1, WESTERN (%) ====")
cv = t[t.nation.isin(["FRA", "GBR", "USA"]) & t.method.isin(["L0 v3", "L4 -text (=v6)"])].groupby(["method", "conv", "regime"]).tv1.mean().unstack("regime")[REGS]
print((100 * cv).round(1).to_string())
print("\n[reference] v1 Table 3 target-vote match (consensus/split/contested): text 97.5/78.9/59.0; CSR-v3 70.7/82.1/11.5; CSR-v6 62.1/75.2/18.0")
