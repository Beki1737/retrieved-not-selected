"""Data accounting and one-ranker funnel (D98; design review item). Prints:
  A  the ranker code the LLM evidence used (ConfigA.strat_slots / rank source), so the funnel can be stated in that space
  B  document counts: the 581 Choi-covered documents by regime, with or without all five votes recorded, and why any are excluded
  C  row counts of every LLM run by query set (main, all-regime, select, guar) and drafts with partially recorded votes
  D  the retrieval funnel in the SAME ranker that produced the LLM evidence (strat_slots), exact five-vote configuration:
     support, top 50 / 10 / 4 / 1, stratified top 4 and top 10, by regime and for the 66 non-adopted drafts
  E  the definition check: exact configuration vs non-target coalition vs target-vote alignment, adopted-only pool, 66 drafts
Usage: python -m coalrag.tools.accounting"""
import inspect, os
os.environ.setdefault("CKG_VERSION", "v1_2")
from pathlib import Path
import numpy as np, pandas as pd
from coalrag.data.configB import load
from coalrag.retrieval.configA import ConfigA

P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]; STORE = Path(os.environ["PROJ"]) / "results" / "store"
cfg = ConfigA(); vot = cfg.vot
print("==== A. ranker used for LLM evidence ====")
for f in ("strat_slots", "rank", "oracle", "eligible"):
    try: print(inspect.getsource(getattr(ConfigA, f)))
    except Exception as e: print(f"{f}: source unavailable ({type(e).__name__})")

Kdf, *_ = load(partial_veto_coding=True, with_emb=True); reg = dict(zip(Kdf.res_id, Kdf.regime))
docs = cfg.A + cfg.N; conf = lambda r: tuple(vot[r][x] for x in P5); full = lambda r: all(vot[r][x] in ("Y", "N", "A") for x in P5)
rows = [dict(res_id=d["res_id"], adopted=d["adopted"], regime=reg.get(d["res_id"], "unmapped"), complete=full(d["res_id"]),
             missing=",".join(x for x in P5 if vot[d["res_id"]][x] not in ("Y", "N", "A"))) for d in docs]
B = pd.DataFrame(rows)
print(f"\n==== B. documents: {len(B)} (adopted {B.adopted.sum()}, non-adopted {(~B.adopted).sum()}) ====")
print(pd.crosstab(B.regime, B.complete, margins=True).rename(columns={True: "all 5 votes", False: "some unrecorded"}).to_string())
inc = B[~B.complete]
if len(inc): print("documents with unrecorded votes (excluded as queries and as exact matches):"); print(inc[["res_id", "regime", "missing"]].to_string(index=False))

print("\n==== C. LLM rows by run and query set ====")
for p in sorted(STORE.glob("e4_*_*.jsonl")):
    if "baselines" in p.name: continue
    d = pd.read_json(p, lines=True); d = d[d.error.isna()] if "error" in d else d
    if not len(d) or "qset" not in d: continue
    c = d.groupby("qset").agg(drafts=("qid", "nunique"), rows=("qid", "size")); conds = d.cond.nunique() if "cond" in d else 1
    print(f"{p.name:45s} conds {conds:3d} | " + " | ".join(f"{k}: {r.drafts} drafts, {r.rows} rows" for k, r in c.iterrows()))
    if p.name.endswith("_allreg.jsonl"):
        part = d.groupby("qid").nation.nunique(); print(f"   drafts with fewer than 5 scored members: {(part < 5).sum()} (scored only for members with a recorded vote)")

print("\n==== D. funnel in the ranker that produced the LLM evidence (exact five-vote configuration) ====")
F = []
for d in docs:
    r = d["res_id"]
    if not full(r) or r not in reg: continue
    ranked = list(cfg.strat_slots(d, 100000, False))
    if len(ranked) < 10: continue
    h = np.array([conf(x["res_id"]) == conf(r) for x in ranked]); s4 = cfg.strat_slots(d, 4, True); s10 = cfg.strat_slots(d, 10, True)
    F.append(dict(regime=reg[r], adopted=d["adopted"], pool=len(ranked), sup=h.any(), t50=h[:50].any(), t10=h[:10].any(), t4=h[:4].any(), t1=h[:1].any(),
                  s4=any(conf(x["res_id"]) == conf(r) for x in s4), s10=any(conf(x["res_id"]) == conf(r) for x in s10), m50=h[:50].sum()))
F = pd.DataFrame(F); cols = ["sup", "t50", "t10", "t4", "t1", "s4", "s10"]
t = (100 * F.groupby("regime")[cols].mean()).round(1); t.insert(0, "n", F.groupby("regime").size()); t["matches_in_top50"] = F.groupby("regime").m50.mean().round(2)
print(t.to_string())
for lab, x in (("contested (split+vetoed+failed)", F[F.regime.isin(["split", "vetoed", "failed"])]), ("non-adopted (vetoed+failed)", F[~F.adopted])):
    print(f"{lab:32s} n={len(x)} | " + " ".join(f"{c} {100 * x[c].mean():.1f}" for c in cols))

print("\n==== E. definitions on the outcome-filtered pool (515 adopted documents), 66 non-adopted drafts ====")
A = [d for d in cfg.A]; out = []
for q in cfg.N:
    r = q["res_id"]; el = [d for d in A if d["date"] < q["date"]]
    for n in P5:
        if vot[r][n] not in ("Y", "N", "A"): continue
        ex = any(conf(d["res_id"]) == conf(r) for d in el)
        others = [x for x in P5 if x != n]
        co = any(all(vot[d["res_id"]][x] == vot[r][x] for x in others if vot[r][x] in ("Y", "N", "A")) for d in el)
        tv = any(vot[d["res_id"]][n] == vot[r][n] for d in el)
        out.append(dict(nation=n, exact=ex, coalition=co, target_vote=tv))
E = pd.DataFrame(out)
print("share of (draft, member) rows whose adopted-only pool holds a precedent that agrees on ... (%)")
print((100 * E.groupby("nation")[["exact", "coalition", "target_vote"]].mean()).round(1).to_string())
print("all rows:", " ".join(f"{c} {100 * E[c].mean():.1f}" for c in ("exact", "coalition", "target_vote")))
