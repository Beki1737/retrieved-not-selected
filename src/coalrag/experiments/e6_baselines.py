"""E6 non-LLM baselines (design review item; D79) on the same 66 non-adopted drafts x 5 nations as the LLM runs, written in
the LLM row schema so every LLM metric applies. Only precedents strictly earlier than the draft, Config A pools (A + N).
  BL-majority            always favour
  BL-nation-all          the nation's vote distribution over all earlier Choi documents (adopted + non-adopted)
  BL-nation-nonadopted   the nation's vote distribution over earlier non-adopted drafts (in-distribution history)
  BL-copy-bge1           the nation's vote on the single most text-similar earlier precedent (bge-small)
  BL-knn10               rank-weighted (1/r) vote distribution of the nation over the 10 most similar precedents
  BL-strat4-vote         rank-weighted vote distribution over exactly the 4 stratified precedents shown in D-strat4-coal
Probabilities carry a fixed pseudo-count of 0.02 per option so that log-loss and AUROC are defined (no tuning)."""
import json, os
from pathlib import Path
import numpy as np
from coalrag.core.match import P5
from coalrag.retrieval.configA import ConfigA

STORE = Path(os.environ["PROJ"]) / "results" / "store"; OUT = STORE / "e4_baselines_main.jsonl"
L = ["Y", "N", "A"]; EPS = 0.02

def dist(votes, w=None):
    c = np.full(3, EPS)
    for i, v in enumerate(votes):
        if v in L: c[L.index(v)] += 1.0 if w is None else w[i]
    return c / c.sum()

cfg = ConfigA(); E = cfg.E["bge-small"]; rows = []
for q in sorted(cfg.N, key=lambda r: r["date"]):
    el = cfg.eligible(q, cfg.A) + cfg.eligible(q, cfg.N); elN = cfg.eligible(q, cfg.N); qv = E[cfg.pos[q["res_id"]]]
    top = [el[i] for i in np.argsort(-np.array([E[cfg.pos[c["res_id"]]] @ qv for c in el]), kind="stable")[:10]] if el else []
    strat = list(cfg.strat_slots(q, 4, True))
    for n in P5:
        t = cfg.vot[q["res_id"]][n]
        if t not in L: continue
        V = lambda docs: [cfg.vot[d["res_id"]][n] for d in docs]
        B = {"BL-majority": np.array([1 - 2 * EPS, EPS, EPS]), "BL-nation-all": dist(V(el)), "BL-nation-nonadopted": dist(V(elN)),
             "BL-copy-bge1": dist(V(top[:1])), "BL-knn10": dist(V(top), [1 / (r + 1) for r in range(len(top))]),
             "BL-strat4-vote": dist(V(strat), [1 / (r + 1) for r in range(len(strat))])}
        for c, p in B.items():
            rows.append(dict(qid=q["res_id"], symbol=q["symbol"], qset="vetoed" if q["vetoed"] else "failed", nation=n, cond=c, model="baseline",
                             pY=float(p[0]), pN=float(p[1]), pA=float(p[2]), pred=L[int(np.argmax(p))], true=t, retrieved=[], prompt_tokens=None, sec=0.0, error=None))
with open(OUT, "w") as f:
    for r in rows: f.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows ({len(rows) // 6} per baseline) to {OUT.name}")
