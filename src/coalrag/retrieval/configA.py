import os, re, json
from pathlib import Path
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi
from coalrag.core.match import P5, norm_entropy
from coalrag.retrieval.lexical import lt_point, tokenize, doc_text

PROJ = Path(os.environ["PROJ"]); PROC = PROJ / "data" / "processed"; DS = PROJ / "data" / "raw" / "Nation-Level_Bias" / "Dataset"
NAME = {"CHN": "China", "FRA": "France", "GBR": "United Kingdom", "RUS": "Russian Federation", "USA": "United States"}

class ConfigA:
    def __init__(self):
        ckg = pd.read_parquet(PROC / "ckg_v1_1.parquet")
        self.A = ckg[ckg.in_choi_adopted].to_dict("records"); self.N = ckg[ckg.in_choi_test].to_dict("records")
        idx = pd.read_parquet(PROC / "emb_configA_index.parquet").res_id.tolist(); self.pos = {r: i for i, r in enumerate(idx)}
        self.E = {m: np.load(PROC / f"emb_configA_{m}.npy") for m in ["bge-small", "mxbai-large"]}
        self.tok = {r["res_id"]: tokenize(doc_text(r)) for r in self.A + self.N}
        self.vot = {r["res_id"]: {n: (r[n] if isinstance(r[n], str) else None) for n in P5} for r in self.A + self.N}
        ca = json.load(open(DS / "adopted_resolutions_dataset.json", encoding="utf-8"))
        cn = json.load(open(DS / "not_adopted_resolutions_dataset.json", encoding="utf-8"))
        rn = lambda s: int((re.search(r"RES/\s*(\d+)", str(s)) or re.search(r"(\d+)", str(s))).group(1))
        num2rid = {int(r["resnum"]): r["res_id"] for r in self.A}; sym2rid = {r["symbol_norm"]: r["res_id"] for r in self.N}
        self.speech = {}
        for r in ca: self.speech[num2rid[rn(r["res_no"])]] = r.get("speech") or {}
        for r in cn: self.speech[sym2rid[re.sub(r"\s+", "", r["res_no"]).upper()]] = r.get("speech") or {}
        self._rk = {}

    def eligible(self, q, pool, strict=True):
        return [c for c in pool if c["symbol_norm"] != q["symbol_norm"] and (c["date"] < q["date"] if strict else c["date"] <= q["date"])]

    def rank(self, q, pool_name, method):
        key = (q["res_id"], pool_name, method)
        if key in self._rk: return self._rk[key]
        P = self.A if pool_name == "A" else self.N
        el = self.eligible(q, P, strict=not method.endswith("orig")); out = []
        if el and method.startswith(("LT", "CSR3")):
            fixed = not method.endswith("orig"); sc = []
            for c in el:
                p, reg = lt_point(q, c, fixed)
                if not fixed and pool_name == "N" and not reg: continue
                sc.append((p, c["date"], c))
            sc.sort(key=lambda t: (t[0], t[1]), reverse=True)
            if method.startswith("CSR3"):
                f = lambda t: 1.0 * norm_entropy(self.vot[t[2]["res_id"]]) + 0.3 * 0.5 ** ((q["date"] - t[2]["date"]).days / 1825) + 0.2 * min(t[0] / 5, 1.0)
                sc = sorted(sc[:50], key=f, reverse=True)
            out = [(c, p) for p, _, c in sc]
        elif el:
            if method == "BM25": s = BM25Okapi([self.tok[c["res_id"]] for c in el]).get_scores(self.tok[q["res_id"]])
            else:
                E = self.E[method]; qv = E[self.pos[q["res_id"]]]; s = np.array([E[self.pos[c["res_id"]]] @ qv for c in el])
            o = np.argsort(-np.asarray(s), kind="stable"); out = [(el[i], float(s[i])) for i in o]
        self._rk[key] = out; return out

    def choi_docs(self, q, method):
        """Choi harness: k=1 per pool; LT variants apply threshold=3; sorted by date."""
        docs = []
        for pool in ("A", "N"):
            r = self.rank(q, pool, method)
            if not r: continue
            c, score = r[0]
            if method in ("LT-orig", "LT") and score < 3: continue
            docs.append(c)
        return sorted(docs, key=lambda d: d["date"])

    def oracle(self, q):
        """Most query-similar earlier precedent whose full 5-vote configuration equals the query's."""
        M = self.vot[q["res_id"]]
        ok = [c for c in self.eligible(q, self.A) + self.eligible(q, self.N)
              if all(M[n] is not None and self.vot[c["res_id"]][n] == M[n] for n in P5)]
        if not ok: return None
        E = self.E["bge-small"]; qv = E[self.pos[q["res_id"]]]
        return max(ok, key=lambda c: float(E[self.pos[c["res_id"]]] @ qv))

def strat_slots(self, q, k=4, stratify=True):
    """Coalition-Stratified Retrieval over Choi's pools (A and N): bge top-50, keep the best-ranked precedent per
    distinct 5-vote configuration (stratify=True) or plain top-k (stratify=False). Returned in rank order."""
    el = self.eligible(q, self.A) + self.eligible(q, self.N)
    if not el: return []
    E = self.E["bge-small"]; qv = E[self.pos[q["res_id"]]]
    o = np.argsort(-np.array([E[self.pos[c["res_id"]]] @ qv for c in el]), kind="stable")[:50]
    if not stratify: return [el[i] for i in o[:k]]
    seen, out = set(), []
    for i in o:
        key = tuple(self.vot[el[i]["res_id"]][n] for n in P5)
        if key in seen: continue
        seen.add(key); out.append(el[i])
        if len(out) == k: break
    return out
ConfigA.strat_slots = strat_slots
