import math
from typing import Mapping, Optional
P5 = ("CHN", "FRA", "GBR", "RUS", "USA")

def _v(x):
    if x is None or (isinstance(x, float) and x != x): return None
    return x

def comparable(M: Mapping, C: Mapping, n: str) -> list:
    """Non-target members recorded in both the query vector M and the candidate C."""
    return [j for j in P5 if j != n and _v(M.get(j)) is not None and _v(C.get(j)) is not None]

def match_exact(M, C, n, m_min: int = 3) -> Optional[bool]:
    """Primary rule (D1, D34). Observed disagreement -> False; no disagreement and < m_min comparable -> None."""
    comp = comparable(M, C, n)
    if any(M[j] != C[j] for j in comp): return False
    if len(comp) < m_min: return None
    return True

def match_answer(M, C, n, m_min: int = 3) -> Optional[bool]:
    """Coalition match AND candidate's target vote equals the true target vote."""
    base = match_exact(M, C, n, m_min)
    if base is None: return None
    if not base: return False
    if _v(C.get(n)) is None or _v(M.get(n)) is None: return None
    return C[n] == M[n]

def match_legacy(M, C, n, thr: float = 0.75) -> Optional[bool]:
    """v1 rule, appendix sensitivity only."""
    comp = comparable(M, C, n)
    if not comp: return None
    return sum(M[j] == C[j] for j in comp) / len(comp) >= thr

RULES = {"exact": match_exact, "answer": match_answer, "legacy": match_legacy}

def resolve(x, policy: str):
    """Undefined-pair policy: lower = non-match, upper = match, drop = skip."""
    if x is None: return {"drop": None, "lower": False, "upper": True}[policy]
    return x

def norm_entropy(C: Mapping) -> float:
    """CSR-v3 entropy over recorded P5 votes (convention a), normalized by log2(3)."""
    vals = [v for v in (_v(C.get(j)) for j in P5) if v is not None]
    if not vals: return 0.0
    ps = [vals.count(v) / len(vals) for v in set(vals)]
    return -sum(p * math.log2(p) for p in ps) / math.log2(3)
