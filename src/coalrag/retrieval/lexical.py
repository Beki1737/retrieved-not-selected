import re
EXCL = {"Member States", "United Nations"}

def kwset(k):
    if k is None: return set()
    if hasattr(k, "tolist"): k = k.tolist()
    if isinstance(k, (list, tuple)): return {str(s).strip() for s in k}
    return {s.strip() for s in str(k).split(",")}

def tset(x):
    if x is None: return set()
    if hasattr(x, "tolist"): x = x.tolist()
    return set(x) if isinstance(x, (list, tuple, set)) else set()

def lt_point(q, c, fixed: bool):
    """Choi et al. KeywordRetriever score. fixed=False reproduces the released code
    (keyword term compares the candidate with itself); fixed=True uses query-candidate overlap."""
    region_eq = q.get("choi_region") is not None and q.get("choi_region") == c.get("choi_region")
    point = 2.0 if region_eq else 0.0
    point += len(tset(q.get("target_nations")) & (tset(c.get("target_nations")) - EXCL))
    qk = kwset(q.get("keyword")) if fixed else kwset(c.get("keyword"))
    point += len((qk & kwset(c.get("keyword"))) - EXCL) / 10
    return point, region_eq

def tokenize(s):
    return re.findall(r"[a-z0-9]+", str(s).lower())

def doc_text(r):
    """Config A retrieval text (D26): summary + action_item + keyword, as in v1."""
    return " ".join(str(r.get(k)) for k in ("summary", "action_item", "keyword") if isinstance(r.get(k), str) and r.get(k))
