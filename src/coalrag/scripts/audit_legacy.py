import os, pickle
from pathlib import Path
import numpy as np, pandas as pd

PROJ = Path(os.environ.get("PROJ", Path.home() / "coalrag"))
LEGACY = PROJ / "legacy"
find = lambda name: next(LEGACY.rglob(name))

class _Stub:
    def __init__(self, *a, **k): self._init_args, self._init_kwargs = a, k
    def __setstate__(self, st): self._state = st
MISSING = set()
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            MISSING.add(f"{module}.{name}")
            return type(name, (_Stub,), {"__module__": module})

def short(v, n=110):
    s = repr(v); return s if len(s) <= n else s[:n] + "..."

def describe(o, name="root", d=0, maxd=3, maxk=6):
    pad = "  " * d
    if isinstance(o, pd.DataFrame):
        print(f"{pad}{name}: DataFrame{o.shape} cols={list(o.columns)[:40]}")
        if len(o): print(f"{pad}  row0: {short(o.iloc[0].to_dict(), 300)}")
    elif isinstance(o, pd.Series):
        print(f"{pad}{name}: Series[{len(o)}] {o.dtype} head={short(o.head(3).to_dict())}")
    elif isinstance(o, np.ndarray):
        print(f"{pad}{name}: ndarray{o.shape} {o.dtype}")
    elif type(o).__name__ == "Tensor":
        print(f"{pad}{name}: Tensor{tuple(o.shape)} {o.dtype}")
    elif isinstance(o, dict):
        print(f"{pad}{name}: dict[{len(o)}]")
        if d < maxd:
            for k in list(o)[:maxk]: describe(o[k], short(k, 60), d + 1, maxd, maxk)
            if len(o) > maxk: print(f"{pad}  ... +{len(o)-maxk} keys, e.g. {short(list(o)[maxk:maxk+5])}")
    elif isinstance(o, (list, tuple, set)):
        seq = list(o); print(f"{pad}{name}: {type(o).__name__}[{len(seq)}]")
        if seq and d < maxd: describe(seq[0], "[0]", d + 1, maxd, maxk)
    elif isinstance(o, _Stub):
        print(f"{pad}{name}: STUB {type(o).__module__}.{type(o).__name__}")
        st = getattr(o, "_state", None)
        if st is not None and d < maxd: describe(st, "state", d + 1, maxd, maxk)
    elif isinstance(o, (str, int, float, bool, type(None), np.generic)):
        print(f"{pad}{name}: {type(o).__name__} = {short(o)}")
    else:
        print(f"{pad}{name}: {type(o).__module__}.{type(o).__name__} attrs={list(getattr(o, '__dict__', {}))[:15]}")

print("######## PICKLES")
for f in ["canonical_state_v3.pkl", "canonical_state_v2.pkl", "canonical_state_v1.pkl",
          "dense_embeddings_v1.pkl", "speech_embeddings_v1.pkl"]:
    p = find(f); print(f"\n==== {f} ({p.stat().st_size/1e6:.1f} MB)")
    try:
        with open(p, "rb") as fh: describe(SafeUnpickler(fh).load(), maxd=3 if "v3" in f else 2)
    except Exception as e:
        print("LOAD ERROR:", type(e).__name__, e)
print("\nSTUBBED CLASSES:", sorted(MISSING))

print("\n######## UN DL SC VOTING CSV")
sv = pd.read_csv(find("2026_02_06_sc_voting.csv"), dtype=str, keep_default_na=False)
print("shape", sv.shape, "| resolutions", sv.resolution.nunique(), "| dates", sv.date.min(), "->", sv.date.max())
print("ms_vote:", sv.ms_vote.value_counts().to_dict())
print("modality:", sv.modality.value_counts().to_dict())
print("rows per resolution:", sv.groupby("resolution").size().value_counts().sort_index().to_dict())
p5 = sv[sv.permanent_member.str.lower().isin(["true", "1"])]
print("P5 codes (seat history):\n", p5.groupby(["ms_code", "ms_name"]).date.agg(["min", "max", "count"]).to_string())
nN = p5.groupby("resolution").ms_vote.apply(lambda s: (s == "N").sum())
viol = nN[nN > 0]
print(f"\nART 27(3) CHECK: adopted resolutions with >=1 P5 'N' = {len(viol)}")
if len(viol):
    print(p5[p5.resolution.isin(viol.index) & (p5.ms_vote == "N")][["resolution", "date", "ms_code", "modality", "vote_note"]].head(25).to_string())
w = p5.pivot_table(index=["resolution", "date"], columns="ms_code", values="ms_vote", aggfunc="first").reset_index()
pc = [c for c in w.columns if c not in ("resolution", "date")]
def regime(r):
    v = [x for x in r[pc] if isinstance(x, str) and x != ""]
    if not v: return "no_recorded_vote"
    return "consensus" if all(x == "Y" for x in v) else ("p5_N" if "N" in v else "split")
w["regime"] = w.apply(regime, axis=1)
for lo, hi in [("2013-01-01", "2026-01-30"), ("2013-01-01", "2024-12-31")]:
    m = (w.date >= lo) & (w.date <= hi)
    print(f"adopted {lo}..{hi}: n={m.sum()} regimes={w[m].regime.value_counts().to_dict()}")

print("\n######## DPPA VETO CSV")
vt = pd.read_csv(find("DPPA-SCVETOES.csv"), dtype=str, keep_default_na=False)
vt["date"] = pd.to_datetime(vt.date, errors="coerce")
print("shape", vt.shape, "| unique drafts", vt["draft_res#"].nunique(), "| blank draft#", (vt["draft_res#"] == "").sum(),
      "| dates", vt.date.min().date(), "->", vt.date.max().date())
print("#_of_pms:", vt["#_of_pms"].value_counts().to_dict())
for lo, hi in [("1946-01-01", "2026-01-30"), ("2013-01-01", "2024-12-31"), ("2013-01-01", "2026-01-30"), ("2026-01-31", "2030-01-01")]:
    m = (vt.date >= lo) & (vt.date <= hi)
    print(f"vetoes {lo}..{hi}: rows={m.sum()} unique drafts={vt[m]['draft_res#'].nunique()}")
print("per year since 2013:", vt[vt.date >= "2013-01-01"].groupby(vt.date.dt.year).size().to_dict())
