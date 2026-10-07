"""Engine-consistency check across runs (D91): an E7 slot row and an F-orc-coal row of the wave-2 run that show the same single
precedent (coalition view) to the same (draft, nation) should be the same prompt, so their probabilities should agree.
Usage: python -m coalrag.tools.samecheck"""
import json, os
from pathlib import Path
import numpy as np

STORE = Path(os.environ["PROJ"]) / "results" / "store"
for m in sorted(p.name[3:-13] for p in STORE.glob("e4_*_select.jsonl")):
    w2 = STORE / f"e4_{m}_wave2.jsonl"
    if not w2.exists(): print(f"{m}: no wave-2 file"); continue
    F = {}
    for r in map(json.loads, open(w2)):
        if r.get("cond") == "F-orc-coal" and not r.get("error") and r.get("retrieved"):
            F[(r["qid"], r["nation"], r["retrieved"][0])] = r
    S = [r for r in map(json.loads, open(STORE / f"e4_{m}_select.jsonl")) if not r.get("error") and r.get("retrieved")]
    pairs = [(s, F[k]) for s in S if (k := (s["qid"], s["nation"], s["retrieved"][0])) in F]
    if not pairs: print(f"{m}: no slot row shares its precedent with an F-orc-coal row"); continue
    a = np.array([[s["pY"], s["pN"], s["pA"]] for s, _ in pairs], float); b = np.array([[f["pY"], f["pN"], f["pA"]] for _, f in pairs], float)
    same = np.mean([s.get("prompt_tokens") == f.get("prompt_tokens") for s, f in pairs]); dp = np.abs(a - b)
    print(f"{m}: {len(pairs)} same-precedent pairs | same prompt length {same:.3f} | max |dp| {dp.max():.4f} | mean |dp| {dp.mean():.5f}"
          f" | argmax agreement {np.mean(a.argmax(1) == b.argmax(1)):.3f}")
