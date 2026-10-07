"""Cell 44 diagnostics: per-file and per-condition coverage (a key counts as done once any attempt succeeded),
top error messages, prompt-length profile from the 7B main run (same tokenizer as the 32B), Llama checkpoint format,
and the Choi paraphrase template audit."""
import os, json, collections
from pathlib import Path
PROJ = Path(os.environ["PROJ"]); S = PROJ / "results" / "store"
for p in sorted(S.glob("e4_*.jsonl")):
    done, msg = {}, collections.Counter()
    for l in open(p):
        r = json.loads(l); k = (r["qid"], r["nation"], r["cond"])
        if not r.get("error"): done[k] = True
        else: done.setdefault(k, False); msg[r["error"][:110]] += 1
    ok = collections.Counter(c for (q, n, c), v in done.items() if v); tot = collections.Counter(c for (q, n, c) in done)
    print(f"\n{p.name}: keys {len(done)} | done {sum(ok.values())} | still failing {len(done) - sum(ok.values())} | lines {sum(1 for _ in open(p))}")
    print("  " + "  ".join(f"{c}={ok[c]}/{tot[c]}" for c in sorted(tot)))
    for m, n in msg.most_common(3): print(f"  error x{n}: {m}")
p = S / "e4_qwen2.5-7b_main.jsonl"
if p.exists():
    t = collections.defaultdict(list)
    for l in open(p):
        r = json.loads(l)
        if r.get("prompt_tokens"): t[r["cond"]].append(r["prompt_tokens"])
    print("\nprompt tokens, 7B main run (Qwen2.5 tokenizer = 32B tokenizer): cond max | n>4000 | n>6000")
    print("  " + "  ".join(f"{c} {max(v)}|{sum(x > 4000 for x in v)}|{sum(x > 6000 for x in v)}" for c, v in sorted(t.items())))
M = Path("$HF_MODELS/MetaAI/Llama-3.3-70B-Instruct-FP8")
if M.exists():
    print("\nLlama checkpoint files (non-weight):", sorted(f for f in os.listdir(M) if not f.endswith(".safetensors")))
    q = M / "hf_quant_config.json"
    if q.exists(): print("hf_quant_config.json:", q.read_text()[:300].replace("\n", " "))
print("\nChoi paraphrase template audit:")
from coalrag.experiments.e5_wave2 import choi_paraphrases
choi_paraphrases(verbose=True)
