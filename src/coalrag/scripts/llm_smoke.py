import math, time
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from transformers import AutoTokenizer

MP = "$HF_MODELS/Qwen/Qwen2.5-7B-Instruct"; M = "qwen2.5-7b"
cli = OpenAI(base_url="http://127.0.0.1:8007/v1", api_key="EMPTY")
tok = AutoTokenizer.from_pretrained(MP)

OPTS = ["favour", "against", "abstention"]
DRAFT = ("The Security Council demands that all parties immediately cease hostilities in Syria and "
         "establishes an independent mechanism to investigate the use of chemical weapons, with a view to accountability.")

def msgs(country):
    return [{"role": "user", "content":
        f"You are a representative of {country} in the United Nations Security Council (UNSC). "
        f"You have to decide how to vote on the following <draft resolution>. "
        f"You only choose the vote among three options: favour, against or abstention.\n"
        f"<draft resolution>: {DRAFT}\nAnswer with exactly one word: favour, against, or abstention."}]

def option_probs(m):
    prefix = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
    pre_ids = tok(prefix, add_special_tokens=False).input_ids
    logp = {}
    for o in OPTS:
        full_ids = tok(prefix + o, add_special_tokens=False).input_ids
        assert full_ids[:len(pre_ids)] == pre_ids, f"tokenizer boundary merge for option {o}"
        k = len(full_ids) - len(pre_ids)
        r = cli.completions.create(model=M, prompt=prefix + o, max_tokens=1, temperature=0, echo=True,
                                   logprobs=0, extra_body={"add_special_tokens": False})
        lp = r.choices[0].logprobs.token_logprobs
        logp[o] = sum(lp[-1 - k:-1])
    z = max(logp.values()); w = {o: math.exp(v - z) for o, v in logp.items()}; s = sum(w.values())
    return {o: round(w[o] / s, 3) for o in OPTS}

print("== 1) chat + first-token top logprobs ==")
for c in ["France", "Russian Federation", "China", "United States", "United Kingdom"]:
    r = cli.chat.completions.create(model=M, messages=msgs(c), max_tokens=4, temperature=0, logprobs=True, top_logprobs=5)
    top = [(t.token, round(t.logprob, 2)) for t in r.choices[0].logprobs.content[0].top_logprobs]
    print(f"{c:20s} answer={r.choices[0].message.content!r:14s} first-token top5={top}")

print("\n== 2) full-option scoring via echo (P over favour/against/abstention) ==")
try:
    for c in ["France", "Russian Federation", "China", "United States", "United Kingdom"]:
        print(f"{c:20s} {option_probs(msgs(c))}")
except Exception as e:
    print("OPTION SCORING FAILED:", type(e).__name__, str(e)[:300])

print("\n== 3) throughput: 128 chat calls, 32 concurrent ==")
t = time.time()
with ThreadPoolExecutor(32) as ex:
    rs = list(ex.map(lambda i: cli.chat.completions.create(model=M, messages=msgs("France"), max_tokens=64, temperature=0.7, seed=i), range(128)))
dt = time.time() - t
ptok = sum(r.usage.prompt_tokens for r in rs); ctok = sum(r.usage.completion_tokens for r in rs)
print(f"{dt:.1f}s | {128/dt:.1f} req/s | prompt {ptok} tok, completion {ctok} tok | {ctok/dt:.0f} gen tok/s")
