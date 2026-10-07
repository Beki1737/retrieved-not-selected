"""Preflight (D68): score 8 near-maximum-length prompts concurrently with the production scoring call (echo log-likelihood).
Exit 1 on any failure, so a server that would run out of memory on long prompts is never used for a real run.
Usage: python -m coalrag.tools.preflight <served> <base_url> <max_model_len>"""
import asyncio, sys
from coalrag.experiments import e5_wave2 as W
from coalrag.llm.client import LLM
served, base, maxlen = sys.argv[1], sys.argv[2], int(sys.argv[3])
llm = LLM(served, W.E4.MODELS[served], base=base, conc=8)
unit = "The Security Council reaffirms its commitment to the sovereignty and territorial integrity of all States. "
n_unit = len(llm.tok(unit, add_special_tokens=False).input_ids)
body = unit * max(1, (maxlen - 400) // n_unit)
async def go():
    return await asyncio.gather(*[llm.score(f"[{i}] {body}\nRespond with exactly one word: favour, against, or abstention.") for i in range(8)],
                                return_exceptions=True)
res = asyncio.run(go())
bad = [x for x in res if isinstance(x, Exception)]; ntok = max((x[1] for x in res if not isinstance(x, Exception)), default=0)
print(f"preflight {served}: {8 - len(bad)}/8 long prompts scored (prefix ~{ntok} tokens)" + (f" | first error: {repr(bad[0])[:240]}" if bad else ""), flush=True)
sys.exit(1 if bad else 0)
