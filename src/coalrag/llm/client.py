import asyncio, math
from openai import AsyncOpenAI
from transformers import AutoTokenizer

OPTS = ["favour", "against", "abstention"]; LBL = {"favour": "Y", "against": "N", "abstention": "A"}

class LLM:
    """OpenAI-compatible client (vLLM). score() = normalized full-option sequence likelihood (D20)."""
    def __init__(self, served, path, base="http://127.0.0.1:8007/v1", conc=32):
        self.m = served; self.cli = AsyncOpenAI(base_url=base, api_key="EMPTY", timeout=900)
        self.tok = AutoTokenizer.from_pretrained(path); self.sem = asyncio.Semaphore(conc)

    def prefix(self, user):
        return self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)

    async def generate(self, user, max_tokens=400):
        async with self.sem:
            r = await self.cli.chat.completions.create(model=self.m, messages=[{"role": "user", "content": user}],
                                                       max_tokens=max_tokens, temperature=0, seed=0)
        return r.choices[0].message.content or "", r.usage.prompt_tokens

    async def _lp(self, prefix, pre_ids, opt):
        full = self.tok(prefix + opt, add_special_tokens=False).input_ids
        if full[:len(pre_ids)] != pre_ids: raise ValueError(f"tokenizer boundary merge for '{opt}'")
        k = len(full) - len(pre_ids)
        async with self.sem:
            r = await self.cli.completions.create(model=self.m, prompt=prefix + opt, max_tokens=1, temperature=0, echo=True,
                                                  logprobs=0, extra_body={"add_special_tokens": False})
        return sum(r.choices[0].logprobs.token_logprobs[-1 - k:-1])

    async def score(self, user):
        prefix = self.prefix(user); pre = self.tok(prefix, add_special_tokens=False).input_ids
        lps = await asyncio.gather(*[self._lp(prefix, pre, o) for o in OPTS])
        z = max(lps); w = [math.exp(v - z) for v in lps]; s = sum(w)
        return {LBL[o]: w[i] / s for i, o in enumerate(OPTS)}, len(pre)

async def verify_served(llm, expected_path):
    """Refuse to run if the served model is not the checkpoint this run claims to evaluate (D46)."""
    from pathlib import Path
    ms = await llm.cli.models.list()
    roots = {m.id: (getattr(m, "root", None) or (getattr(m, "model_extra", None) or {}).get("root")) for m in ms.data}
    root = roots.get(llm.m)
    if root is None or Path(root).resolve() != Path(expected_path).resolve():
        raise SystemExit(f"MODEL MISMATCH: served name '{llm.m}' -> {root}, expected {expected_path}. Aborting.")
    print(f"[verified] {llm.m} -> {root}")
    return root
