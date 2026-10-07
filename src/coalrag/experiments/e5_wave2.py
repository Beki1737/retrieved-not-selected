"""Wave 2 (D60). Reuses the pre-registered e4 runner unchanged: main conditions are delegated to e4_llm.Runner.run_one,
new conditions build prompts here and are scored with the same option-likelihood method.
  F  evidence-intervention factorial on the oracle precedent (one precedent, identical text across arms)
  R  order and k robustness of stratified evidence
  P  Choi et al. paraphrase templates (auto-discovered in VoteTemplates.py), zero-shot and with stratified evidence
  M  memorization probe (draft symbol and date only)
  AR all-regime run (use --queries adopted)
  --think off|on  Qwen3 reasoning mode: off = empty think block; on = sampled reasoning (budget-forced), then scoring
Output: results/store/e4_{served}_{tag}.jsonl, same schema as e4 plus optional fields (shown, template, think, ...)."""
import argparse, ast, asyncio, json, math, os, random, re, string, time
from pathlib import Path
from coalrag.core.match import P5
from coalrag.retrieval.configA import ConfigA, NAME
from coalrag.llm.client import LLM, verify_served, OPTS, LBL
from coalrag.llm import prompts as PR
from coalrag.experiments import e4_llm as E4

PROJ = Path(os.environ["PROJ"]); STORE = PROJ / "results" / "store"; LOGS = PROJ / "logs"; HOME = Path.home()
for k, v in {"qwen2.5-32b-gptq-int8": HOME / "models/Qwen2.5-32B-Instruct-GPTQ-Int8",
             "mistral-small-24b-2501-fp8": HOME / "models/Mistral-Small-24B-Instruct-2501-FP8-dynamic",
             "qwen3-8b": HOME / "models/Qwen3-8B"}.items():
    E4.MODELS.setdefault(k, str(v))
WORD, ctx = E4.WORD, E4.ctx
FLIP = {"Y": "N", "N": "Y", "A": "N"}
MEM = ("You are a representative of {country} in the United Nations Security Council (UNSC).\n"
       "On {date}, the Security Council voted on draft resolution {symbol}.\n"
       "How did {country} vote on this draft? The vote must be one of three options: favour, against, or abstention.\n\n{fmt}")
GROUPS = {
    "F": ["F-orc-text", "F-orc-coal", "F-orc-flip", "F-orc-others", "F-rnd-coal", "F-rnd-orcv"],
    "R": ["R-strat4-rev", "R-strat4-shuf", "R-strat2-coal", "R-strat6-coal"],
    "M": ["M-symbol"],
    "AR": ["B0-std", "D-bge-coal", "D-strat4-coal"],
    "THINK": ["B0-std", "D-bge-coal", "D-random-coal", "D-oracle-coal", "D-strat4-coal", "F-orc-coal", "F-orc-flip"]}
SENT = "@@DRAFT@@"


def choi_paraphrases(limit=4, verbose=False):
    """String templates in Choi's VoteTemplates.py that (i) take a country and a draft, (ii) are single-turn vote prompts
    (no history or reflexion), (iii) differ from the templates already used in the main run. Keys are STABLE ids: the
    position of the string among all string constants in the file (D71), so changing this filter never renumbers P-conditions."""
    src = next((PROJ / "data/raw/Nation-Level_Bias").rglob("VoteTemplates.py"), None)
    if src is None: return {}
    found = []
    for node in ast.walk(ast.parse(src.read_text(encoding="utf-8"))):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)): continue
        tg = node.targets[0] if isinstance(node, ast.Assign) else node.target
        name = tg.id if isinstance(tg, ast.Name) else getattr(tg, "attr", "anon")
        val = node.value
        if isinstance(val, ast.Constant): items = [(name, val)]
        elif isinstance(val, (ast.List, ast.Tuple)): items = [(f"{name}[{i}]", v) for i, v in enumerate(val.elts)]
        elif isinstance(val, ast.Dict): items = [(f"{name}[{getattr(k, 'value', i)}]", v) for i, (k, v) in enumerate(zip(val.keys, val.values))]
        else: items = []
        found += [(n, v.value) for n, v in items if isinstance(v, ast.Constant) and isinstance(v.value, str)]
    norm = lambda t: re.sub(r"\s+", " ", t).strip()
    seen = {norm(t) for t in (PR.BASIC, PR.WITH_HISTORY, PR.WITHOUT_HISTORY, PR.REFLEXION)}; out = {}
    def classify(t):
        low = t.lower()
        if "against" not in low: return "no 'against'", None
        bad = [w for w in ("predecessor", "previous", "histor") if w in low]  # D74: "reflect" removed
        if bad: return f"history/reflexion word {bad}", None
        try: fields = {f for _, f, _, _ in string.Formatter().parse(t) if f}
        except ValueError: return "unparseable braces", None
        role = {}
        for f in fields:
            fl = f.lower()
            if "country" in fl or "nation" in fl: role[f] = "country"
            elif "format" in fl: role[f] = "fmt"
            elif any(w in fl for w in ("resolution", "query", "draft", "context")): role[f] = "draft"
            else: return f"unknown field {{{f}}}", None
        if "draft" not in role.values() or "country" not in role.values(): return f"fields {sorted(fields)}", None
        if norm(t) in seen: return "duplicate of a used template", None
        try: t.format(**{f: "x" for f in role})
        except Exception as e: return f"format error {type(e).__name__}", None
        return None, role
    for i, (name, t) in enumerate(found, 1):
        why, role = classify(t)
        if why is None and len(out) < limit: seen.add(norm(t)); out[i] = (name, t, role)
        if verbose: print(f"  #{i:<3d} {name[:38]:38s} {'ACCEPT -> P' + str(i) if i in out else 'reject: ' + (why or 'over limit'):42s} | {norm(t)[:80]}")
    return out


def para_prompt(tpl, role, country, q, blk=""):
    u = tpl.format(**{f: (country if r == "country" else PR.ANSWER_ONE_WORD if r == "fmt" else SENT) for f, r in role.items()})
    if "fmt" not in role.values(): u = u.rstrip() + "\n\n" + PR.ANSWER_ONE_WORD
    if blk:  # evidence goes before the draft header line, as in the main D-conditions
        L = u.split("\n"); i = next(j for j, l in enumerate(L) if SENT in l)
        if L[i].strip() == SENT and i > 0 and "draft" in L[i - 1].lower() and len(L[i - 1]) < 80: i -= 1
        L.insert(i, blk.rstrip("\n")); u = "\n".join(L)
    return u.replace(SENT, ctx(q))


def block(cfg, slots, n, view, over=None):
    """Identical layout to prompts.precedent_block, plus view 'others' and displayed-vote overrides (res_id -> votes)."""
    if not slots: return ""
    L = ["Consider the following similar past draft resolutions in <similar_past_resolutions>.", "<similar_past_resolutions>"]
    for i, d in enumerate(slots, 1):
        L.append(f"[{i}] Date: {d['date'].date()} | Outcome: {'adopted' if d['adopted'] else 'not adopted'}")
        L.append(f"Summary: {d.get('summary') or ''}")
        v = (over or {}).get(d["res_id"], cfg.vot[d["res_id"]])
        if view == "target": L.append(f"Vote of {NAME[n]}: {WORD[v[n]]}")
        if view == "coalition": L.append("Votes of the permanent members: " + "; ".join(f"{NAME[m]}: {WORD[v[m]]}" for m in P5))
        if view == "others": L.append("Votes of the other permanent members: " + "; ".join(f"{NAME[m]}: {WORD[v[m]]}" for m in P5 if m != n))
    L.append("</similar_past_resolutions>")
    return "\n".join(L) + "\n"


def d_prompt(country, q, blk):
    """Exactly the e4 D-condition construction: WITHOUT_HISTORY with the evidence block before the draft header."""
    u = PR.WITHOUT_HISTORY.format(country=country, answer_format=PR.ANSWER_ONE_WORD, query=ctx(q))
    return u.replace("<draft_resolution> :", blk + "\n<draft_resolution> :", 1)


async def score_prefix(llm, prefix):
    pre = llm.tok(prefix, add_special_tokens=False).input_ids
    lps = await asyncio.gather(*[llm._lp(prefix, pre, o) for o in OPTS])
    z = max(lps); w = [math.exp(v - z) for v in lps]; s = sum(w)
    return {LBL[o]: w[i] / s for i, o in enumerate(OPTS)}, len(pre)


async def think_prefix(llm, user, mode, budget):
    base = llm.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=(mode == "on"))
    if mode == "off": return base, {}
    async with llm.sem:
        r = await llm.cli.completions.create(model=llm.m, prompt=base, max_tokens=budget, temperature=0.6, top_p=0.95, seed=0,
                                             stop=["</think>"], extra_body={"add_special_tokens": False, "top_k": 20})
    c = r.choices[0]; txt = c.text or ""
    if "<think>" not in txt[:20]: txt = "<think>\n" + txt.lstrip()
    return base + txt.rstrip() + "\n</think>\n\n", {"reason_tokens": r.usage.completion_tokens, "truncated": c.finish_reason == "length"}


class W2(E4.Runner):
    def __init__(self, llm, cfg, out, think="none", budget=1024, paras=None):
        super().__init__(llm, cfg, out)
        for k, v in dict(cache={}, samples={}, lock=asyncio.Lock(), n=0, t0=time.time()).items():
            if not hasattr(self, k): setattr(self, k, v)
        self.out, self.think, self.budget, self.paras = out, think, budget, dict(paras or {})

    @staticmethod
    def qset(q): return q.get("regime") if q["adopted"] else ("vetoed" if q["vetoed"] else "failed")

    def fslots(self, q, n, cond):
        cfg = self.cfg; o = cfg.oracle(q)
        if o is None: return None
        doc = o
        if cond.startswith("F-rnd"):
            pool = [c for c in cfg.eligible(q, cfg.A if o["adopted"] else cfg.N) if c["res_id"] != o["res_id"]]
            if not pool: return None
            doc = random.Random(f"{q['res_id']}|rnd").choice(pool)
        view, over = {"text": "text", "others": "others"}.get(cond.split("-")[-1], "coalition"), None
        if cond == "F-orc-flip":
            v = dict(cfg.vot[o["res_id"]]); v[n] = FLIP[v[n]]; over = {o["res_id"]: v}
        if cond == "F-rnd-orcv": over = {doc["res_id"]: dict(cfg.vot[o["res_id"]])}
        shown = (over or {}).get(doc["res_id"], cfg.vot[doc["res_id"]])[n] if view == "coalition" else None
        return doc, view, over, shown

    def rslots(self, q, cond):
        c = self.cfg
        if cond == "R-strat4-rev": return list(c.strat_slots(q, 4, True))[::-1]
        if cond == "R-strat4-shuf":
            s = list(c.strat_slots(q, 4, True)); random.Random(f"{q['res_id']}|shuf").shuffle(s); return s
        return c.strat_slots(q, int(cond[len("R-strat")]), True)

    async def build(self, q, n, cond):
        cfg, country = self.cfg, NAME[n]
        if cond == "B0-std": return PR.WITHOUT_HISTORY.format(country=country, answer_format=PR.ANSWER_ONE_WORD, query=ctx(q)), [], {}
        if cond.startswith("D-"):
            sl = self.slots(q, cond); view = {"text": "text", "target": "target", "coal": "coalition"}[cond.split("-")[-1]]
            return d_prompt(country, q, PR.precedent_block(sl, view, n, cfg.vot, NAME, WORD, P5)), [d["res_id"] for d in sl], {}
        if cond.startswith("F-"):
            r = self.fslots(q, n, cond)
            if r is None: return None
            doc, view, over, shown = r
            return d_prompt(country, q, block(cfg, [doc], n, view, over)), [doc["res_id"]], {"shown": shown}
        if cond.startswith("R-"):
            sl = self.rslots(q, cond)
            return d_prompt(country, q, block(cfg, sl, n, "coalition")), [d["res_id"] for d in sl], {}
        if cond.startswith("M-"):
            return MEM.format(country=country, date=q["date"].date().isoformat(), symbol=q["symbol"], fmt=PR.ANSWER_ONE_WORD), [], {}
        m = re.fullmatch(r"P(\d+)-(B0|strat4)", cond)
        if m:
            name, tpl, role = self.paras[int(m.group(1))]
            sl = cfg.strat_slots(q, 4, True) if m.group(2) == "strat4" else []
            return para_prompt(tpl, role, country, q, block(cfg, sl, n, "coalition")), [d["res_id"] for d in sl], {"template": name}
        raise ValueError(f"unknown condition {cond}")

    async def run_one(self, q, n, cond):
        if cond in E4.CONDS and not q["adopted"] and self.think == "none":
            return await super().run_one(q, n, cond)
        t = time.time(); err = p = ntok = None; ret = []; extra = {}
        try:
            b = await self.build(q, n, cond)
            if b is None: return
            user, ret, extra = b
            self.samples.setdefault(cond, f"[{q['symbol']} | {NAME[n]}]\n{user}")
            if self.think == "none": p, ntok = await self.llm.score(user)
            else:
                pre, info = await think_prefix(self.llm, user, self.think, self.budget); extra.update(info)
                p, ntok = await score_prefix(self.llm, pre)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        rec = dict(qid=q["res_id"], symbol=q["symbol"], qset=self.qset(q), nation=n, cond=cond, model=self.llm.m,
                   pY=p["Y"] if p else None, pN=p["N"] if p else None, pA=p["A"] if p else None,
                   pred=max(p, key=p.get) if p else None, true=self.cfg.vot[q["res_id"]][n], retrieved=ret,
                   prompt_tokens=ntok, sec=round(time.time() - t, 2), error=err, think=self.think, **extra)
        async with self.lock:
            with open(self.out, "a") as f: f.write(json.dumps(rec) + "\n")
            self.n += 1
            if self.n % 200 == 0: print(f"  {self.n} done | {time.time() - self.t0:.0f}s", flush=True)


def expand(spec, paras):
    G = dict(GROUPS, main=list(E4.CONDS), P=[f"P{i}-{k}" for i in paras for k in ("B0", "strat4")])
    G["wave2"] = G["F"] + G["R"] + G["P"] + G["M"]
    out = []
    for tok in spec.split(","):
        out += G.get(tok, [tok])
    return list(dict.fromkeys(out))


def selftest(cfg):
    """The local block builder must reproduce prompts.precedent_block exactly for the shared views."""
    for q in cfg.N[:5]:
        sl = cfg.strat_slots(q, 4, True)
        for view in ("text", "target", "coalition"):
            for n in P5:
                assert block(cfg, sl, n, view) == PR.precedent_block(sl, view, n, cfg.vot, NAME, WORD, P5), (q["res_id"], view, n)
    print("[selftest] evidence block identical to e4 for text/target/coalition views")


async def amain(a):
    if a.list_templates: choi_paraphrases(verbose=True); return
    cfg = ConfigA(); selftest(cfg)
    llm = LLM(a.served, E4.MODELS[a.served], base=a.base, conc=a.conc)
    await verify_served(llm, E4.MODELS[a.served])
    paras = choi_paraphrases()
    print("paraphrase templates:", {f"P{i}": v[0] for i, v in paras.items()} or "NONE FOUND", flush=True)
    conds = expand(a.conds, paras)
    if a.think != "none" and any(c.startswith("CF-") or c == "B0-choi" for c in conds): raise SystemExit("--think supports B0-std, D-*, F-*, R-*, P*, M-*")
    Q = sorted({"nonadopted": cfg.N, "adopted": cfg.A, "all": cfg.A + cfg.N}[a.queries], key=lambda r: r["date"])
    if a.limit: Q = Q[::max(1, len(Q) // a.limit)][:a.limit]
    out = STORE / f"e4_{a.served}_{a.tag}.jsonl"; done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r.get("error") is None: done.add((r["qid"], r["nation"], r["cond"]))
    has_orc = {q["res_id"]: cfg.oracle(q) is not None for q in Q} if any(c.startswith("F-") for c in conds) else {}
    def want(q, n, c):
        if (q["res_id"], n, c) in done: return False
        if c in E4.CONDS and not q["adopted"] and a.think == "none": return True
        if cfg.vot[q["res_id"]][n] not in ("Y", "N", "A"): return False
        return has_orc.get(q["res_id"], True) if c.startswith("F-") else True
    R = W2(llm, cfg, out, a.think, a.budget, paras); t = time.time()
    tasks = [R.run_one(q, n, c) for c in conds for q in Q for n in P5 if want(q, n, c)]
    print(f"{len(tasks)} tasks | {len(Q)} {a.queries} queries | {len(conds)} conditions | think={a.think} | model {a.served}", flush=True)
    await asyncio.gather(*tasks)
    print(f"wall {time.time() - t:.0f}s")
    (LOGS / f"e4_prompt_samples_{a.served}_{a.tag}.txt").write_text("\n\n".join(f"########## {k}\n{v}" for k, v in R.samples.items()))
    rows = [json.loads(l) for l in open(out)] if out.exists() else []
    bad = sum(1 for r in rows if r.get("error")); print(f"rows={len(rows)} error_rows={bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", default=""); ap.add_argument("--conds", default="wave2"); ap.add_argument("--tag", default="wave2")
    ap.add_argument("--queries", default="nonadopted", choices=["nonadopted", "adopted", "all"]); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conc", type=int, default=10); ap.add_argument("--base", default="http://127.0.0.1:8007/v1")
    ap.add_argument("--think", default="none", choices=["none", "off", "on"]); ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--list-templates", action="store_true")
    asyncio.run(amain(ap.parse_args()))
