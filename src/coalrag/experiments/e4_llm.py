import argparse, asyncio, json, os, random, time
from pathlib import Path
import numpy as np, pandas as pd
from coalrag.core.match import P5
from coalrag.retrieval.configA import ConfigA, NAME
from coalrag.llm.client import LLM, verify_served
from coalrag.llm import prompts as PR

PROJ = Path(os.environ["PROJ"]); STORE = PROJ / "results" / "store"; LOGS = PROJ / "logs"
MODELS = {"qwen2.5-32b-gptq-int8": os.path.expanduser("~/models/Qwen2.5-32B-Instruct-GPTQ-Int8"),
          "mistral-small-2501-fp8": os.path.expanduser("~/models/Mistral-Small-24B-Instruct-2501-FP8-dynamic"),
          "qwen2.5-7b": "$HF_MODELS/Qwen/Qwen2.5-7B-Instruct",
          "qwen2.5-32b": "$HF_MODELS/Qwen/Qwen2.5-32B-Instruct",
          "mistral-small-3.1": "$HF_MODELS/MistralAI/Mistral-Small-3.1-24B-Instruct-2503",
          "llama-3.3-70b-fp8": "$HF_MODELS/MetaAI/Llama-3.3-70B-Instruct-FP8"}
WORD = {"Y": "favour", "N": "against", "A": "abstention", None: "not recorded"}
CONDS = ["B0-choi", "B0-std", "CF-LTorig", "CF-LT", "CF-bge", "CF-CSR3",
         "D-bge-text", "D-bge-target", "D-bge-coal", "D-oracle-coal", "D-random-coal",
         "D-top4-coal", "D-strat4-coal", "D-strat4-target"]
CF_METHOD = {"CF-LTorig": "LT-orig", "CF-LT": "LT", "CF-bge": "bge-small", "CF-CSR3": "CSR3"}
CTX_MAX = 12000
ctx = lambda r: str(r.get("context") or r.get("summary") or "")[:CTX_MAX]

def history(cfg, docs, preds, refls, n, final):
    """Choi RAG_RFLX_GRAPH history string (adopted precedents expose no votes)."""
    s = ""
    for i, d in enumerate(docs):
        s += f"##Previous resolution Predicted {i}" + f" - Resolution context :\n{d.get('summary') or ''}" + f" - Your Prediction : \n{preds[i]}"
        if not d["adopted"]: s += f" - True vote : \n{WORD[cfg.vot[d['res_id']][n]]}"
        else: s += " - Result of the vote : Adopted." if final else " - The result of vote : Adopted."
        s += f" - Reflection : \n{refls[i]}\n" + "\n\n"
    return s

class Runner:
    def __init__(self, llm, cfg, out):
        self.llm, self.cfg, self.out = llm, cfg, out
        self.cache, self.samples, self.lock, self.n, self.t0 = {}, {}, asyncio.Lock(), 0, time.time()

    async def choi_flow(self, q, n, method):
        cfg, country = self.cfg, NAME[n]; docs = cfg.choi_docs(q, method); preds, refls = [], []
        for i, d in enumerate(docs):
            key = (n, tuple(x["res_id"] for x in docs[:i + 1]))
            if key not in self.cache:
                if i == 0: u = PR.WITHOUT_HISTORY.format(country=country, answer_format=PR.ANSWER_JSON, query=ctx(d))
                else: u = PR.WITH_HISTORY.format(country=country, answer_format=PR.ANSWER_JSON, query=ctx(d),
                                                 previous_prediction_histories=history(cfg, docs[:i], preds, refls, n, False))
                txt, _ = await self.llm.generate(u, 400); v, rat = PR.parse_vote_json(txt)
                if v is None:
                    p, _ = await self.llm.score(u.replace(PR.ANSWER_JSON, PR.ANSWER_ONE_WORD))
                    v = {"Y": "favour", "N": "against", "A": "abstention"}[max(p, key=p.get)]
                gt = "favour or abstention" if d["adopted"] else WORD[cfg.vot[d["res_id"]][n]]
                res = ("possibly correct" if v in gt else "wrong") if d["adopted"] else ("correct" if v == gt else "wrong")
                sp = PR.format_speech(cfg.speech.get(d["res_id"], {}), country)
                has = ("No comments" not in sp) and ("Cannot find the meeting script" not in sp)
                ru = PR.REFLEXION.format(country=country, historical_res_vote_predict=v, rationale=rat, ground_truth=gt,
                                         prediction_result=res, if_speech_exists=PR.IF_SPEECH if has else "",
                                         to_be_executed_resolution_context=d.get("summary") or "",
                                         speech_record_if_speech_exists=("<your_predecessor's_speech>\n" + sp) if has else "")
                rf, _ = await self.llm.generate(ru, 400)
                self.cache[key] = ({"vote": v, "rationale": rat}, rf)
            pr, rf = self.cache[key]; preds.append(pr); refls.append(rf)
        if docs:
            user = PR.WITH_HISTORY.format(country=country, answer_format=PR.ANSWER_ONE_WORD, query=ctx(q),
                                          previous_prediction_histories=history(cfg, docs, preds, refls, n, True))
        else: user = PR.WITHOUT_HISTORY.format(country=country, answer_format=PR.ANSWER_ONE_WORD, query=ctx(q))
        return user, [d["res_id"] for d in docs]

    def slots(self, q, cond):
        cfg = self.cfg
        if cond.startswith("D-strat4"): return cfg.strat_slots(q, 4, True)
        if cond.startswith("D-top4"): return cfg.strat_slots(q, 4, False)
        a = cfg.rank(q, "A", "bge-small"); b = cfg.rank(q, "N", "bge-small")
        sA = [a[0][0]] if a else []; sN = [b[0][0]] if b else []
        if cond == "D-oracle-coal":
            o = cfg.oracle(q)
            if o is not None: sN = [o]
        if cond == "D-random-coal":
            el = cfg.eligible(q, cfg.N)
            if el: sN = [random.Random(q["res_id"]).choice(el)]
        out = {d["res_id"]: d for d in sA + sN}
        return sorted(out.values(), key=lambda d: d["date"])

    async def run_one(self, q, n, cond):
        t = time.time(); err = p = ntok = None; ret = []
        try:
            country = NAME[n]
            if cond == "B0-choi": user = PR.BASIC.format(country=country, resolution=ctx(q), format=PR.ANSWER_ONE_WORD)
            elif cond == "B0-std": user = PR.WITHOUT_HISTORY.format(country=country, answer_format=PR.ANSWER_ONE_WORD, query=ctx(q))
            elif cond.startswith("CF-"): user, ret = await self.choi_flow(q, n, CF_METHOD[cond])
            else:
                sl = self.slots(q, cond); ret = [d["res_id"] for d in sl]
                view = {"text": "text", "target": "target", "coal": "coalition"}[cond.split("-")[-1]]
                user = PR.WITHOUT_HISTORY.format(country=country, answer_format=PR.ANSWER_ONE_WORD, query=ctx(q))
                user = user.replace("<draft_resolution> :", PR.precedent_block(sl, view, n, self.cfg.vot, NAME, WORD, P5) + "\n<draft_resolution> :", 1)
            self.samples.setdefault(cond, f"[{q['symbol']} | {country}]\n{user}")
            p, ntok = await self.llm.score(user)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        rec = dict(qid=q["res_id"], symbol=q["symbol"], qset="vetoed" if q["vetoed"] else "failed", nation=n, cond=cond, model=self.llm.m,
                   pY=p["Y"] if p else None, pN=p["N"] if p else None, pA=p["A"] if p else None,
                   pred=max(p, key=p.get) if p else None, true=self.cfg.vot[q["res_id"]][n], retrieved=ret,
                   prompt_tokens=ntok, sec=round(time.time() - t, 2), error=err)
        async with self.lock:
            with open(self.out, "a") as f: f.write(json.dumps(rec) + "\n")
            self.n += 1
            if self.n % 100 == 0: print(f"  {self.n} done | {time.time() - self.t0:.0f}s", flush=True)

def summarize(out, conds):
    df = pd.read_json(out, lines=True); err = df.error.notna().sum()
    df = df[df.error.isna()].drop_duplicates(["qid", "nation", "cond"], keep="last"); rows = []
    for c, g in df.groupby("cond"):
        y, yp = g.true.values, g.pred.values; fs = []
        for l in ["Y", "N", "A"]:
            tp = ((y == l) & (yp == l)).sum(); fp = ((y != l) & (yp == l)).sum(); fn = ((y == l) & (yp != l)).sum()
            pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0; fs.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
        rows.append(dict(cond=c, n=len(g), acc=(y == yp).mean(), macroF1=np.mean(fs), yes_rate=(yp == "Y").mean(), true_yes=(y == "Y").mean(),
                         no_recall=((y == "N") & (yp == "N")).sum() / max((y == "N").sum(), 1), mean_pY=g.pY.mean(),
                         tok=g.prompt_tokens.mean(), sec=g.sec.mean()))
    df_sum = pd.DataFrame(rows)
    if "cond" in df_sum.columns:
        df_sum = df_sum.set_index("cond")
    print(df_sum.round(3).to_string())
    print(f"error rows: {err}"); 
    if err: print(pd.read_json(out, lines=True).error.dropna().value_counts().head(5).to_string())

async def amain(a):
    cfg = ConfigA(); llm = LLM(a.served, MODELS[a.served], base=a.base, conc=a.conc)
    await verify_served(llm, MODELS[a.served])
    Q = sorted(cfg.N, key=lambda r: r["date"])
    if a.limit: Q = Q[::max(1, len(Q) // a.limit)][:a.limit]
    out = STORE / f"e4_{a.served}_{a.tag}.jsonl"; done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r.get("error") is None: done.add((r["qid"], r["nation"], r["cond"]))
    conds = CONDS if a.conds == "all" else a.conds.split(",")
    R = Runner(llm, cfg, out); t = time.time()
    tasks = [R.run_one(q, n, c) for c in conds for q in Q for n in P5 if (q["res_id"], n, c) not in done]
    print(f"{len(tasks)} tasks | {len(Q)} queries x 5 nations x {len(conds)} conditions | model {a.served}", flush=True)
    await asyncio.gather(*tasks)
    print(f"wall {time.time() - t:.0f}s")
    (LOGS / f"e4_prompt_samples_{a.served}_{a.tag}.txt").write_text("\n\n".join(f"########## {k}\n{v}" for k, v in R.samples.items()))
    summarize(out, conds)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--served", default="qwen2.5-32b"); ap.add_argument("--conds", default="all")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--tag", default="main"); ap.add_argument("--conc", type=int, default=32); ap.add_argument("--base", default="http://127.0.0.1:8007/v1")
    asyncio.run(amain(ap.parse_args()))
