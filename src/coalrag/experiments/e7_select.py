"""E7 selection experiment (pre-registered H13-H14, D81). D-strat4-coal shows four precedents with four different vote
configurations in one prompt, and the model gains nothing (H5 not supported), although one matching precedent alone helps
a lot (D-oracle-coal). E7 separates the two steps:
  slot scoring  each of the 4 stratified precedents is shown ALONE (coalition view; same construction as F-orc-coal)
                -> rows S-slot1..S-slot4 in results/store/e4_<served>_select.jsonl
  selection     once per draft, the model is asked which earlier draft is the closest precedent (texts only, no votes),
                scored by option likelihood over "1".."k" -> results/store/e7_<served>_selector.jsonl
Aggregations are fixed in analysis/e7_analysis.py before any output exists. Nothing is tuned."""
import argparse, asyncio, json, math, time
from coalrag.core.match import P5
from coalrag.retrieval.configA import ConfigA, NAME
from coalrag.llm.client import LLM, verify_served
from coalrag.experiments import e5_wave2 as W

E4, STORE = W.E4, W.STORE
SEL = ("You will read a draft resolution submitted to the United Nations Security Council (UNSC) and {k} earlier draft resolutions.\n"
       "Which earlier draft is the closest precedent for the current draft, that is, the one concerning the same situation "
       "and proposing the same kind of action?\n\n<earlier_drafts>\n{blocks}</earlier_drafts>\n\n<current_draft>\n{query}\n</current_draft>\n\n"
       "Answer with the number of the closest precedent only ({opts}).")


def sel_block(slots):
    L = []
    for i, d in enumerate(slots, 1):
        L.append(f"[{i}] Date: {d['date'].date()} | Outcome: {'adopted' if d['adopted'] else 'not adopted'}")
        L.append(f"Summary: {d.get('summary') or ''}")
    return "\n".join(L) + "\n"


async def score_opts(llm, user, opts):
    prefix = llm.prefix(user); pre = llm.tok(prefix, add_special_tokens=False).input_ids
    lps = await asyncio.gather(*[llm._lp(prefix, pre, o) for o in opts])
    z = max(lps); w = [math.exp(v - z) for v in lps]; s = sum(w)
    return [x / s for x in w], len(pre)


async def amain(a):
    cfg = ConfigA(); llm = LLM(a.served, E4.MODELS[a.served], base=a.base, conc=a.conc)
    await verify_served(llm, E4.MODELS[a.served])
    Q = sorted(cfg.N, key=lambda r: r["date"])[: a.limit or None]
    out, sout = STORE / f"e4_{a.served}_select.jsonl", STORE / f"e7_{a.served}_selector.jsonl"
    done = {(r["qid"], r["nation"], r["cond"]) for r in map(json.loads, open(out)) if not r.get("error")} if out.exists() else set()
    sdone = {r["qid"] for r in map(json.loads, open(sout)) if not r.get("error")} if sout.exists() else set()
    lock = asyncio.Lock(); cnt = [0]; t0 = time.time(); sample = {}

    async def write(path, rec):
        async with lock:
            with open(path, "a") as f: f.write(json.dumps(rec) + "\n")
            cnt[0] += 1
            if cnt[0] % 200 == 0: print(f"  {cnt[0]} done | {time.time() - t0:.0f}s", flush=True)

    async def slot_task(q, n, i, doc):
        t = time.time(); err = p = ntok = None
        try:
            user = W.d_prompt(NAME[n], q, W.block(cfg, [doc], n, "coalition")); sample.setdefault("slot", user)
            p, ntok = await llm.score(user)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        await write(out, dict(qid=q["res_id"], symbol=q["symbol"], qset="vetoed" if q["vetoed"] else "failed", nation=n, cond=f"S-slot{i}",
                              slot=i, model=llm.m, pY=p["Y"] if p else None, pN=p["N"] if p else None, pA=p["A"] if p else None,
                              pred=max(p, key=p.get) if p else None, true=cfg.vot[q["res_id"]][n], retrieved=[doc["res_id"]],
                              prompt_tokens=ntok, sec=round(time.time() - t, 2), error=err))

    async def sel_task(q, slots):
        err = w = ntok = None; opts = [str(i) for i in range(1, len(slots) + 1)]
        try:
            user = SEL.format(k=len(slots), blocks=sel_block(slots), query=E4.ctx(q), opts=", ".join(opts)); sample.setdefault("selector", user)
            w, ntok = await score_opts(llm, user, opts)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        await write(sout, dict(qid=q["res_id"], ids=[d["res_id"] for d in slots], w=w, model=llm.m, prompt_tokens=ntok, error=err))

    tasks = []
    for q in Q:
        slots = list(cfg.strat_slots(q, 4, True))
        if not slots: continue
        if q["res_id"] not in sdone: tasks.append(sel_task(q, slots))
        for n in P5:
            if cfg.vot[q["res_id"]][n] not in ("Y", "N", "A"): continue
            tasks += [slot_task(q, n, i, d) for i, d in enumerate(slots, 1) if (q["res_id"], n, f"S-slot{i}") not in done]
    print(f"{len(tasks)} tasks | {len(Q)} drafts | model {a.served}", flush=True)
    await asyncio.gather(*tasks)
    if sample: (W.LOGS / f"e7_prompt_samples_{a.served}.txt").write_text("\n\n".join(f"########## {k}\n{v}" for k, v in sample.items()))
    bad = sum(1 for p in (out, sout) if p.exists() for r in map(json.loads, open(p)) if r.get("error"))
    print(f"wall {time.time() - t0:.0f}s | error rows {bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--served", required=True); ap.add_argument("--base", default="http://127.0.0.1:8006/v1")
    ap.add_argument("--conc", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    asyncio.run(amain(ap.parse_args()))
