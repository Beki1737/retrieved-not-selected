"""E10 guaranteed-match selection experiment (pre-registered H17, D99). Answers design review item directly:
the matching precedent is ALWAYS in the evidence set, so selection is separated from retrieval coverage without any decomposition.
Drafts: the non-adopted drafts with an exact-configuration precedent (oracle) and at least three earlier stratified precedents of
other configurations. Set = oracle precedent + the three best-ranked stratified precedents whose configuration differs from the
draft's (distractors), so the four show four different configurations and exactly one matches.
Conditions (same prompt builder and coalition view as E4/E5/E7; all scored in this one run):
  G-B0             no evidence
  G-orc            the matching precedent alone (= F-orc-coal construction)
  G-d1..G-d3       each distractor alone
  G4-p1..G4-p4     all four shown together, matching precedent at position p (distractors keep their rank order)
  selector         the model's choice of the closest precedent from texts only (E7 prompt), for each of the four orderings
Output: results/store/e4_<served>_guar.jsonl and e10_<served>_selector.jsonl. Nothing is tuned."""
import argparse, asyncio, json, time
from coalrag.core.match import P5
from coalrag.retrieval.configA import ConfigA, NAME
from coalrag.llm.client import LLM, verify_served
from coalrag.experiments import e5_wave2 as W
from coalrag.experiments.e7_select import SEL, sel_block, score_opts

E4, STORE = W.E4, W.STORE


def conf(cfg, rid): return tuple(cfg.vot[rid][x] for x in P5)


def design(cfg, q):
    o = cfg.oracle(q)
    if o is None: return None
    cq = conf(cfg, q["res_id"]); ranked = list(cfg.strat_slots(q, 100000, False)); rank = {d["res_id"]: i for i, d in enumerate(ranked)}
    dis = [d for d in cfg.strat_slots(q, 12, True) if conf(cfg, d["res_id"]) != cq and d["res_id"] != o["res_id"]][:3]
    if len(dis) < 3: return None
    return o, dis, rank


async def amain(a):
    cfg = ConfigA(); llm = LLM(a.served, E4.MODELS[a.served], base=a.base, conc=a.conc)
    await verify_served(llm, E4.MODELS[a.served])
    out, sout = STORE / f"e4_{a.served}_guar.jsonl", STORE / f"e10_{a.served}_selector.jsonl"
    done = {(r["qid"], r["nation"], r["cond"]) for r in map(json.loads, open(out)) if not r.get("error")} if out.exists() else set()
    sdone = {(r["qid"], r["pos"]) for r in map(json.loads, open(sout)) if not r.get("error")} if sout.exists() else set()
    lock = asyncio.Lock(); cnt = [0]; t0 = time.time(); sample = {}

    async def write(path, rec):
        async with lock:
            with open(path, "a") as f: f.write(json.dumps(rec) + "\n")
            cnt[0] += 1
            if cnt[0] % 200 == 0: print(f"  {cnt[0]} done | {time.time() - t0:.0f}s", flush=True)

    async def row(q, n, cond, docs, extra):
        t = time.time(); err = p = ntok = None
        try:
            user = (W.PR.WITHOUT_HISTORY.format(country=NAME[n], answer_format=W.PR.ANSWER_ONE_WORD, query=W.ctx(q)) if not docs
                    else W.d_prompt(NAME[n], q, W.block(cfg, docs, n, "coalition")))
            sample.setdefault(cond, user); p, ntok = await llm.score(user)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        await write(out, dict(qid=q["res_id"], symbol=q["symbol"], qset="vetoed" if q["vetoed"] else "failed", nation=n, cond=cond, model=llm.m,
                              pY=p["Y"] if p else None, pN=p["N"] if p else None, pA=p["A"] if p else None, pred=max(p, key=p.get) if p else None,
                              true=cfg.vot[q["res_id"]][n], retrieved=[d["res_id"] for d in docs], prompt_tokens=ntok, sec=round(time.time() - t, 2),
                              error=err, **extra))

    async def sel(q, pos, docs, mpos):
        err = w = ntok = None; opts = [str(i) for i in range(1, len(docs) + 1)]
        try:
            user = SEL.format(k=len(docs), blocks=sel_block(docs), query=E4.ctx(q), opts=", ".join(opts)); w, ntok = await score_opts(llm, user, opts)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        await write(sout, dict(qid=q["res_id"], pos=pos, match_pos=mpos, ids=[d["res_id"] for d in docs], w=w, model=llm.m, prompt_tokens=ntok, error=err))

    tasks, nd = [], 0
    for q in sorted(cfg.N, key=lambda r: r["date"]):
        dz = design(cfg, q)
        if dz is None: continue
        o, dis, rank = dz; nd += 1
        orders = {p: dis[:p - 1] + [o] + dis[p - 1:] for p in (1, 2, 3, 4)}
        for p, docs in orders.items():
            if (q["res_id"], p) not in sdone: tasks.append(sel(q, p, docs, p))
        for n in P5:
            if cfg.vot[q["res_id"]][n] not in ("Y", "N", "A"): continue
            conds = [("G-B0", [], {}), ("G-orc", [o], {"rank": rank.get(o["res_id"])})]
            conds += [(f"G-d{i}", [d], {"rank": rank.get(d["res_id"])}) for i, d in enumerate(dis, 1)]
            conds += [(f"G4-p{p}", docs, {"match_pos": p}) for p, docs in orders.items()]
            tasks += [row(q, n, c, docs, ex) for c, docs, ex in conds if (q["res_id"], n, c) not in done]
    print(f"{nd} drafts with a guaranteed match | {len(tasks)} tasks | model {a.served}", flush=True)
    await asyncio.gather(*tasks)
    if sample: (W.LOGS / f"e10_prompt_samples_{a.served}.txt").write_text("\n\n".join(f"########## {k}\n{v}" for k, v in sample.items()))
    bad = sum(1 for p in (out, sout) if p.exists() for r in map(json.loads, open(p)) if r.get("error"))
    print(f"wall {time.time() - t0:.0f}s | error rows {bad}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--served", required=True); ap.add_argument("--base", default="http://127.0.0.1:8007/v1")
    ap.add_argument("--conc", type=int, default=16); asyncio.run(amain(ap.parse_args()))
