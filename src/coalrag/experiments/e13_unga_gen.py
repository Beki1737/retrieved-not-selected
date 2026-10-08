"""E13 (pre-registered H20, wave 8): generation-stage replication on the UN General Assembly (auxiliary to the Security Council).
Queries: Assembly roll calls 2013-2019 (E9 data, unga_p5.parquet) with all five P5 votes, at least 50 earlier roll calls, a matching
precedent (identical P5 configuration) among earlier roll calls after removing those with the same normalized short title (E9 rule),
and at least three stratified precedents of other configurations within the bge-small top 50. Uniform sample of 150 (seed 0).
Candidate set (E10 construction): most similar matching precedent + the three best-ranked stratified precedents of other configurations.
Conditions (one run per model): G-B0 none | G-orc match alone | G-d1..G-d3 each distractor alone | G4-p1..G4-p4 four together, match at
position p | selector: the model's choice of the closest precedent from the four texts (no votes), per ordering.
Judge (--judge): E12 rubric adapted to the Assembly, votes hidden, on every (query, candidate) pair (selection appropriateness).
Usage: --design | --served NAME --base URL | --judge --served NAME --base URL"""
import argparse, asyncio, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from coalrag.retrieval.configA import NAME
from coalrag.llm.client import LLM, verify_served
from coalrag.experiments import e5_wave2 as W
from coalrag.experiments.e7_select import score_opts
from coalrag.experiments.e12_appropriateness import Q as Q12

E4, STORE, LOGS = W.E4, W.STORE, W.LOGS
PROC = Path(os.environ["PROJ"]) / "data" / "processed"; DESIGN = STORE / "e13_design.json"
P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]; WORD = {"Y": "favour", "N": "against", "A": "abstention"}
EMB = "$HF_MODELS/BAAI/bge-small-en-v1.5"; NQ, DEPTH = 150, 50

GA = """
Your answer should be very concise.
You are a representative of {country} in the United Nations General Assembly (UNGA). You must decide how to vote on the following <draft_resolution> in the UNGA.
Your decision should be based on  {country}'s diplomatic stance and also its national interest.

**Important guidelines:**
- Most importantly, do **not** incorporate biases about {country}.
- The vote must be chosen from three options only: **favour, against, or abstention**.
- Your reasoning should be logical and derived strictly from the given information.

Your answer format SHOULD STRICTLY follow this structure, WITHOUT ANY OTHER RESPONSE:
{answer_format}

<draft_resolution> :
{query}
</draft_resolution>
"""
SEL = ("You will read a draft resolution submitted to the United Nations General Assembly (UNGA) and {k} earlier Assembly resolutions.\n"
       "Which earlier resolution is the closest precedent for the current draft, that is, the one concerning the same issue "
       "and proposing the same kind of action?\n\n<earlier_resolutions>\n{blocks}</earlier_resolutions>\n\n<current_draft>\n{query}\n</current_draft>\n\n"
       "Answer with the number of the closest precedent only ({opts}).")
JUDGE = ("You will compare a current draft resolution of the United Nations General Assembly with an earlier Assembly resolution. "
         "Judge only their content. Voting results are not shown and must not be guessed.\n\n"
         "<current_draft>\n{draft}\n</current_draft>\n\n<earlier_document>\n{cand}\n</earlier_document>\n\n{question}")
QGA = dict(Q12)
QGA["A"] = ("Does the earlier resolution take the same kind of action as the current draft (for example a condemnation or demand, a call "
            "for negotiations or self-determination, a request to the Secretary-General, the establishment or renewal of a body or "
            "mandate, a disarmament or non-proliferation measure, or a procedural decision)?\n1 = a different kind of action\n"
            "2 = a similar kind of action\n3 = the same kind of action\nAnswer with the number only (1, 2 or 3).", ["1", "2", "3"])


def qtext(d): return f"Date: {d['date']}\n{d['text']}"


def block(docs, view):
    L = ["Consider the following similar past resolutions in <similar_past_resolutions>.", "<similar_past_resolutions>"]
    for i, c in enumerate(docs, 1):
        L.append(f"[{i}] Date: {c['date']}"); L.append(f"Summary: {c['text']}")
        if view == "coalition": L.append("Votes of the permanent members: " + "; ".join(f"{NAME[m]}: {WORD[c['votes'][m]]}" for m in P5))
    L.append("</similar_past_resolutions>"); return "\n".join(L) + "\n"


def prompt(country, q, docs):
    u = GA.format(country=country, answer_format=W.PR.ANSWER_ONE_WORD, query=qtext(q))
    return u.replace("<draft_resolution> :", block(docs, "coalition") + "\n<draft_resolution> :", 1) if docs else u


def make_design():
    d = pd.read_parquet(PROC / "unga_p5.parquet").sort_values(["date", "rcid"]).reset_index(drop=True)
    cf, dates = d.config.values, d.date.values
    title = d["short"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.split().str.join(" ").values
    ef, idf = PROC / "emb_unga_bge-small.npy", PROC / "emb_unga_index.txt"; ids = [str(x) for x in d.rcid]
    if ef.exists() and idf.exists() and idf.read_text().split() == ids: E = np.load(ef)
    else:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(EMB); m.max_seq_length = 256
        E = m.encode(list(d.text.values), batch_size=64, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        np.save(ef, E); idf.write_text("\n".join(ids))
    qs = [i for i in np.where(d.date >= "2013-01-01")[0] if (dates < dates[i]).sum() >= 50]; C = []; why = {"no match": 0, "<3 distractors": 0}
    for i in qs:
        pool = np.where((dates < dates[i]) & (title != title[i]))[0]
        o = pool[np.argsort(-(E[pool] @ E[i]), kind="stable")]; rank = {int(j): r for r, j in enumerate(o)}
        mm = o[cf[o] == cf[i]]
        if not len(mm): why["no match"] += 1; continue
        seen, sl = set(), []
        for j in o[:DEPTH]:
            if cf[j] not in seen: seen.add(cf[j]); sl.append(int(j))
        dis = [j for j in sl if cf[j] != cf[i]][:3]
        if len(dis) < 3: why["<3 distractors"] += 1; continue
        C.append((i, int(mm[0]), dis, rank))
    rng = np.random.default_rng(0); pick = sorted(rng.choice(len(C), min(NQ, len(C)), replace=False)) if len(C) > NQ else range(len(C))
    rec = lambda j, rank=None: dict(rid=str(d.rcid[j]), unres=str(d.unres[j]), date=str(pd.Timestamp(d.date[j]).date()), text=str(d.text[j]),
                                     votes={p: d[p][j] for p in P5}, **({} if rank is None else {"rank": rank}))
    Q = []
    for k in pick:
        i, m, dis, rank = C[k]
        Q.append(dict(**rec(i), cands=[rec(m, rank[m])] + [rec(j, rank[j]) for j in dis], match_rank=rank[m]))
    D = dict(n_eligible=len(C), excluded=why, queries=Q); json.dump(D, open(DESIGN, "w"))
    allv = np.array([q["votes"][p] for q in Q for p in P5])
    print(f"[design] queries from 2013 with >= 50 earlier: {len(qs)} | eligible {len(C)} | excluded {why} | sampled {len(Q)}")
    print(f"[design] rows {len(allv)} | vote shares Y {np.mean(allv == 'Y'):.3f} N {np.mean(allv == 'N'):.3f} A {np.mean(allv == 'A'):.3f}")
    print(f"[design] dense rank of the match (median, IQR): {np.median([q['match_rank'] for q in Q]):.0f} "
          f"({np.percentile([q['match_rank'] for q in Q], 25):.0f} to {np.percentile([q['match_rank'] for q in Q], 75):.0f})")
    print("[design] example query:", qtext(Q[0])[:300].replace("\n", " | "))


async def run(a):
    D = json.load(open(DESIGN)); llm = LLM(a.served, E4.MODELS[a.served], base=a.base, conc=a.conc)
    await verify_served(llm, E4.MODELS[a.served])
    tag = "judge" if a.judge else "gen"; out = STORE / f"e13_{a.served}_{tag}.jsonl"
    done = {r["key"] for r in map(json.loads, open(out)) if r.get("error") is None} if out.exists() else set()
    lock = asyncio.Lock(); cnt = [0]; t0 = time.time(); sample = {}

    async def write(rec):
        async with lock:
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
            cnt[0] += 1
            if cnt[0] % 500 == 0: print(f"  {cnt[0]} done | {time.time() - t0:.0f}s", flush=True)

    async def row(q, n, cond, docs, extra):
        err = p = ntok = None
        try:
            user = prompt(NAME[n], q, docs); sample.setdefault(cond, user); p, ntok = await llm.score(user)
        except Exception as e: err = f"{type(e).__name__}: {str(e)[:200]}"
        await write(dict(key=f"{q['rid']}|{n}|{cond}", qid=q["rid"], nation=n, cond=cond, model=llm.m, pY=p["Y"] if p else None,
                         pN=p["N"] if p else None, pA=p["A"] if p else None, true=q["votes"][n], retrieved=[c["rid"] for c in docs],
                         prompt_tokens=ntok, error=err, **extra))

    async def sel(q, pos, docs):
        err = w = ntok = None; opts = [str(i) for i in range(1, 5)]
        try:
            blocks = "\n".join(f"[{i}] Date: {c['date']}\nSummary: {c['text']}" for i, c in enumerate(docs, 1)) + "\n"
            user = SEL.format(k=4, blocks=blocks, query=qtext(q), opts=", ".join(opts)); sample.setdefault("selector", user)
            w, ntok = await score_opts(llm, user, opts)
        except Exception as e: err = f"{type(e).__name__}: {str(e)[:200]}"
        await write(dict(key=f"{q['rid']}|sel|{pos}", qid=q["rid"], pos=pos, ids=[c["rid"] for c in docs], w=w, model=llm.m, prompt_tokens=ntok, error=err))

    async def judge(q, c, qq):
        err = p = ntok = None
        try:
            user = JUDGE.format(draft=qtext(q), cand=f"Date: {c['date']}\nSummary: {c['text']}", question=QGA[qq][0]); sample.setdefault(f"judge-{qq}", user)
            p, ntok = await score_opts(llm, user, QGA[qq][1])
        except Exception as e: err = f"{type(e).__name__}: {str(e)[:200]}"
        await write(dict(key=f"{q['rid']}|{c['rid']}|{qq}", qid=q["rid"], cid=c["rid"], q=qq, p=p, judge=llm.m, prompt_tokens=ntok, error=err))

    tasks = []
    for q in D["queries"]:
        m, dis = q["cands"][0], q["cands"][1:]
        if a.judge:
            tasks += [judge(q, c, qq) for c in q["cands"] for qq in QGA if f"{q['rid']}|{c['rid']}|{qq}" not in done]; continue
        orders = {p: dis[:p - 1] + [m] + dis[p - 1:] for p in (1, 2, 3, 4)}
        tasks += [sel(q, p, docs) for p, docs in orders.items() if f"{q['rid']}|sel|{p}" not in done]
        conds = [("G-B0", [], {}), ("G-orc", [m], {"rank": m["rank"]})] + [(f"G-d{i}", [c], {"rank": c["rank"]}) for i, c in enumerate(dis, 1)]
        conds += [(f"G4-p{p}", docs, {"match_pos": p}) for p, docs in orders.items()]
        tasks += [row(q, n, c, docs, ex) for n in P5 for c, docs, ex in conds if f"{q['rid']}|{n}|{c}" not in done]
    print(f"{len(D['queries'])} queries | {len(tasks)} tasks | {tag} | model {a.served}", flush=True)
    await asyncio.gather(*tasks)
    if sample: (LOGS / f"e13_prompt_samples_{a.served}_{tag}.txt").write_text("\n\n".join(f"########## {k}\n{v}" for k, v in sample.items()))
    rows = [json.loads(l) for l in open(out)] if out.exists() else []
    print(f"wall {time.time() - t0:.0f}s | rows {len(rows)} | error rows {sum(1 for r in rows if r.get('error'))} | "
          f"max tokens {max([r.get('prompt_tokens') or 0 for r in rows] or [0])}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--design", action="store_true"); ap.add_argument("--judge", action="store_true")
    ap.add_argument("--served", default=""); ap.add_argument("--base", default=""); ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args(); make_design() if a.design else asyncio.run(run(a))
