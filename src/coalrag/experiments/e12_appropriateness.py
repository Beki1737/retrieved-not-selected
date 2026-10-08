"""E12 (pre-registered H19, wave 7): content appropriateness of precedents, judged with votes and outcomes masked.
design review item: identical five-member votes are not the same as a precedent being content-appropriate for the agenda
item. Every (draft, candidate) pair that the selection experiments showed to the LLMs is judged on content only:
  pairs     E10 candidate sets (matching precedent + three distractors, 50 drafts) and E7 stratified slots (66 drafts)
  questions T same situation/agenda item (1-3), C same countries (1-3), A same kind of Council action (1-3),
            O appropriate precedent judging only by content (1 no, 2 yes); option-likelihood scoring (D73)
  info      P = exactly the text the predicting LLM sees (draft text; candidate date and summary), votes and outcome removed
            R = P plus UNSC-CKG metadata (agenda, subjects, countries concerned, action items, keywords)
  masking   sentences with vote or outcome wording are removed from every text field (counts are logged)
Usage: --pairs (CPU; writes the pair table and the blind human-annotation sheet)
       --served NAME --base URL [--shard i/n] (judge run; output results/store/e12_<served>_s<i>.jsonl; resumable)"""
import argparse, asyncio, json, random, re, time
import numpy as np, pandas as pd
from coalrag.core.match import P5
from coalrag.retrieval.configA import ConfigA
from coalrag.llm.client import LLM, verify_served
from coalrag.experiments import e5_wave2 as W
from coalrag.experiments.e7_select import score_opts
from coalrag.experiments.e10_guaranteed import design

E4, STORE, LOGS = W.E4, W.STORE, W.LOGS
PAIRS = STORE / "e12_pairs.parquet"
VOTEWORDS = re.compile(r"\b(veto(?:ed|es|ing)?|votes?|voted|voting|abstain(?:ed|ing|s)?|abstentions?|unanimous(?:ly)?|"
                       r"was adopted|were adopted|not adopted|failed to (?:be )?adopt\w*|rejected|negative vote)\b", re.I)
SENT = re.compile(r"[^.;\n]*[.;\n]?")
NMASK = {"n": 0}


def mask(t):
    t = "" if t is None or (isinstance(t, float) and np.isnan(t)) else str(t)
    out = []
    for s in SENT.findall(t):
        if s and VOTEWORDS.search(s): NMASK["n"] += 1; out.append(" [removed]")
        else: out.append(s)
    return "".join(out).strip()


def fmt(v, cap=400):
    if v is None: return ""
    if isinstance(v, (list, tuple, np.ndarray)): v = ", ".join(map(str, list(v)))
    if isinstance(v, float) and np.isnan(v): return ""
    return mask(str(v))[:cap]


META = [("Agenda item", "agenda"), ("Subjects", "subjects"), ("Countries concerned", "target_nations"),
        ("Action items", "action_item"), ("Keywords", "keyword")]
JUDGE = ("You will compare a current draft resolution of the United Nations Security Council with an earlier Council document. "
         "Judge only their content. Voting results and outcomes are not shown and must not be guessed.\n\n"
         "<current_draft>\n{draft}\n</current_draft>\n\n<earlier_document>\n{cand}\n</earlier_document>\n\n{question}")
Q = {"T": ("Does the earlier document concern the same situation or agenda item as the current draft?\n1 = a different situation\n"
           "2 = a related situation (same region or theme, but a different situation)\n3 = the same situation or agenda item\n"
           "Answer with the number only (1, 2 or 3).", ["1", "2", "3"]),
     "C": ("Do the two documents concern the same countries or parties?\n1 = no overlap\n2 = partial overlap\n"
           "3 = the same main countries or parties\nAnswer with the number only (1, 2 or 3).", ["1", "2", "3"]),
     "A": ("Does the earlier document take the same kind of Council action as the current draft (for example sanctions, a peacekeeping "
           "or political mission mandate, a condemnation or demand, a call for a ceasefire or humanitarian access, a referral or "
           "accountability measure, or a procedural decision)?\n1 = a different kind of action\n2 = a similar kind of action\n"
           "3 = the same kind of action\nAnswer with the number only (1, 2 or 3).", ["1", "2", "3"]),
     "O": ("Judging only by content (situation, countries and kind of action), is the earlier document an appropriate precedent "
           "for deciding how to vote on the current draft?\n1 = no\n2 = yes\nAnswer with the number only (1 or 2).", ["1", "2"])}


def texts(rec, q, c, cond):
    draft = mask(E4.ctx(q)); cand = f"Date: {c['date'].date()}\nSummary: {fmt(c.get('summary'), 3000)}"
    if cond == "R":
        draft += "\n" + "\n".join(f"{k}: {fmt(q.get(f))}" for k, f in META)
        cand += "\n" + "\n".join(f"{k}: {fmt(c.get(f))}" for k, f in META)
    return draft, cand


def build_pairs():
    cfg = ConfigA(); conf = lambda r: tuple(cfg.vot[r][n] for n in P5); P = {}
    for q in sorted(cfg.N, key=lambda r: r["date"]):
        dz = design(cfg, q)
        if dz:
            o, dis, _ = dz
            for i, d in enumerate([o] + dis): P.setdefault((q["res_id"], d["res_id"]), []).append(f"E10:{i}")
        for k, d in enumerate(cfg.strat_slots(q, 4, True), 1): P.setdefault((q["res_id"], d["res_id"]), []).append(f"E7:{k}")
    rows = [dict(qid=a, cid=b, roles="|".join(r), match=conf(a) == conf(b)) for (a, b), r in P.items()]
    df = pd.DataFrame(rows); df.to_parquet(PAIRS, index=False)
    e10 = df[df.roles.str.contains("E10")]; e7 = df[df.roles.str.contains("E7")]
    print(f"[pairs] {len(df)} unique pairs | E10 {len(e10)} ({e10.qid.nunique()} drafts) | E7 {len(e7)} ({e7.qid.nunique()} drafts) | matches {int(df.match.sum())}")
    # blind human-annotation sheet: 30 E10 drafts x 4 candidates, provided information only, order shuffled, no votes
    rec = {r["res_id"]: r for r in cfg.A + cfg.N}; rng = random.Random(0)
    ds = sorted(e10.qid.unique()); rng.shuffle(ds); sheet, key = [], []
    for j, qid in enumerate(ds[:30], 1):
        cs = list(e10[e10.qid == qid].itertuples()); rng.shuffle(cs)
        for k, r in enumerate(cs, 1):
            draft, cand = texts(rec, rec[qid], rec[r.cid], "P"); pid = f"D{j:02d}-C{k}"
            sheet.append(dict(pair_id=pid, draft_symbol=rec[qid]["symbol"], draft_text=draft[:2500], candidate_text=cand,
                              same_situation_1to3="", same_countries_1to3="", same_action_1to3="", appropriate_yes_no="", notes=""))
            key.append(dict(pair_id=pid, qid=qid, cid=r.cid, match=bool(r.match)))
    pd.DataFrame(sheet).to_csv(LOGS / "e12_human_sheet.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(STORE / "e12_human_key.csv", index=False)
    print(f"[human] wrote logs/e12_human_sheet.csv ({len(sheet)} pairs, 30 drafts); key kept in results/store/e12_human_key.csv (do not open before annotating)")
    print(f"[mask] sentences removed while building the sheet: {NMASK['n']}")


async def run(a):
    cfg = ConfigA(); rec = {r["res_id"]: r for r in cfg.A + cfg.N}
    df = pd.read_parquet(PAIRS); i, n = map(int, a.shard.split("/"))
    df = df[[hash_(q) % n == i for q in df.qid]]
    llm = LLM(a.served, E4.MODELS[a.served], base=a.base, conc=a.conc); await verify_served(llm, E4.MODELS[a.served])
    out = STORE / f"e12_{a.served}_s{i}.jsonl"
    done = {(r["qid"], r["cid"], r["cond"], r["q"]) for r in map(json.loads, open(out)) if r.get("error") is None} if out.exists() else set()
    lock = asyncio.Lock(); cnt = [0]; t0 = time.time(); sample = {}

    async def one(qid, cid, cond, qq):
        err = p = ntok = None
        try:
            draft, cand = texts(rec, rec[qid], rec[cid], cond)
            user = JUDGE.format(draft=draft, cand=cand, question=Q[qq][0]); sample.setdefault(f"{cond}-{qq}", user)
            p, ntok = await score_opts(llm, user, Q[qq][1])
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        async with lock:
            with open(out, "a") as f: f.write(json.dumps(dict(qid=qid, cid=cid, cond=cond, q=qq, p=p, ntok=ntok, judge=llm.m, error=err)) + "\n")
            cnt[0] += 1
            if cnt[0] % 200 == 0: print(f"  {cnt[0]} done | {time.time() - t0:.0f}s", flush=True)

    tasks = [one(r.qid, r.cid, c, qq) for r in df.itertuples() for c in ("P", "R") for qq in Q if (r.qid, r.cid, c, qq) not in done]
    print(f"shard {i}/{n}: {len(df)} pairs | {len(tasks)} tasks | judge {a.served}", flush=True)
    await asyncio.gather(*tasks)
    if sample: (LOGS / f"e12_prompt_samples_{a.served}.txt").write_text("\n\n".join(f"########## {k}\n{v}" for k, v in sample.items()))
    rows = [json.loads(l) for l in open(out)] if out.exists() else []
    print(f"wall {time.time() - t0:.0f}s | rows {len(rows)} | error rows {sum(1 for r in rows if r.get('error'))} | "
          f"max tokens {max([r['ntok'] or 0 for r in rows] or [0])} | sentences masked {NMASK['n']}", flush=True)


def hash_(s): return sum(map(ord, s))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", action="store_true"); ap.add_argument("--served", default=""); ap.add_argument("--base", default="")
    ap.add_argument("--shard", default="0/1"); ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args()
    build_pairs() if a.pairs else asyncio.run(run(a))
