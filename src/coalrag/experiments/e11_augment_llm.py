"""E11: corpus augmentation tested at final prediction (pre-registered H18, wave 6, configs/prereg.md).
Answers design review item (augmentation tested at retrieval AND final prediction). E8 (H15) tested retrieval only;
E11 re-uses E8's record selection unchanged and feeds the result to the LLM.
  Queries   the non-adopted drafts (LT-RAG test set) x P5 members with a recorded vote.
  Base pool the LT-RAG adopted documents dated strictly before the draft (outcome-filtered pool), as E8 base F.
  Arms      A0 none | A1 +100 most similar dissent-bearing | A2 +100 random dissent-bearing (E8 seeds 0, 1, 2)
            | A3 +100 most similar unanimous. Candidates: UNSC-CKG v1.2 records outside the LT-RAG documents, before 2013,
            all five P5 votes recorded (E8 definition, identical code).
  Evidence  four stratified precedents: bge-small (UNSC-CKG text space) ranking over the augmented pool, depth 50, the
            best-ranked document of each distinct five-vote configuration, rank order, coalition view, same template and
            option-likelihood scoring as D-strat4-coal. Every displayed summary is the UNSC-CKG text (added records have no
            released summary), so arms differ only in which documents are in the pool.
  Conditions B0 (no evidence), S4-<arm> for every arm, <arm>-match = the matching precedent alone when the four slots of that
            arm contain one (A1 pre-registered for H18c; A0 and A3 exploratory).
Usage: --design (CPU; fixes the evidence for every model in results/store/e11_design.json)
       --served NAME --base URL (scoring; output results/store/e11_<served>.jsonl; resumable)."""
import argparse, asyncio, hashlib, json, os, time, zlib
os.environ.setdefault("CKG_VERSION", "v1_2")
import numpy as np, pandas as pd
from coalrag.retrieval.configA import ConfigA, NAME
from coalrag.llm.client import LLM, verify_served
from coalrag.experiments import e5_wave2 as W

E4, STORE, LOGS = W.E4, W.STORE, W.LOGS
DESIGN = STORE / "e11_design.json"
K, SEEDS, DEPTH, KSLOT = 100, [0, 1, 2], 50, 4
ARMS = ["A0", "A1"] + [f"A2s{s}" for s in SEEDS] + ["A3"]
MATCH_ARMS = ["A0", "A1", "A3"]
LET = {1: "Y", 2: "N", 3: "A"}


def make_design():
    from coalrag.data.configB import load, P5 as P5B
    Kdf, T, TOK, V, H, EMB = load(partial_veto_coding=True, with_emb=True)
    E = EMB["bge-small"]; rid = Kdf.res_id.tolist(); pos = {r: i for i, r in enumerate(rid)}; dates = Kdf.date.values
    cfg = ConfigA()
    A = [pos[r["res_id"]] for r in cfg.A if r["res_id"] in pos]; N = [pos[r["res_id"]] for r in cfg.N if r["res_id"] in pos]
    choi = set(A) | set(N); complete = lambda i: bool(((V[i] >= 1) & (V[i] <= 3)).all())
    cand = np.array([i for i in range(len(rid)) if i not in choi and dates[i] < np.datetime64("2013-01-01") and complete(i)])
    MODE = {"unanimous": cand[(V[cand] == 1).all(1)], "dissent": cand[~(V[cand] == 1).all(1)]}
    print(f"[design] candidates {len(cand)} (unanimous {len(MODE['unanimous'])}, dissent-bearing {len(MODE['dissent'])}) | "
          f"LT-RAG docs mapped {len(choi)} | adopted base docs {len(A)} | non-adopted drafts mapped {len(N)} of {len(cfg.N)}")
    drafts, used, skipped, late = [], set(), [], 0
    for r in sorted(cfg.N, key=lambda r: r["date"]):
        if r["res_id"] not in pos: skipped.append((r["symbol"], "not in CKG v1.2")); continue
        q = pos[r["res_id"]]
        if not complete(q): skipped.append((r["symbol"], "incomplete P5 votes")); continue
        bp = [j for j in A if dates[j] < dates[q]]
        if len(bp) < 10: skipped.append((r["symbol"], f"base pool {len(bp)} < 10")); continue
        def similar(C):
            return C[np.argsort(-(E[C] @ E[q]), kind="stable")[:min(K, len(C))]].tolist()
        def random_(C, sd):  # identical generator and key to E8 (e8_augment.py)
            return np.random.default_rng(zlib.crc32(f"{rid[q]}|dissent|{K}|{sd}".encode())).choice(C, min(K, len(C)), replace=False).tolist()
        add = {"A0": [], "A1": similar(MODE["dissent"]), "A3": similar(MODE["unanimous"])}
        for sd in SEEDS: add[f"A2s{sd}"] = random_(MODE["dissent"], sd)
        arms = {}
        for a in ARMS:
            late += sum(1 for j in add[a] if not dates[j] < dates[q])
            P = np.asarray(bp + [j for j in add[a] if dates[j] < dates[q]], dtype=int)
            o = np.argsort(-(E[P] @ E[q]), kind="stable"); seen, sl = set(), []
            for j in o[:DEPTH]:
                key = tuple(V[P[j]])
                if key in seen: continue
                seen.add(key); sl.append(int(P[j]))
                if len(sl) == KSLOT: break
            m = (V[P] == V[q]).all(1); hits = [j for j in sl if bool((V[j] == V[q]).all())]
            arms[a] = dict(slots=[rid[j] for j in sl], match=rid[hits[0]] if hits else None, match_slot=(sl.index(hits[0]) + 1) if hits else None,
                           sup=bool(m.any()), m50=int(m[o[:DEPTH]].sum()), n_pool=int(len(P)), n_added=int(len(P) - len(bp)),
                           added_in_slots=int(sum(1 for j in sl if j not in set(bp))))
            used.update(sl)
        drafts.append(dict(qid=r["res_id"], symbol=r["symbol"], date=str(r["date"].date()), arms=arms))
    docs, mism = {}, 0
    for j in sorted(used):
        rec = Kdf.iloc[j]
        votes = {n: LET.get(int(V[j][k])) for k, n in enumerate(P5B)}
        if rid[j] in cfg.vot and any(cfg.vot[rid[j]][n] != votes[n] for n in P5B): mism += 1
        docs[rid[j]] = dict(res_id=rid[j], date=str(pd.Timestamp(rec["date"]).date()), adopted=bool(rec["adopted"]), summary=T[j], votes=votes)
    D = dict(prereg="wave 6 (H18)", K=K, seeds=SEEDS, depth=DEPTH, kslot=KSLOT, drafts=drafts, docs=docs)
    print(f"[design] drafts used {len(drafts)} | skipped {len(skipped)} {skipped} | added docs dated after the draft (dropped) {late}")
    print(f"[design] displayed documents {len(docs)} | LT-RAG documents whose CKG v1.2 votes differ from the LT-RAG votes {mism}")
    print("[design] manipulation check (share of drafts; n = %d)" % len(drafts))
    print(f"  {'arm':6s} {'4 slots contain a match':>24s} {'Sup (match in pool)':>20s} {'mean m50':>9s} {'added docs in 4 slots':>22s}")
    for a in ARMS:
        x = [d["arms"][a] for d in drafts]
        print(f"  {a:6s} {np.mean([v['match'] is not None for v in x]):24.3f} {np.mean([v['sup'] for v in x]):20.3f} "
              f"{np.mean([v['m50'] for v in x]):9.2f} {np.mean([v['added_in_slots'] for v in x]):22.2f}")
    lens = [len(docs[i]["summary"]) for i in docs]
    print(f"[design] displayed summary length (chars): median {int(np.median(lens))}, max {max(lens)}")
    return D


def digest(D): return hashlib.sha256(json.dumps(D, sort_keys=True).encode()).hexdigest()[:12]


def design_main():
    D = make_design(); h = digest(D)
    if DESIGN.exists():
        old = json.load(open(DESIGN))
        print(f"[design] existing {DESIGN.name} sha {digest(old)} | recomputed sha {h} | {'UNCHANGED' if digest(old) == h else 'DIFFERENT (kept the existing file)'}")
    else:
        json.dump(D, open(DESIGN, "w")); print(f"[design] wrote {DESIGN} sha {h}")


async def score_main(a):
    D = json.load(open(DESIGN)); print(f"[design] {DESIGN.name} sha {digest(D)}", flush=True)
    docs = D["docs"]
    for d in docs.values(): d["date"] = pd.Timestamp(d["date"])
    over = {k: v["votes"] for k, v in docs.items()}
    cfg = ConfigA(); qrec = {r["res_id"]: r for r in cfg.N}
    for k, v in over.items(): cfg.vot.setdefault(k, v)  # W.block evaluates cfg.vot[res_id] eagerly; displayed votes come from `over`
    llm = LLM(a.served, E4.MODELS[a.served], base=a.base, conc=a.conc)
    await verify_served(llm, E4.MODELS[a.served])
    out = STORE / f"e11_{a.served}.jsonl"
    done = {(r["qid"], r["nation"], r["cond"]) for r in map(json.loads, open(out)) if r.get("error") is None} if out.exists() else set()
    lock = asyncio.Lock(); cnt = [0]; t0 = time.time(); sample = {}

    async def row(q, n, cond, ids, extra):
        t = time.time(); err = p = ntok = None
        try:
            user = (W.PR.WITHOUT_HISTORY.format(country=NAME[n], answer_format=W.PR.ANSWER_ONE_WORD, query=W.ctx(q)) if not ids
                    else W.d_prompt(NAME[n], q, W.block(cfg, [docs[i] for i in ids], n, "coalition", over)))
            sample.setdefault(cond, f"[{q['symbol']} | {NAME[n]}]\n{user}"); p, ntok = await llm.score(user)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:200]}"
        rec = dict(qid=q["res_id"], symbol=q["symbol"], qset="vetoed" if q["vetoed"] else "failed", nation=n, cond=cond, model=llm.m,
                   pY=p["Y"] if p else None, pN=p["N"] if p else None, pA=p["A"] if p else None, pred=max(p, key=p.get) if p else None,
                   true=cfg.vot[q["res_id"]][n], retrieved=list(ids), prompt_tokens=ntok, sec=round(time.time() - t, 2), error=err, **extra)
        async with lock:
            with open(out, "a") as f: f.write(json.dumps(rec) + "\n")
            cnt[0] += 1
            if cnt[0] % 200 == 0: print(f"  {cnt[0]} done | {time.time() - t0:.0f}s", flush=True)

    tasks = []
    for d in D["drafts"]:
        q = qrec[d["qid"]]
        for n in NAME:
            if cfg.vot[q["res_id"]][n] not in ("Y", "N", "A"): continue
            conds = [("B0", [], {})] + [(f"S4-{x}", d["arms"][x]["slots"], {"arm": x, "match_slot": d["arms"][x]["match_slot"]}) for x in ARMS]
            conds += [(f"{x}-match", [d["arms"][x]["match"]], {"arm": x, "match_slot": d["arms"][x]["match_slot"]}) for x in MATCH_ARMS if d["arms"][x]["match"]]
            tasks += [row(q, n, c, ids, ex) for c, ids, ex in conds if (q["res_id"], n, c) not in done]
    print(f"{len(D['drafts'])} drafts | {len(tasks)} tasks to run | model {a.served}", flush=True)
    await asyncio.gather(*tasks)
    if sample: (LOGS / f"e11_prompt_samples_{a.served}.txt").write_text("\n\n".join(f"########## {k}\n{v}" for k, v in sample.items()))
    rows = [json.loads(l) for l in open(out)] if out.exists() else []
    ok = {(r["qid"], r["nation"], r["cond"]) for r in rows if r.get("error") is None}
    toks = [r["prompt_tokens"] for r in rows if r.get("prompt_tokens")]
    print(f"wall {time.time() - t0:.0f}s | rows {len(rows)} | distinct ok {len(ok)} | error rows {sum(1 for r in rows if r.get('error'))} | "
          f"prompt tokens max {max(toks) if toks else 0}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", action="store_true"); ap.add_argument("--served", default="")
    ap.add_argument("--base", default="http://127.0.0.1:8011/v1"); ap.add_argument("--conc", type=int, default=16)
    a = ap.parse_args()
    design_main() if a.design else asyncio.run(score_main(a))
