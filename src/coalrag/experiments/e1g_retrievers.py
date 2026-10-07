"""E1g (exploratory, D92): is crowding an artefact of a small retriever? The Config A' funnel (581 Choi documents, their summaries
as symmetric text, exact five-vote configuration, strictly earlier precedents) for larger embedders and for cross-encoder rerankers
of the bge-large top 50. Anchor: bge-small must reproduce the Config A' funnel (top 10: split 48.1, vetoed 47.4, failed 32.1).
lift1 = top-1 recall / mean share of matches in the top 50 (how much better than its own top-50 composition the ranker does).
Usage: python -m coalrag.experiments.e1g_retrievers [--smoke]"""
import argparse, os, time
os.environ.setdefault("CKG_VERSION", "v1_2")
import numpy as np, pandas as pd
from coalrag.data.configB import load
from coalrag.retrieval.configA import ConfigA

M = "$HF_MODELS"
EMBS = {"bge-small": "BAAI/bge-small-en-v1.5", "bge-large": "BAAI/bge-large-en-v1.5", "gte-large": "Alibaba-NLP/gte-large-en-v1.5",
        "mxbai-large": "MixedBread-AI/mxbai-embed-large-v1"}
RERANK = {"bge-reranker-v2-m3": "BAAI/bge-reranker-v2-m3", "mxbai-rerank-large": "MixedBread-AI/mxbai-rerank-large-v1"}
P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]; REGS = ["consensus", "split", "vetoed", "failed"]


def encode(name, texts, smoke):
    if smoke: E = np.random.default_rng(len(name)).normal(size=(len(texts), 16)); return E / np.linalg.norm(E, axis=1, keepdims=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(f"{M}/{EMBS[name]}", trust_remote_code=True); m.max_seq_length = 512
    return m.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)


def rerank(name, pairs, smoke):
    if smoke: return np.random.default_rng(1).normal(size=len(pairs))
    from sentence_transformers import CrossEncoder
    ce = CrossEncoder(f"{M}/{RERANK[name]}", max_length=512, trust_remote_code=True)
    return np.asarray(ce.predict(pairs, batch_size=32, show_progress_bar=False), float).ravel()


def strat(order, cf, k=10):
    seen, out = set(), []
    for j in order:
        if cf[j] not in seen: seen.add(cf[j]); out.append(j)
        if len(out) == k: break
    return out


def main(smoke):
    Kdf, *_ = load(partial_veto_coding=True, with_emb=True); reg = dict(zip(Kdf.res_id, Kdf.regime))
    cfg = ConfigA(); docs = [d for d in cfg.A + cfg.N if d["res_id"] in reg]
    ids = [d["res_id"] for d in docs]; text = [str(d.get("summary") or d.get("context") or "") for d in docs]
    dates = np.array([np.datetime64(d["date"]) for d in docs]); cf = [tuple(cfg.vot[i][x] for x in P5) for i in ids]
    ok = np.array([all(v in ("Y", "N", "A") for v in c) for c in cf]); R = np.array([reg[i] for i in ids])
    print(f"documents {len(docs)} | complete votes {ok.sum()} | empty texts {sum(t == '' for t in text)}")
    qs = [i for i in range(len(docs)) if ok[i] and (dates < dates[i]).sum() >= 10]
    rows, emb = [], {}
    for name in EMBS:
        t0 = time.time()
        try: emb[name] = encode(name, text, smoke)
        except Exception as e: print(f"{name}: FAILED {type(e).__name__}: {str(e)[:150]}"); continue
        S = emb[name] @ emb[name].T
        for i in qs:
            pool = np.where(dates < dates[i])[0]; o = pool[np.argsort(-S[i, pool], kind="stable")]; h = np.array([cf[j] == cf[i] for j in o])
            rows.append(dict(ret=name, regime=R[i], qid=ids[i], sup=h.any(), r50=h[:50].any(), r10=h[:10].any(), r1=h[:1].any(),
                             share50=h[:50].mean(), s10=float(any(cf[j] == cf[i] for j in strat(o, cf)))))
        print(f"{name}: encoded and ranked in {time.time() - t0:.0f}s")
    if "bge-large" in emb:
        S = emb["bge-large"] @ emb["bge-large"].T; cand = {}
        for i in qs:
            pool = np.where(dates < dates[i])[0]; cand[i] = pool[np.argsort(-S[i, pool], kind="stable")][:50]
        pairs = [(text[i], text[j]) for i in qs for j in cand[i]]
        for name in RERANK:
            t0 = time.time()
            try: sc = rerank(name, pairs, smoke)
            except Exception as e: print(f"{name}: FAILED {type(e).__name__}: {str(e)[:150]}"); continue
            p = 0
            for i in qs:
                c = cand[i]; s = sc[p:p + len(c)]; p += len(c); o = c[np.argsort(-s, kind="stable")]; h = np.array([cf[j] == cf[i] for j in o])
                full = np.where(dates < dates[i])[0]
                rows.append(dict(ret=f"bge-large + {name}", regime=R[i], qid=ids[i], sup=any(cf[j] == cf[i] for j in full), r50=h.any(),
                                 r10=h[:10].any(), r1=h[:1].any(), share50=h.mean(), s10=np.nan))
            print(f"{name}: {len(pairs)} pairs in {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows); out = []
    for (r, g), x in df.groupby(["ret", "regime"]):
        out.append(dict(retriever=r, regime=g, n=len(x), support=100 * x.sup.mean(), top50=100 * x.r50.mean(), top10=100 * x.r10.mean(),
                        top1=100 * x.r1.mean(), lift1=x.r1.mean() / max(x.share50.mean(), 1e-9), strat10_whole=100 * x.s10.mean()))
    t = pd.DataFrame(out); t["regime"] = pd.Categorical(t.regime, REGS, ordered=True)
    with pd.option_context("display.width", 200): print("\n==== E1g: funnel by retriever and regime (%) ====\n" + t.sort_values(["regime", "retriever"]).round(1).to_string(index=False))
    c = df[df.regime.isin(["split", "vetoed", "failed"])]
    print("\n-- contested drafts pooled (split + vetoed + failed): top-10 recall % --")
    print((100 * c.groupby("ret")[["sup", "r50", "r10", "r1", "s10"]].mean()).round(1).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); main(ap.parse_args().smoke)
