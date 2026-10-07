"""E9 (exploratory replication in a second institution, D92): UN General Assembly roll calls of the same five members
(TidyTuesday 2021-03-23 extract of the UN votes data of Voeten et al., 1946-2019). No outcome filter applies: every recorded vote is in
the archive, and recorded votes are mostly contested, so the dominant mode differs from the Security Council's.
Documents: resolution-level roll calls (no amendment or paragraph votes) with all five votes recorded. Text: short title + description.
Queries: roll calls from 2013 on; pool: all roll calls dated strictly earlier. Exact five-vote configuration, as for UNSC.
Prediction tested: crowding depends on whether the query's configuration is a minority among its neighbours, not on dissent itself;
whole-pool configuration stratification recovers matches wherever they exist.
Writes data/processed/unga_p5.parquet. Usage: python -m coalrag.experiments.e9_unga [--smoke]"""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd

PROJ = Path(os.environ["PROJ"]); U = PROJ / "data" / "raw" / "unga"; OUT = PROJ / "data" / "processed"
CODES = ["CN", "FR", "GB", "RU", "US"]; P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]; VM = {"yes": "Y", "no": "N", "abstain": "A"}
M = "$HF_MODELS"; EMBS = {"bge-small": "BAAI/bge-small-en-v1.5", "bge-large": "BAAI/bge-large-en-v1.5"}


def build():
    v = pd.read_csv(U / "unvotes.csv"); rc = pd.read_csv(U / "roll_calls.csv")
    w = v[v.country_code.isin(CODES)].pivot_table(index="rcid", columns="country_code", values="vote", aggfunc="first")
    rc = rc.set_index("rcid"); rc["date"] = pd.to_datetime(rc.date)
    for c in ("amend", "para"): rc[c] = pd.to_numeric(rc[c], errors="coerce").fillna(0)
    d = rc.join(w, how="inner"); n0 = len(d)
    d = d[(d.amend != 1) & (d.para != 1)]; n1 = len(d)
    d = d.dropna(subset=CODES)
    for c, p in zip(CODES, P5): d[p] = d[c].map(VM)
    d = d.dropna(subset=P5).copy()
    s, ds = d["short"].fillna("").str.strip(), d["descr"].fillna("").str.strip()
    d["text"] = np.where((ds == "") | (ds.str.lower() == s.str.lower()), s, s + ". " + ds)
    d["config"] = d[P5].agg("".join, axis=1)
    d = d[d.text.str.len() > 0].sort_values("date").reset_index()
    print(f"roll calls with P5 votes {n0} | resolution-level {n1} | complete P5 and text {len(d)} | {d.date.min().date()} to {d.date.max().date()}")
    OUT.mkdir(parents=True, exist_ok=True); d[["rcid", "date", "session", "unres", "importantvote", "short", "descr", "text", "config"] + P5].to_parquet(OUT / "unga_p5.parquet", index=False)
    return d


def encode(name, texts, smoke):
    if smoke: E = np.random.default_rng(len(name)).normal(size=(len(texts), 16)); return E / np.linalg.norm(E, axis=1, keepdims=True)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(f"{M}/{EMBS[name]}"); m.max_seq_length = 256
    return m.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)


def strat(order, cf, depth=None, k=10):
    seen, out = set(), []
    for j in (order if depth is None else order[:depth]):
        if cf[j] not in seen: seen.add(cf[j]); out.append(j)
        if len(out) == k: break
    return out


def main(smoke, dedup=False):
    d = build(); cf = d.config.values; dates = d.date.values
    title = d["short"].fillna("").str.lower().str.replace(r"[^a-z0-9 ]", " ", regex=True).str.split().str.join(" ").values
    if dedup: print("DEDUP: precedents with the same short title as the query are excluded (recurring annual resolutions)")
    print("configuration shares, whole archive (CHN FRA GBR RUS USA):"); print(d.config.value_counts(normalize=True).head(8).round(3).to_string())
    qs = [i for i in np.where(d.date >= "2013-01-01")[0] if (dates < dates[i]).sum() >= 50]
    rep = np.mean([(title[:i] == title[i]).any() for i in qs]); print(f"queries whose exact short title occurred before: {rep:.1%}")
    print(f"queries from 2013: {len(qs)}"); rows = []
    for name in EMBS:
        E = encode(name, d.text.values, smoke)
        for i in qs:
            pool = np.where((dates < dates[i]) & ((title != title[i]) if dedup else True))[0]; o = pool[np.argsort(-(E[pool] @ E[i]), kind="stable")]; h = cf[o] == cf[i]
            if len(pool) < 10: continue
            top = pd.Series(cf[o[:50]]).value_counts(); maj = top.index[0] == cf[i]
            rows.append(dict(ret=name, qid=int(d.rcid[i]), config=cf[i], status="majority in its top 50" if maj else "minority in its top 50",
                             sup=h.any(), r50=h[:50].any(), r10=h[:10].any(), r1=h[:1].any(), share50=h[:50].mean(),
                             s10_d50=any(cf[j] == cf[i] for j in strat(o, cf, 50)), s10=any(cf[j] == cf[i] for j in strat(o, cf))))
    df = pd.DataFrame(rows); top6 = df[df.ret == "bge-small"].config.value_counts().index[:6]
    df["group"] = np.where(df.config.isin(top6), df.config, "other")
    cols = ["sup", "r50", "r10", "r1", "s10_d50", "s10"]

    def show(by, title):
        g = df.groupby(["ret", by]); t = (100 * g[cols].mean()).round(1); t.insert(0, "n", g.size())
        t["lift1"] = (g.r1.mean() / g.share50.mean().clip(lower=1e-9)).round(2)
        with pd.option_context("display.width", 200): print(f"\n==== E9 UNGA: {title} (%) ====\n" + t.to_string())
    show("status", "funnel by whether the query's configuration is the majority among its dense top 50")
    show("group", "funnel by query configuration (CHN FRA GBR RUS USA; Y favour, N against, A abstention)")
    rng = np.random.default_rng(0)
    for name in EMBS:
        x = df[(df.ret == name) & (df.status.str.startswith("minority"))]; dd = (x.s10.astype(float) - x.r10.astype(float)).values
        bs = dd[rng.integers(0, len(dd), (5000, len(dd)))].mean(1)
        print(f"{name}: minority queries (n={len(x)}) whole-pool stratified minus dense top 10: {100 * dd.mean():+.1f} pp "
              f"[{100 * np.percentile(bs, 2.5):+.1f}, {100 * np.percentile(bs, 97.5):+.1f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--dedup", action="store_true"); a = ap.parse_args(); main(a.smoke, a.dedup)
