"""HDA = 1 - JSD_2(P_model || P_ref): agreement with a historical reference distribution (not calibration).
P_model: mean option probabilities (soft) or argmax frequencies (hard), per nation and condition.
References by period and vote type, all strictly pre-2013; 'oracle' row = the true label distribution of the evaluated drafts."""
import argparse, os
from pathlib import Path
import numpy as np, pandas as pd
P5 = ["CHN", "FRA", "GBR", "RUS", "USA"]; LAB = ["Y", "N", "A"]
def jsd(p, q):
    m = (p + q) / 2; kl = lambda a, b: float(np.sum(np.where(a > 0, a * np.log2(np.where(a > 0, a, 1) / np.where(b > 0, b, 1e-12)), 0)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)
def dist(s):
    v = s[s.isin(LAB)]; return np.array([(v == c).mean() for c in LAB]) if len(v) else None
def main(path, conds):
    PROC = Path(os.environ["PROJ"]) / "data" / "processed"
    ckg = pd.read_parquet(PROC / f"ckg_{os.environ.get('CKG_VERSION', 'v1_2')}.parquet"); pre = ckg[ckg.date < "2013-01-01"]
    REFS = {"all pre-2013": pre, "1946-1990": pre[pre.date < "1991-01-01"], "1991-2012": pre[pre.date >= "1991-01-01"],
            "2003-2012": pre[pre.date >= "2003-01-01"], "adopted consensus+split": pre[pre.adopted & pre.regime.isin(["consensus", "split"])],
            "adopted split only": pre[pre.regime == "split"], "non-adopted (recorded; vetoer-biased)": pre[~pre.adopted]}
    ref = {(k, n): dist(v[n]) for k, v in REFS.items() for n in P5}
    df = pd.read_json(path, lines=True); df = df[df.error.isna()].drop_duplicates(["qid", "nation", "cond"], keep="last")
    if conds: df = df[df.cond.str.contains(conds)]
    rows = []
    for (c, n), g in df.groupby(["cond", "nation"]):
        soft = g[["pY", "pN", "pA"]].mean().values; hard = np.array([(g[["pY", "pN", "pA"]].values.argmax(1) == i).mean() for i in range(3)])
        truth = dist(g.true)
        for k in REFS:
            r = ref[(k, n)]
            if r is None: continue
            rows.append(dict(cond=c, nation=n, ref=k, soft=1 - jsd(soft, r), hard=1 - jsd(hard, r), oracle=1 - jsd(truth, r)))
        rows.append(dict(cond=c, nation=n, ref="TRUE labels (target)", soft=1 - jsd(soft, truth), hard=1 - jsd(hard, truth), oracle=1.0))
    out = pd.DataFrame(rows)
    for col in ["soft", "hard"]:
        tab = out.groupby(["cond", "ref"])[col].mean().unstack("ref")
        tab.loc["ORACLE (true labels)"] = out[out.cond == out.cond.iloc[0]].groupby("ref").oracle.mean()
        with pd.option_context("display.width", 250): print(f"\n==== HDA ({col}), mean over nations ====\n{tab.round(3).to_string()}")
    print("\n-- per nation, reference 'all pre-2013', soft --")
    print(out[out.ref == "all pre-2013"].pivot_table(index="cond", columns="nation", values="soft").round(3).to_string())
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--conds", default="")
    a = ap.parse_args(); main(a.path, a.conds)
