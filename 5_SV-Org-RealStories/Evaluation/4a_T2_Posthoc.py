"""KW+Dunn between-group analysis on T2 rank_biserial values, grouped by each feature."""
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import scikit_posthocs as sp
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["JustDetention", "SurvivorStories", "both"], default="both")
args = parser.parse_args()

in_dir  = Path("../StatisticalTestResults/MYTHS")
out_dir = Path("../StatisticalTestResults/POST-MYTHS")
out_dir.mkdir(parents=True, exist_ok=True)

SINGLE_MYTHS = {"clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"}
FEATURES     = ["model", "myth", "frame", "dose", "task", "dataset", "prompt_variant"]

datasets = ["JustDetention", "SurvivorStories"] if args.dataset == "both" else [args.dataset]
dfs = []
for ds in datasets:
    for task in ["advice", "summary"]:
        path = in_dir / f"T2-MythAlignment-{ds}-{task}.csv"
        if path.exists():
            tmp = pd.read_csv(path)
            tmp["task"]    = task
            tmp["dataset"] = ds
            dfs.append(tmp)

df = pd.concat(dfs, ignore_index=True)
df = df[df["myth"].isin(SINGLE_MYTHS)]

kw_rows, dunn_rows = [], []
for feature in FEATURES:
    feat_df = df[df["prompt_variant"] != "default"] if feature == "prompt_variant" else df
    groups  = {k: g["rank_biserial"].values for k, g in feat_df.groupby(feature)}
    if len(groups) < 2:
        continue
    stat, p  = kruskal(*groups.values())
    n_total  = sum(len(v) for v in groups.values())
    eta2     = (stat - len(groups) + 1) / (n_total - len(groups))
    kw_rows.append({"feature": feature, "stat": stat, "p_value": p, "n_groups": len(groups), "eta2": round(eta2, 4)})
    print(f"{feature}: KW stat={stat:.3f} p={p:.4e} eta2={eta2:.4f}")
    if p > 0.05:
        continue
    dunn_raw = sp.posthoc_dunn(feat_df, val_col="rank_biserial", group_col=feature, p_adjust=None)
    pairs, raw_ps = [], []
    for g1 in dunn_raw.index:
        for g2 in dunn_raw.columns:
            if g1 < g2:
                pairs.append((g1, g2))
                raw_ps.append(dunn_raw.loc[g1, g2])
    _, p_bh, _, _ = multipletests(raw_ps, method="fdr_bh")
    for (g1, g2), p_raw, p_corr in zip(pairs, raw_ps, p_bh):
        m1     = feat_df[feat_df[feature] == g1]["rank_biserial"].median()
        m2     = feat_df[feat_df[feature] == g2]["rank_biserial"].median()
        n_pair = len(feat_df[feat_df[feature].isin([g1, g2])])
        z      = np.abs(stats.norm.ppf(p_raw / 2))
        rb     = (1 if m1 > m2 else -1) * z / np.sqrt(n_pair)
        dunn_rows.append({"feature": feature, "group1": g1, "group2": g2,
                          "p_value": p_raw, "p_bh": p_corr, "significant": p_corr < 0.05,
                          "median_g1": round(m1, 4), "median_g2": round(m2, 4),
                          "rank_biserial": round(rb, 4)})

suffix = args.dataset
pd.DataFrame(kw_rows).to_csv(out_dir / f"KW-T2-delta-{suffix}.csv", index=False)
pd.DataFrame(dunn_rows).to_csv(out_dir / f"Dunn-T2-delta-{suffix}.csv", index=False)