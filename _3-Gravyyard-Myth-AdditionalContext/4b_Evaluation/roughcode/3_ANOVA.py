"""
3_ANOVA.py
===========
Step 3: Factorial analysis — ANOVA and linear regression.

ANOVA factors: model, myth_type, frame, dose, is_pair, demographic features
Regression: predicts Cohen's d from model × myth × demographic indicators

Reads pkl output from 3b_MythAlignment.py.
Output CSVs are standalone — no downstream step depends on them.

Usage:
  python 3c_ANOVA.py --append path/ExpAppend_WithAlignment.pkl
                     --organic path/ExpOrganic_WithAlignment.pkl
                     --task advice
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from Config import (
    MYTH_TYPES, MYTH_PAIRS, DEMOGRAPHIC_FEATURES, get_output_dir,
)
from StatisticalTests import run_anova, run_regression, paired_test

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--append",  required=True)
parser.add_argument("--organic", required=True)
parser.add_argument("--task",    choices=["advice", "summarization"])
args = parser.parse_args()

OUT_DIR = get_output_dir(args.task, "3_ANOVA")
print(f"Output dir: {OUT_DIR}")

# ── Load inputs from Step 2 ───────────────────────────────────────────────────
print(f"Loading Exp-Append from {args.append}")
append_df = pd.read_pickle(args.append)

print(f"Loading Exp-Organic from {args.organic}")
organic_df = pd.read_pickle(args.organic)

# ── Target metrics ────────────────────────────────────────────────────────────
TARGET_METRICS = [
    ("sbert_cosine_distance",  "sbert_cosine_distance_t1_t2"),
    ("w2v_cosine_distance",    "w2v_cosine_distance_t1_t2"),
    ("glove_cosine_distance",  "glove_cosine_distance_t1_t2"),
    ("proj_subspace_delta",    "proj_subspace_delta"),
    ("proj_myth_unit_delta",   "proj_myth_unit_delta"),
]
# CHECK: if any of these columns are missing, verify 1 and 2 ran successfully

# ── Exp-Append ANOVA ──────────────────────────────────────────────────────────
print("\nRunning ANOVA (Exp-Append)...")
df = append_df.copy()
df["model_cat"] = pd.Categorical(df["model"])
df["myth_cat"]  = pd.Categorical(
    df["myth_type"].fillna(df.get("myth_pair", pd.Series(dtype=str)).fillna(""))
)
df["frame_cat"] = pd.Categorical(df["frame"])
df["dose_cat"]  = pd.Categorical(df["dose"].astype(str))
df["is_pair_int"] = df["is_pair"].astype(int)

demo_cols = [c for c in DEMOGRAPHIC_FEATURES if c in df.columns]
factor_cols = ["model_cat", "myth_cat", "frame_cat", "dose_cat", "is_pair_int"]

anova_results = {}
for metric_name, metric_col in TARGET_METRICS:
    result = run_anova(df, metric_col, factor_cols, demo_cols)
    if result is not None:
        result["metric"] = metric_name
        anova_results[metric_name] = result
        print(f"  ANOVA done: {metric_name}")

if anova_results:
    out = OUT_DIR / "ExpAppend_ANOVA_AllMetrics_AllFactors.csv"
    pd.concat(anova_results.values(), keys=anova_results.keys()).to_csv(out)
    print(f"  Saved: {out.name}")

# ── Exp-Append regression (Cohen's d as target) ───────────────────────────────
print("\nRunning regression (Exp-Append)...")
models = df["model"].unique().tolist()
regression_results = {}

for metric_name, metric_col in TARGET_METRICS:
    reg_rows = []
    for model in df["model"].unique():
        for myth in df["myth_cat"].unique():
            grp = df[(df["model"] == model) & (df["myth_cat"] == myth)]
            if len(grp) < 5:
                continue
            result     = paired_test(grp[metric_col].dropna().tolist(),
                                     [0.0] * len(grp.dropna(subset=[metric_col])))
            demo_means = {dc: grp[dc].mean() for dc in demo_cols if dc in grp.columns}
            row_d = {
                "model": model, "myth": myth, "metric": metric_name,
                "cohens_dz":  result["cohens_dz"],
                "cohens_dav": result["cohens_dav"],
                "cohens_drm": result["cohens_drm"],
                "hedges_g":   result["hedges_g"],
                "n":          result["n"],
                "is_pair":    "+" in str(myth),
                **demo_means,
            }
            for m in models:
                row_d[f"model_{m}"] = int(model == m)
            for mt in MYTH_TYPES + MYTH_PAIRS:
                row_d[f"myth_{mt.replace('+', '_')}"] = int(myth == mt)
            reg_rows.append(row_d)

    if not reg_rows:
        continue

    reg_df   = pd.DataFrame(reg_rows)
    feat_cols = (
        [f"model_{m}" for m in models] +
        [f"myth_{mt.replace('+', '_')}" for mt in MYTH_TYPES + MYTH_PAIRS] +
        demo_cols + ["is_pair"]
    )

    for effect_col in ["cohens_dz", "cohens_dav", "cohens_drm", "hedges_g"]:
        summary = run_regression(reg_df, feat_cols, effect_col, metric_name)
        if summary is not None:
            regression_results[f"{metric_name}_{effect_col}"] = summary

if regression_results:
    out = OUT_DIR / "ExpAppend_Regression_CohensD_AllMetrics.csv"
    pd.concat(regression_results.values(), ignore_index=True).to_csv(out, index=False)
    print(f"  Saved: {out.name}")

# ── Exp-Organic ANOVA ─────────────────────────────────────────────────────────
print("\nRunning ANOVA (Exp-Organic)...")
org = organic_df.copy()
org["model_cat"] = pd.Categorical(org["model"])
org["myth_cat"]  = pd.Categorical(org["myth_type"])
org["label_cat"] = pd.Categorical(org["narrative_nli_label"])
demo_cols_org    = [c for c in DEMOGRAPHIC_FEATURES if c in org.columns]

for metric_col in ["proj_subspace", "proj_myth_unit"]:
    if metric_col not in org.columns:
        print(f"  Skipping {metric_col} — not found in organic_df")
        continue
    result = run_anova(
        org, metric_col,
        ["model_cat", "myth_cat", "label_cat"],
        demo_cols_org,
    )
    if result is not None:
        out = OUT_DIR / f"ExpOrganic_ANOVA_{metric_col}.csv"
        result.to_csv(out)
        print(f"  Saved: {out.name}")

print("\nStep 3 complete.")