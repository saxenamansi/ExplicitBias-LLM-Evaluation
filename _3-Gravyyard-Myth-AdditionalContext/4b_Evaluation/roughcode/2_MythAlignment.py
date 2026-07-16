"""
2_MythAlignment.py
===================
Step 2: Myth-alignment of semantic shift.

Metrics:
  proj_subspace_delta  — projection of (T2 - T1) onto myth-alignment subspace
  proj_myth_unit_delta — cosine of shift direction with myth unit vector
  proj_t1_subspace     — absolute projection of T1 onto subspace
  proj_t2_subspace     — absolute projection of T2 onto subspace

Reads pkl output from 1_SemanticShift.py (contains embedding arrays).
Output CSV is the input to 3_ANOVA.py.

Usage:
  python 2_MythAlignment.py --append path/ExpAppend_WithEmbeddings.pkl
                              --organic path/ExpOrganic_WithEmbeddings.pkl
                              --task advice
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from Config import MYTH_TYPES, MYTH_PAIRS, get_output_dir
from Embeddings import get_subspace, get_myth_unit_vecs, cosine_sim
from StatisticalTests import paired_test, independent_test

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--append",  required=True, help="ExpAppend pkl")
parser.add_argument("--organic", required=True, help="ExpOrganic pkl")
parser.add_argument("--task",    choices=["advice", "summarization"])
args = parser.parse_args()

OUT_DIR = get_output_dir(args.task, "2_MythAlignment")
print(f"Output dir: {OUT_DIR}")

subspace       = get_subspace()
myth_unit_vecs = get_myth_unit_vecs()

# ── Load inputs from Step 1 ───────────────────────────────────────────────────
print(f"Loading Exp-Append from {args.append}")
append_df = pd.read_pickle(args.append)
print(f"  Rows: {len(append_df)}")

print(f"Loading Exp-Organic from {args.organic}")
organic_df = pd.read_pickle(args.organic)
print(f"  Rows: {len(organic_df)}")

# ── Exp-Append: compute projection deltas ────────────────────────────────────
print("\nComputing alignment scores (Exp-Append)...")

def compute_shift_alignment(row) -> dict:
    t1_vec = np.array(row["sbert_t1"])
    t2_vec = np.array(row["sbert_t2"])
    delta  = t2_vec - t1_vec

    proj_sub_delta = float(np.dot(delta, subspace)) if subspace is not None else np.nan
    proj_t1        = float(np.dot(t1_vec, subspace)) if subspace is not None else np.nan
    proj_t2        = float(np.dot(t2_vec, subspace)) if subspace is not None else np.nan

    myth_type = row.get("myth_type")
    if myth_type and myth_type in myth_unit_vecs:
        delta_norm      = delta / (np.linalg.norm(delta) + 1e-10)
        proj_myth_delta = float(np.dot(delta_norm, myth_unit_vecs[myth_type]))
    else:
        proj_myth_delta = np.nan

    return {
        "proj_subspace_delta":  proj_sub_delta,
        "proj_t1_subspace":     proj_t1,
        "proj_t2_subspace":     proj_t2,
        "proj_myth_unit_delta": proj_myth_delta,
    }

alignment_cols = pd.DataFrame(
    [compute_shift_alignment(row) for _, row in append_df.iterrows()]
)
append_df = pd.concat([append_df.reset_index(drop=True),
                       alignment_cols.reset_index(drop=True)], axis=1)

# Statistical tests on alignment metrics
align_results = []
for myth in MYTH_TYPES:
    sub = append_df[append_df["myth_type"] == myth]
    for metric, col in [("proj_subspace_delta",  "proj_subspace_delta"),
                        ("proj_myth_unit_delta", "proj_myth_unit_delta")]:
        vals = sub[col].dropna().tolist()
        if len(vals) < 5:
            continue
        r = paired_test(vals, [0.0] * len(vals))
        r.update({"myth": myth, "metric": metric, "is_pair": False,
                  "mean": sub[col].mean(), "median": sub[col].median()})
        align_results.append(r)

for myth_pair in MYTH_PAIRS:
    sub = append_df[append_df["myth_pair"] == myth_pair]
    vals = sub["proj_subspace_delta"].dropna().tolist()
    if len(vals) < 5:
        continue
    r = paired_test(vals, [0.0] * len(vals))
    r.update({"myth": myth_pair, "metric": "proj_subspace_delta", "is_pair": True,
              "mean": sub["proj_subspace_delta"].mean(),
              "median": sub["proj_subspace_delta"].median()})
    align_results.append(r)

out = OUT_DIR / "ExpAppend_MythAlignment_StatTests.csv"
pd.DataFrame(align_results).to_csv(out, index=False)
print(f"  Saved: {out.name}")

# ── Exp-Organic: projection of T1 responses ───────────────────────────────────
print("\nComputing alignment scores (Exp-Organic)...")

organic_df["proj_subspace"] = [
    float(np.dot(np.array(v), subspace)) if subspace is not None else np.nan
    for v in organic_df["sbert_t1"]
]
organic_df["proj_myth_unit"] = organic_df.apply(
    lambda r: float(np.dot(np.array(r["sbert_t1"]), myth_unit_vecs[r["myth_type"]]))
    if r["myth_type"] in myth_unit_vecs else np.nan,
    axis=1,
)

# Build neutral baseline
neutral_rows = []
from 0_Config import NLI_CSV_SAMPLE
nli_df   = pd.read_csv(NLI_CSV_SAMPLE)
myth_nli = nli_df[nli_df["myth_category"] == "MYTH"]
nli_labels = (
    myth_nli.groupby("narrative_index")
    .apply(lambda g: dict(zip(g["myth_type"], g["overall_label"])))
    .to_dict()
)

# Load T1 from organic_df to get all neutral responses
# CHECK: organic_df has response_t1 column
all_t1 = organic_df[["narrative_idx", "model", "myth_type",
                      "response_t1", "prompt_variant", "sbert_t1"]].copy()

neutral_baseline_rows = []
for _, row in all_t1.drop_duplicates(["narrative_idx", "model"]).iterrows():
    idx    = int(row["narrative_idx"])
    labels = nli_labels.get(idx, {})
    for myth_type in MYTH_TYPES:
        if labels.get(myth_type, "neutral") == "neutral":
            neutral_baseline_rows.append({
                "narrative_idx": idx,
                "model":         row["model"],
                "myth_type":     myth_type,
                "sbert_t1":      row["sbert_t1"],
            })

neutral_df = pd.DataFrame(neutral_baseline_rows)
neutral_df["proj_subspace"] = [
    float(np.dot(np.array(v), subspace)) if subspace is not None else np.nan
    for v in neutral_df["sbert_t1"]
]
neutral_df["proj_myth_unit"] = neutral_df.apply(
    lambda r: float(np.dot(np.array(r["sbert_t1"]), myth_unit_vecs[r["myth_type"]]))
    if r["myth_type"] in myth_unit_vecs else np.nan,
    axis=1,
)

# Group comparison: organic (entailment/contradiction) vs neutral
organic_stat_rows = []
for myth in MYTH_TYPES:
    for label in ["entailment", "contradiction"]:
        grp_org = organic_df[
            (organic_df["myth_type"] == myth) &
            (organic_df["narrative_nli_label"] == label)
        ]
        grp_neu = neutral_df[neutral_df["myth_type"] == myth]
        if len(grp_org) < 5 or len(grp_neu) < 5:
            continue
        for metric_name, col in [("proj_subspace",  "proj_subspace"),
                                  ("proj_myth_unit", "proj_myth_unit")]:
            result = independent_test(
                grp_org[col].dropna().tolist(),
                grp_neu[col].dropna().tolist(),
            )
            result.update({
                "myth":         myth,
                "organic_label": label,
                "metric":        metric_name,
                "mean_organic":  grp_org[col].mean(),
                "mean_neutral":  grp_neu[col].mean(),
            })
            organic_stat_rows.append(result)

out = OUT_DIR / "ExpOrganic_MythAlignment_IndependentGroupComparison.csv"
pd.DataFrame(organic_stat_rows).to_csv(out, index=False)
print(f"  Saved: {out.name}")

# ── Save outputs ──────────────────────────────────────────────────────────────
out = OUT_DIR / "ExpAppend_WithAlignment.pkl"
append_df.to_pickle(out)
print(f"  Saved (Exp-Append with alignment): {out.name}")

out = OUT_DIR / "ExpOrganic_WithAlignment.pkl"
organic_df.to_pickle(out)
print(f"  Saved (Exp-Organic with alignment): {out.name}")

# Scalar CSV for inspection
keep = ["narrative_idx", "model", "myth_type", "myth_pair", "frame", "dose",
        "is_pair", "prompt_variant",
        "proj_subspace_delta", "proj_t1_subspace", "proj_t2_subspace",
        "proj_myth_unit_delta",
        "sbert_cosine_t1_t2", "sbert_cosine_distance_t1_t2",
        "w2v_cosine_t1_t2",   "w2v_cosine_distance_t1_t2",
        "glove_cosine_t1_t2", "glove_cosine_distance_t1_t2"]
out = OUT_DIR / "ExpAppend_AlignmentScores.csv"
append_df[[c for c in keep if c in append_df.columns]].to_csv(out, index=False)
print(f"  Saved (scores CSV): {out.name}")

print("\nStep 2 complete.")
print(f"Next: python 3_ANOVA.py "
      f"--append {OUT_DIR}/ExpAppend_WithAlignment.pkl "
      f"--organic {OUT_DIR}/ExpOrganic_WithAlignment.pkl "
      f"--task {args.task}")