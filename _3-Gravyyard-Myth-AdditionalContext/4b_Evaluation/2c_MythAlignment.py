"""
2c_MythAlignment.py
===================
Step 2c: Statistical tests on myth-alignment projection scores.

ContextAppend:
  Paired test — projection scores (proj_myth_magnitude, proj_myth_direction)
  tested against a zero baseline (no directional shift).
  For myth pairs: proj_t2_subspace - proj_t1_subspace tested against zero.
  Grouped by prompt_variant when present (Advice); single group for Summarization.

Original-NoContextAppend:
  Independent test — projection scores (proj_subspace, proj_myth_unit) for
  entailing vs contradicting narratives compared against a neutral baseline.
  Neutral baseline is built from the NLI CSV: narratives where the myth label
  is "neutral" for a given myth type.
  Grouped by prompt_variant when present.

Reads pkl output from 2b_Projection.py (must contain projection score columns).

Outputs
-------
  ContextAppend_MythAlignment.csv
  Original_MythAlignment.csv

Output path: {task}/2_Projection/SampleResults/

Usage
-----
  # Both experiments
  python 2c_MythAlignment.py \\
      --ContextAppend path/to/ContextAppend_Projections.pkl \\
      --Original      path/to/Original_Projections.pkl \\
      --nli           path/to/nli.csv \\
      --model gemma [--task AdviceGeneration]

  # ContextAppend only
  python 2c_MythAlignment.py --ContextAppend path/to/ContextAppend_Projections.pkl --model gemma

  # Original only
  python 2c_MythAlignment.py \\
      --Original path/to/Original_Projections.pkl \\
      --nli path/to/nli.csv --model gemma
"""

import argparse
import numpy as np
import pandas as pd

from Config import MYTH_TYPES, MYTH_PAIRS, NLI_CSV_SAMPLE, get_output_dir
from Embeddings import get_subspace, get_myth_unit_vecs
from StatisticalTests import paired_test, independent_test, apply_bh_correction

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ContextAppend", default=None, help="Path to ContextAppend_Projections.pkl")
parser.add_argument("--Original",      default=None, help="Path to Original_Projections.pkl")
parser.add_argument("--nli",  default=str(NLI_CSV_SAMPLE), help="Path to NLI CSV (required for --Original)")
parser.add_argument("--task", choices=["AdviceGeneration", "Summarization"])
parser.add_argument("--model", required=True)
args = parser.parse_args()

if not args.ContextAppend and not args.Original:
    raise ValueError("Provide at least one of --ContextAppend or --Original.")
if args.Original and not args.nli:
    raise ValueError("--nli is required when running --Original.")

OUT_DIR = get_output_dir(args.task, "2_Projection/SampleResults/")
print(f"Output dir: {OUT_DIR}")

subspace       = get_subspace()
myth_unit_vecs = get_myth_unit_vecs()

PROJECTION_METRICS = [
    ("proj_myth_magnitude", "proj_myth_magnitude"),
    ("proj_myth_direction", "proj_myth_direction"),
]
ORIGINAL_METRICS = [
    ("proj_subspace",  "proj_subspace"),
    ("proj_myth_unit", "proj_myth_unit"),
]


def get_variants(df: pd.DataFrame) -> list:
    if "prompt_variant" in df.columns and df["prompt_variant"].notna().any():
        return sorted(df["prompt_variant"].dropna().unique().tolist())
    return [None]


def filter_variant(df: pd.DataFrame, variant) -> pd.DataFrame:
    if variant is not None:
        return df[df["prompt_variant"] == variant]
    return df


# ── ContextAppend: paired test against zero baseline ─────────────────────────
if args.ContextAppend:
    print(f"\nLoading ContextAppend from {args.ContextAppend}")
    append_df = pd.read_pickle(args.ContextAppend)
    print(f"  Rows: {len(append_df)}")

    variants = get_variants(append_df)
    print(f"  Prompt variants: {variants}")

    results = []

    for variant in variants:
        sub_all = filter_variant(append_df, variant)

        # Single myths
        for myth in MYTH_TYPES:
            sub = sub_all[sub_all["myth_type"] == myth]
            for metric_name, col in PROJECTION_METRICS:
                vals = sub[col].dropna().tolist()
                if len(vals) < 5:
                    continue
                r = paired_test(vals, [0.0] * len(vals))
                r.update({
                    "myth":           myth,
                    "metric":         metric_name,
                    "is_pair":        False,
                    "prompt_variant": variant,
                    "n":              len(vals),
                    "mean":           sub[col].mean(),
                    "median":         sub[col].median(),
                })
                results.append(r)

        # Myth pairs
        for myth_pair in MYTH_PAIRS:
            sub  = sub_all[sub_all["myth_pair"] == myth_pair]
            vals = (sub["proj_t2_subspace"] - sub["proj_t1_subspace"]).dropna().tolist()
            if len(vals) < 5:
                continue
            r = paired_test(vals, [0.0] * len(vals))
            r.update({
                "myth":           myth_pair,
                "metric":         "proj_subspace_delta",
                "is_pair":        True,
                "prompt_variant": variant,
                "n":              len(vals),
                "mean":           np.mean(vals),
                "median":         np.median(vals),
            })
            results.append(r)

    results = apply_bh_correction(results)
    out = OUT_DIR / "ContextAppend_MythAlignment.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"  Saved: {out.name}")


# ── Original: independent test vs neutral baseline ────────────────────────────
if args.Original:
    print(f"\nLoading Original from {args.Original}")
    original_df = pd.read_pickle(args.Original)
    print(f"  Rows: {len(original_df)}")

    variants = get_variants(original_df)
    print(f"  Prompt variants: {variants}")

    print(f"  Building neutral baseline from {args.nli}")
    nli_df   = pd.read_csv(args.nli)
    myth_nli = nli_df[nli_df["myth_category"] == "MYTH"]
    nli_labels = (
        myth_nli.groupby("narrative_index")
        .apply(lambda g: dict(zip(g["myth_type"], g["overall_label"])))
        .to_dict()
    )

    results = []

    for variant in variants:
        sub_original = filter_variant(original_df, variant)

        # Build neutral baseline — deduplicated per (narrative_idx, model) within variant
        dedup_cols = ["narrative_idx", "model", "sbert_t1"]
        t1_lookup  = sub_original[dedup_cols].drop_duplicates(["narrative_idx", "model"])

        neutral_rows = []
        for _, row in t1_lookup.iterrows():
            idx    = int(row["narrative_idx"])
            labels = nli_labels.get(idx, {})
            for myth_type in MYTH_TYPES:
                if labels.get(myth_type, "neutral") != "neutral":
                    continue
                t1_vec    = np.array(row["sbert_t1"])
                proj_sub  = float(np.dot(t1_vec, subspace)) if subspace is not None else np.nan
                proj_myth = (
                    float(np.dot(t1_vec, myth_unit_vecs[myth_type]))
                    if myth_type in myth_unit_vecs else np.nan
                )
                neutral_rows.append({
                    "narrative_idx":  idx,
                    "model":          row["model"],
                    "myth_type":      myth_type,
                    "proj_subspace":  proj_sub,
                    "proj_myth_unit": proj_myth,
                })

        neutral_df = pd.DataFrame(neutral_rows)

        for myth in MYTH_TYPES:
            grp_neu = neutral_df[neutral_df["myth_type"] == myth]
            for label in ["entailment", "contradiction"]:
                grp_org = sub_original[
                    (sub_original["myth_type"] == myth) &
                    (sub_original["narrative_nli_label"] == label)
                ]
                if len(grp_org) < 5 or len(grp_neu) < 5:
                    continue
                for metric_name, col in ORIGINAL_METRICS:
                    r = independent_test(
                        grp_org[col].dropna().tolist(),
                        grp_neu[col].dropna().tolist(),
                    )
                    r.update({
                        "myth":           myth,
                        "organic_label":  label,
                        "metric":         metric_name,
                        "prompt_variant": variant,
                        "mean_original":  grp_org[col].mean(),
                        "mean_neutral":   grp_neu[col].mean(),
                    })
                    results.append(r)

    results = apply_bh_correction(results)
    out = OUT_DIR / "Original_MythAlignment.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"  Saved: {out.name}")