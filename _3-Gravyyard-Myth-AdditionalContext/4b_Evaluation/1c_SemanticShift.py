"""
1c_SemanticShift.py
===============
Run statistical tests to check semantic shift (ContextAppend) and original T1 distributions (Original).

ContextAppend:  paired test — cosine similarity of T1 vs T2 responses,
             compared against baseline of 1.0 (no shift).
             Grouped by prompt_variant when present (Advice); single group for Summarization.

Original: independent test — T1 embedding scores for entailing
             vs contradicting narratives, per myth type.
             Grouped by prompt_variant when present.

Uses StatisticalTests.py — normality is checked per-group and the appropriate test is selected automatically.

Outputs
-------
  {model}_Append_Embeddings.csv   (if --ContextAppend provided)
  {model}_Original_Embeddings.csv (if --Original provided)

Usage
-----
  # Both conditions
  python 1c_SemanticShift.py \\
      --ContextAppend  path/to/ContextAppend_WithEmbeddings.pkl \\
      --Original path/to/Original_WithEmbeddings.pkl \\
      --model gemma [--task AdviceGeneration]

  # ContextAppend only
  python 1c_SemanticShift.py --ContextAppend path/to/ContextAppend_WithEmbeddings.pkl --model gemma

  # Original only
  python 1c_SemanticShift.py --Original path/to/Original_WithEmbeddings.pkl --model gemma
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from Config import MYTH_TYPES, MYTH_PAIRS, get_output_dir
from StatisticalTests import paired_test, independent_test, apply_bh_correction

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ContextAppend", default=None, help="Path to ContextAppend_WithEmbeddings.pkl")
parser.add_argument("--Original",      default=None, help="Path to Original_WithEmbeddings.pkl")
parser.add_argument("--task",    choices=["AdviceGeneration", "Summarization"])
parser.add_argument("--model",   required=True, help="Model name e.g. gemma, llama")
args = parser.parse_args()

if not args.ContextAppend and not args.Original:
    raise ValueError("Provide at least one of --ContextAppend or --Original.")

OUT_DIR = get_output_dir(args.task, "1_SemanticShift/SampleResults/")
print(f"Output dir: {OUT_DIR}")

EMBEDDINGS = ["sbert", "w2v", "glove"]


def get_variants(df: pd.DataFrame) -> list:
    """
    Returns list of prompt_variant values if column exists and has values,
    otherwise [None] — collapses to single group (Summarization).
    """
    if "prompt_variant" in df.columns and df["prompt_variant"].notna().any():
        return sorted(df["prompt_variant"].dropna().unique().tolist())
    return [None]


# ── ContextAppend: paired test (T1 cosine similarity vs baseline of 1.0) ─────
if args.ContextAppend:
    print(f"\nLoading ContextAppend from {args.ContextAppend}")
    context_append_df = pd.read_pickle(args.ContextAppend)
    print(f"  Rows: {len(context_append_df)}")

    variants = get_variants(context_append_df)
    print(f"  Prompt variants: {variants}")

    results = []
    for emb in EMBEDDINGS:
        sim_col  = f"{emb}_cosine_t1_t2"
        dist_col = f"{emb}_cosine_distance_t1_t2"

        for myth in MYTH_TYPES + MYTH_PAIRS:
            col_filter = "myth_pair" if "+" in myth else "myth_type"

            for variant in variants:
                if variant is not None:
                    sub = context_append_df[
                        (context_append_df[col_filter] == myth) &
                        (context_append_df["prompt_variant"] == variant)
                    ]
                else:
                    sub = context_append_df[context_append_df[col_filter] == myth]

                if len(sub) < 5:
                    continue

                sims     = sub[sim_col].fillna(0).tolist()
                baseline = [1.0] * len(sims)
                result   = paired_test(sims, baseline)
                result.update({
                    "embedding":              emb,
                    "myth":                   myth,
                    "is_pair":                "+" in myth,
                    "prompt_variant":         variant,
                    "n":                      len(sub),
                    "mean_cosine_distance":   sub[dist_col].mean(),
                    "median_cosine_distance": sub[dist_col].median(),
                })
                results.append(result)

    results = apply_bh_correction(results)
    out = OUT_DIR / f"{args.model}_Append_Embeddings.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"  Saved: {out.name}")


# ── Original: independent test (entailment vs contradiction per myth) ─────────
if args.Original:
    print(f"\nLoading Original from {args.Original}")
    original_df = pd.read_pickle(args.Original)
    print(f"  Rows: {len(original_df)}")

    variants = get_variants(original_df)
    print(f"  Prompt variants: {variants}")

    results = []
    for emb in EMBEDDINGS:
        emb_col = f"{emb}_t1"
        if emb_col not in original_df.columns:
            print(f"  Skipping {emb}: column {emb_col} not found")
            continue

        for myth_type in MYTH_TYPES:
            for variant in variants:
                if variant is not None:
                    sub = original_df[
                        (original_df["myth_type"] == myth_type) &
                        (original_df["prompt_variant"] == variant)
                    ]
                else:
                    sub = original_df[original_df["myth_type"] == myth_type]

                entailing     = sub[sub["narrative_nli_label"] == "entailment"]
                contradicting = sub[sub["narrative_nli_label"] == "contradiction"]

                if len(entailing) < 5 or len(contradicting) < 5:
                    continue

                ent_scores = [np.linalg.norm(v) for v in entailing[emb_col]]
                con_scores = [np.linalg.norm(v) for v in contradicting[emb_col]]

                result = independent_test(ent_scores, con_scores)
                result.update({
                    "embedding":      emb,
                    "myth_type":      myth_type,
                    "prompt_variant": variant,
                    "n_entail":       len(entailing),
                    "n_contradict":   len(contradicting),
                })
                results.append(result)

    results = apply_bh_correction(results)
    out = OUT_DIR / f"{args.model}_Original_Embeddings.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"  Saved: {out.name}")