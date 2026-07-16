"""
2b_Projection.py
================
Step 2b: Compute myth-alignment projection scores for ContextAppend and
Original-NoContextAppend experiments.

Metrics computed
----------------
  ContextAppend:
    proj_t1_subspace     — absolute projection of T1 onto myth-alignment subspace
    proj_t2_subspace     — absolute projection of T2 onto myth-alignment subspace
    delta_magnitude      — L2 norm of shift vector (T2 - T1), independent of myth direction
    proj_myth_magnitude  — signed magnitude of shift in myth direction:
                           dot(delta, myth_unit_vec) = ||delta|| * cos(angle)
                           sensitive to both direction and shift size
    proj_myth_direction  — cosine similarity between shift direction and myth vector:
                           dot(delta_norm, myth_unit_vec), bounded [-1, 1]
                           pure direction signal, magnitude-agnostic

  Original-NoContextAppend:
    proj_subspace        — absolute projection of T1 onto myth-alignment subspace
    proj_myth_unit       — dot product of T1 with per-myth unit vector

Reads pkl output from 1a / 1b (must contain sbert embedding arrays).

Outputs
-------
  ContextAppend_WithProjections.pkl
  ContextAppend_ProjectionScores.csv
  ExpOriginal-NoContextAppend_WithProjections.pkl
  ExpOriginal-NoContextAppend_ProjectionScores.csv

Output path: {task}/2_Projection/SampleResults/{model}/

Usage
-----
  # Both experiments
  python 2b_Projection.py \\
      --ContextAppend path/to/ContextAppend_WithEmbeddings.pkl \\
      --Original      path/to/ExpOriginal-NoContextAppend_WithEmbeddings.pkl \\
      --model gemma [--task advice] [--full]

  # ContextAppend only
  python 2b_Projection.py --ContextAppend path/to/ContextAppend_WithEmbeddings.pkl --model gemma

  # Original only
  python 2b_Projection.py --Original path/to/ExpOriginal-NoContextAppend_WithEmbeddings.pkl --model gemma
"""

import argparse
import numpy as np
import pandas as pd

from Config import MYTH_TYPES, MYTH_PAIRS, get_output_dir
from Embeddings import get_subspace, get_myth_unit_vecs

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ContextAppend",   default=None, help="Path to ContextAppend_WithEmbeddings.pkl")
parser.add_argument("--Original", default=None, help="Path to ExpOriginal-NoContextAppend_WithEmbeddings.pkl")
parser.add_argument("--task",     choices=["AdviceGeneration", "Summarization"])
parser.add_argument("--model",    required=True, help="Model name e.g. gemma, llama")
parser.add_argument("--full",     action="store_true")
args = parser.parse_args()

if not args.ContextAppend and not args.Original:
    raise ValueError("Provide at least one of --ContextAppend or --Original.")

OUT_DIR = get_output_dir(args.task, f"2_Projection/SampleResults/")
print(f"Output dir: {OUT_DIR}")

subspace       = get_subspace()
myth_unit_vecs = get_myth_unit_vecs()

SCALAR_COLS_APPEND = [
    "narrative_idx", "model", "myth_type", "myth_pair", "frame", "dose",
    "is_pair", "prompt_variant",
    "proj_t1_subspace", "proj_t2_subspace",
    "delta_magnitude", "proj_myth_magnitude", "proj_myth_direction",
    "sbert_cosine_t1_t2",  "sbert_cosine_distance_t1_t2",
    "w2v_cosine_t1_t2",    "w2v_cosine_distance_t1_t2",
    "glove_cosine_t1_t2",  "glove_cosine_distance_t1_t2",
]

SCALAR_COLS_ORIGINAL = [
    "narrative_idx", "model", "myth_type", "narrative_nli_label",
    "prompt_variant", "proj_subspace", "proj_myth_unit",
]


# ── Projection helpers ────────────────────────────────────────────────────────

def project_shift(row) -> dict:
    
    t1_vec = np.array(row["sbert_t1"])
    t2_vec = np.array(row["sbert_t2"])
    delta  = t2_vec - t1_vec

    if subspace is not None:
        proj_t1 = float(np.dot(t1_vec, subspace))
        proj_t2 = float(np.dot(t2_vec, subspace))
    else:
        proj_t1, proj_t2 = np.nan, np.nan

    myth_type       = row.get("myth_type")
    delta_mag       = float(np.linalg.norm(delta))
    delta_norm      = delta / (delta_mag + 1e-10)

    if myth_type and myth_type in myth_unit_vecs:
        proj_myth_magnitude  = float(np.dot(delta,      myth_unit_vecs[myth_type]))
        proj_myth_direction  = float(np.dot(delta_norm, myth_unit_vecs[myth_type]))
    else:
        proj_myth_magnitude  = np.nan
        proj_myth_direction  = np.nan

    return {
        "proj_t1_subspace":     proj_t1,
        "proj_t2_subspace":     proj_t2,
        "delta_magnitude":      delta_mag,
        "proj_myth_magnitude":  proj_myth_magnitude,
        "proj_myth_direction":  proj_myth_direction,
    }

def project_t1(row) -> dict:
    """
    Compute projection metrics for a T1-only row (Original experiment).
    All projections use SBERT embeddings.
    """
    t1_vec    = np.array(row["sbert_t1"])
    myth_type = row.get("myth_type")

    proj_sub  = float(np.dot(t1_vec, subspace)) if subspace is not None else np.nan
    proj_myth = (
        float(np.dot(t1_vec, myth_unit_vecs[myth_type]))
        if myth_type in myth_unit_vecs else np.nan
    )
    return {
        "proj_subspace":  proj_sub,
        "proj_myth_unit": proj_myth,
    }


# ── ContextAppend ─────────────────────────────────────────────────────────────
if args.ContextAppend:
    print(f"\nLoading ContextAppend from {args.ContextAppend}")
    append_df = pd.read_pickle(args.ContextAppend)
    print(f"  Rows: {len(append_df)}")

    print("  Computing shift projections...")
    proj_cols = pd.DataFrame(
        [project_shift(row) for _, row in append_df.iterrows()]
    )
    append_df = pd.concat(
        [append_df.reset_index(drop=True), proj_cols.reset_index(drop=True)], axis=1
    )

    out_pkl = OUT_DIR / f"{args.model}_ContextAppend_Projections.pkl"
    append_df.to_pickle(out_pkl)
    print(f"  Saved (with projections): {out_pkl.name}")

    out_csv = OUT_DIR / f"{args.model}_ContextAppend_ProjectionsScores.csv"
    append_df[[c for c in SCALAR_COLS_APPEND if c in append_df.columns]].to_csv(out_csv, index=False)
    print(f"  Saved (scores only):      {out_csv.name}")


# ── Original-NoContextAppend ──────────────────────────────────────────────────
if args.Original:
    print(f"\nLoading Original-NoContextAppend from {args.Original}")
    original_df = pd.read_pickle(args.Original)
    print(f"  Rows: {len(original_df)}")

    print("  Computing T1 projections...")
    proj_cols = pd.DataFrame(
        [project_t1(row) for _, row in original_df.iterrows()]
    )
    original_df = pd.concat(
        [original_df.reset_index(drop=True), proj_cols.reset_index(drop=True)], axis=1
    )

    out_pkl = OUT_DIR / f"{args.model}_Original_Projections.pkl"
    original_df.to_pickle(out_pkl)
    print(f"  Saved (with projections): {out_pkl.name}")

    out_csv = OUT_DIR / f"{args.model}_Original_ProjectionsScores.csv"
    original_df[[c for c in SCALAR_COLS_ORIGINAL if c in original_df.columns]].to_csv(out_csv, index=False)
    print(f"  Saved (scores only):      {out_csv.name}")