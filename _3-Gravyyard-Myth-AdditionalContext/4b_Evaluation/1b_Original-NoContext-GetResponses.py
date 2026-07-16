"""
1b_Original-NoContext-GetResponses.py
====================
Step 1b: Build T1-only DataFrame for the Original condition.

Original-NoContextAppend Experiment  logic: uses narratives that *already* entail or contradict a myth
(i.e., the myth is originally present or absent in the narrative). No T2
is needed — we analyze the T1 response directly.

Each row in the output corresponds to one (narrative, model, myth_type) triple
where the narrative's NLI label for that myth is entailment or contradiction
(never neutral).

Outputs
-------
  ExpOriginal-NoContextAppend_WithEmbeddings.pkl  — full DataFrame including embedding arrays
  ExpOriginal-NoContextAppend_Scores.csv          — scalar columns only (no embedding arrays)

Usage
-----
  python 1b_Original-NoContext-GetResponses.py \\
      --t1   path/to/t1.csv \\
      --nli  path/to/nli.csv \\
      --model gemma \\
      [--task advice] [--full]

Input columns expected
----------------------
  T1:  narrative_idx, model, response_t1
       [optional] prompt_variant
  NLI: narrative_index, myth_category, myth_type, overall_label
"""

import argparse
import pandas as pd
from pathlib import Path

from Config import MYTH_TYPES, DEMOGRAPHIC_FEATURES, NLI_CSV_SAMPLE, get_output_dir
from Embeddings import encode_sbert, get_w2v, get_glove, mean_word_embedding

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--t1",    required=True, help="Path to T1 CSV")
parser.add_argument("--nli",   default=str(NLI_CSV_SAMPLE), help="Path to NLI CSV")
parser.add_argument("--task",  choices=["AdviceGeneration", "Summarization"],
                    help="Task type — affects output subdirectory only")
parser.add_argument("--model", required=True, help="Model name e.g. gemma, llama")
parser.add_argument("--full",  action="store_true")
args = parser.parse_args()

OUT_DIR = get_output_dir(args.task, f"SampleResults")
print(f"Output dir: {OUT_DIR}")

# ── Load NLI ──────────────────────────────────────────────────────────────────
print(f"Loading NLI from {args.nli}")
nli_df   = pd.read_csv(args.nli)
myth_nli = nli_df[nli_df["myth_category"] == "MYTH"]

# Per-narrative NLI labels: {narrative_idx: {myth_type: label}}
nli_labels_per_narrative = (
    myth_nli.groupby("narrative_index")
    .apply(lambda g: dict(zip(g["myth_type"], g["overall_label"])))
    .to_dict()
)

# Demographic features per narrative (binary indicators for entailing demo features)
demo_rows = {}
for narrative_idx, grp in nli_df[nli_df["myth_category"] != "MYTH"].groupby("narrative_index"):
    row = {}
    entailing = grp[grp["overall_label"] == "entailment"]
    for _, r in entailing.iterrows():
        if r["myth_type"] in DEMOGRAPHIC_FEATURES:
            row[r["myth_type"]] = 1
    for feat in DEMOGRAPHIC_FEATURES:
        row.setdefault(feat, 0)
    demo_rows[int(narrative_idx)] = row
demographic_df = pd.DataFrame.from_dict(demo_rows, orient="index")
demographic_df.index.name = "narrative_idx"

# ── Load T1 ───────────────────────────────────────────────────────────────────
print(f"Loading T1 from {args.t1}")
t1_df = pd.read_csv(args.t1)
print(f"  T1 rows: {len(t1_df)}")

# ── Build Original-NoContextAppend Experiment  DataFrame ───────────────────────────────────────────────
print("\nBuilding Original-NoContextAppend Experiment  DataFrame...")
original_rows = []
for _, row in t1_df.iterrows():
    idx    = int(row["narrative_idx"])
    labels = nli_labels_per_narrative.get(idx, {})
    for myth_type in MYTH_TYPES:
        label = labels.get(myth_type, "neutral")
        if label == "neutral":
            continue  # original condition requires entailment or contradiction
        original_rows.append({
            "narrative_idx":       idx,
            "model":               row["model"],
            "myth_type":           myth_type,
            "narrative_nli_label": label,
            "prompt_variant":      row.get("prompt_variant", None),
            "response_t1":         str(row.get("response_t1", "")),
        })

original_df = pd.DataFrame(original_rows)
original_df = original_df.merge(demographic_df.reset_index(), on="narrative_idx", how="left")
print(f"  Original-NoContextAppend Experiment  rows: {len(original_df)}")

# ── Compute embeddings ────────────────────────────────────────────────────────
print("\nComputing embeddings...")
w2v_model   = get_w2v()
glove_model = get_glove()

texts = original_df["response_t1"].tolist()
print("  SBERT (t1)...")
original_df["sbert_t1"]  = list(encode_sbert(texts))
print("  Word2Vec (t1)...")
original_df["w2v_t1"]    = [mean_word_embedding(t, w2v_model)  for t in texts]
print("  GloVe (t1)...")
original_df["glove_t1"]  = [mean_word_embedding(t, glove_model) for t in texts]

# ── Save ──────────────────────────────────────────────────────────────────────
scalar_cols = [
    c for c in original_df.columns
    if not c.endswith("_t1") or c == "response_t1"
]

out_pkl = OUT_DIR / f"Original/{args.model}_WithEmbeddings.pkl"
original_df.to_pickle(out_pkl)
print(f"\nSaved (with embeddings): {out_pkl.name}")

out_csv = OUT_DIR / f"Original/{args.model}_Scores.csv"
original_df[scalar_cols].to_csv(out_csv, index=False)
print(f"Saved (scores only):     {out_csv.name}")