"""
1a_ContextAppend-GetResponses.py
======================
Step 1a: Build T1/T2 paired responses DataFrame for context append experiment "ContextAppend".

Context-Append logic: narratives must be neutral, not entailing or contradicting, for additional myth context to be appended
Each row pairs the original T1 response with the revised myth-conditioned T2 response for the same narrative

Outputs
-------
  ContextAppend_WithEmbeddings.pkl   — full DataFrame including embedding arrays
  ContextAppend_Scores.csv           — scalar columns only (no embedding arrays)

Usage
-----
  python 1a_ContextAppend-GetResponses.py \\
      --t1  path/to/t1.csv \\
      --t2  path/to/t2.csv \\
      --nli path/to/nli.csv \\
      --model gemma \\
      [--task advice] [--full]

Input columns expected
----------------------
  T1: narrative_idx, model, response_t1
      [optional] prompt_variant
  T2: narrative_idx, model, myth_type, myth_pair, frame, dose, response_t2
      [optional] prompt_variant
  NLI: narrative_index, myth_category, myth_type, overall_label
"""

import argparse
import pandas as pd
from pathlib import Path

from Config import MYTH_TYPES, MYTH_PAIRS, DEMOGRAPHIC_FEATURES, NLI_CSV_SAMPLE, get_output_dir
from Embeddings import encode_sbert, get_w2v, get_glove, mean_word_embedding, cosine_sim

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--t1", required=True, help="Path to T1 CSV")
parser.add_argument("--t2",    required=True, help="Path to T2 CSV")
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

# ── Load T1 / T2 ──────────────────────────────────────────────────────────────
print(f"Loading T1 from {args.t1}")
t1_df = pd.read_csv(args.t1)
print(f"  T1 rows: {len(t1_df)}")

print(f"Loading T2 from {args.t2}")
t2_df = pd.read_csv(args.t2, on_bad_lines="skip", engine="python")
print(f"  T2 rows: {len(t2_df)}")

# ── Neutrality checker ────────────────────────────────────────────────────────
def is_neutral(narrative_idx: int, myth_type: str) -> bool:
    return nli_labels_per_narrative.get(narrative_idx, {}).get(myth_type, "neutral") == "neutral"

# ── Build paired DataFrame ────────────────────────────────────────────────────
print("\nBuilding Context-Append paired DataFrame...")

has_variant = "prompt_variant" in t1_df.columns
if has_variant:
    t1_index = t1_df.set_index(["narrative_idx", "model", "prompt_variant"])
else:
    t1_index = t1_df.set_index(["narrative_idx", "model"])

paired_rows = []
for _, r2 in t2_df.iterrows():
    idx       = int(r2["narrative_idx"])
    model     = r2["model"]
    myth_type = r2.get("myth_type")
    myth_pair = r2.get("myth_pair")
    variant   = r2.get("prompt_variant", None)

    # Neutrality check — skip narratives that already entail/contradict the myth
    if pd.notna(myth_type):
        if not is_neutral(idx, myth_type):
            continue
    elif pd.notna(myth_pair):
        if not all(is_neutral(idx, m) for m in str(myth_pair).split("+")):
            continue

    # Look up matching T1 row
    try:
        if has_variant and variant is not None:
            t1_row = t1_index.loc[(idx, model, variant)]
        else:
            t1_row = t1_index.loc[(idx, model)]
        if isinstance(t1_row, pd.DataFrame):
            t1_row = t1_row.iloc[0]
    except KeyError:
        continue

    paired_rows.append({
        "narrative_idx":  idx,
        "model":          model,
        "myth_type":      myth_type if pd.notna(myth_type) else None,
        "myth_pair":      myth_pair if pd.notna(myth_pair) else None,
        "frame":          r2.get("frame"),
        "dose":           r2.get("dose"),
        "is_pair":        pd.notna(myth_pair),
        "prompt_variant": variant,
        "response_t1":    str(t1_row.get("response_t1", "")),
        "response_t2":    str(r2.get("response_t2", "")),
    })

append_df = pd.DataFrame(paired_rows)
append_df = append_df.merge(demographic_df.reset_index(), on="narrative_idx", how="left")
print(f"  Context-Append paired rows: {len(append_df)}")

# ── Compute embeddings ────────────────────────────────────────────────────────
print("\nComputing embeddings...")
w2v_model   = get_w2v()
glove_model = get_glove()

for col, suffix in [("response_t1", "t1"), ("response_t2", "t2")]:
    texts = append_df[col].tolist()
    print(f"  SBERT ({suffix})...")
    append_df[f"sbert_{suffix}"]  = list(encode_sbert(texts))
    print(f"  Word2Vec ({suffix})...")
    append_df[f"w2v_{suffix}"]    = [mean_word_embedding(t, w2v_model)  for t in texts]
    print(f"  GloVe ({suffix})...")
    append_df[f"glove_{suffix}"]  = [mean_word_embedding(t, glove_model) for t in texts]

# Cosine similarity and distance between T1 and T2
for emb in ["sbert", "w2v", "glove"]:
    append_df[f"{emb}_cosine_t1_t2"] = [
        cosine_sim(r[f"{emb}_t1"], r[f"{emb}_t2"])
        for _, r in append_df.iterrows()
    ]
    append_df[f"{emb}_cosine_distance_t1_t2"] = 1 - append_df[f"{emb}_cosine_t1_t2"]

# ── Save ──────────────────────────────────────────────────────────────────────
embedding_suffixes = ["_t1", "_t2"]
scalar_cols = [
    c for c in append_df.columns
    if not any(c.endswith(s) for s in embedding_suffixes)
    or c in [
        "response_t1", "response_t2",
        "sbert_cosine_t1_t2",    "sbert_cosine_distance_t1_t2",
        "w2v_cosine_t1_t2",      "w2v_cosine_distance_t1_t2",
        "glove_cosine_t1_t2",    "glove_cosine_distance_t1_t2",
    ]
]

out_pkl = OUT_DIR / f"ContextAppend/{args.model}_WithEmbeddings.pkl"
append_df.to_pickle(out_pkl)
print(f"\nSaved (with embeddings): {out_pkl.name}")

out_csv = OUT_DIR / f"ContextAppend/{args.model}_Scores.csv"
append_df[scalar_cols].to_csv(out_csv, index=False)
print(f"Saved (scores only):     {out_csv.name}")