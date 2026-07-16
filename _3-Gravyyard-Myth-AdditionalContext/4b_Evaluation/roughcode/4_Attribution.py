"""
4_Attribution.py
=================
Step 4: Sentence-level attribution — which sentences in T2 responses
drive myth-aligned semantic shift.

Methods (convergent validity):
  cosine_to_delta    — cosine similarity of sentence to shift vector (T2 - T1)
  cosine_to_myth     — cosine similarity of sentence to myth unit vector
  proj_subspace      — projection of sentence onto myth-alignment subspace
  max_nli_score      — max NLI entailment score against ATTR_NLI hypotheses
  n_methods_flagged  — how many of the above exceed threshold (0-4)

Runs on Exp-Append data only (T1/T2 pairs needed).
Reads pkl output from 2_MythAlignment.py.

Usage:
  python 3d_Attribution.py --append path/ExpAppend_WithAlignment.pkl
                            --task advice
                            --sample_n 360
"""

import argparse
import re
import numpy as np
import pandas as pd
from transformers import pipeline as hf_pipeline

from Config import MYTH_TYPES, ATTR_NLI, get_output_dir
from Embeddings import encode_sbert, get_subspace, get_myth_unit_vecs, cosine_sim

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--append",   required=True, help="ExpAppend pkl from 2")
parser.add_argument("--task",     choices=["advice", "summarization"])
parser.add_argument("--sample_n", type=int, default=360,
                    help="Max narratives to run attribution on (expensive)")
args = parser.parse_args()

OUT_DIR = get_output_dir(args.task, "4_Attribution")
print(f"Output dir: {OUT_DIR}")

subspace       = get_subspace()
myth_unit_vecs = get_myth_unit_vecs()

# ── Load NLI pipeline ─────────────────────────────────────────────────────────
print("Loading NLI pipeline (DeBERTa zero-shot)...")
nli_pipe = hf_pipeline(
    "zero-shot-classification",
    model="MoritzLaurer/deberta-v3-base-zeroshot-v1",
    device=0,
)

def nli_entailment_score(text: str, hypothesis: str) -> float:
    try:
        result = nli_pipe(str(text), candidate_labels=[hypothesis], multi_label=False)
        return result["scores"][0]
    except Exception:
        return 0.0

# ── Sentence splitter ─────────────────────────────────────────────────────────
def split_sentences(text: str) -> list:
    parts = re.split(r"\n|\d+\.\s+|<0x0A>", str(text))
    return [p.strip() for p in parts if len(p.strip()) > 10]

# ── Load input ────────────────────────────────────────────────────────────────
print(f"Loading Exp-Append from {args.append}")
append_df = pd.read_pickle(args.append)
print(f"  Rows: {len(append_df)}")

# Sample to keep runtime manageable
# CHECK: increase sample_n if you have more GPU time
sample = append_df[append_df["myth_type"].notna()].sample(
    min(args.sample_n, append_df["myth_type"].notna().sum()),
    random_state=42,
)
print(f"  Running attribution on {len(sample)} rows")

# ── Attribution ───────────────────────────────────────────────────────────────
attr_rows = []
for _, row in sample.iterrows():
    myth_type = row["myth_type"]
    if not isinstance(myth_type, str):
        continue

    t1_vec     = np.array(row["sbert_t1"])
    t2_vec     = np.array(row["sbert_t2"])
    delta      = t2_vec - t1_vec
    delta_norm = delta / (np.linalg.norm(delta) + 1e-10)
    myth_unit  = myth_unit_vecs.get(myth_type)
    hypotheses = ATTR_NLI.get(myth_type, [])

    for sent in split_sentences(row["response_t2"]):
        sent_vec  = encode_sbert([sent])[0]
        cos_delta = cosine_sim(sent_vec, delta_norm)
        cos_myth  = cosine_sim(sent_vec, myth_unit) if myth_unit is not None else np.nan
        proj_sent = float(np.dot(sent_vec, subspace)) if subspace is not None else np.nan
        nli_scores = {hyp: nli_entailment_score(sent, hyp) for hyp in hypotheses}
        max_nli    = max(nli_scores.values()) if nli_scores else 0.0

        attr_rows.append({
            "narrative_idx":       row["narrative_idx"],
            "model":               row["model"],
            "myth_type":           myth_type,
            "frame":               row["frame"],
            "dose":                row["dose"],
            "prompt_variant":      row.get("prompt_variant"),
            "sentence":            sent,
            "cosine_to_delta":     cos_delta,
            "cosine_to_myth_unit": cos_myth,
            "proj_subspace":       proj_sent,
            "max_nli_score":       max_nli,
            "nli_scores":          str(nli_scores),
            "n_methods_flagged":   sum([
                cos_delta > 0.3,
                (cos_myth  > 0.3 if not np.isnan(cos_myth)  else False),
                (proj_sent > 0.0 if not np.isnan(proj_sent) else False),
                max_nli   > 0.5,
            ]),
        })

attr_df = pd.DataFrame(attr_rows)
out = OUT_DIR / "SentenceAttribution_ConvergentValidity.csv"
attr_df.to_csv(out, index=False)
print(f"  Saved: {out.name}")

print("\nStep 4 complete.")