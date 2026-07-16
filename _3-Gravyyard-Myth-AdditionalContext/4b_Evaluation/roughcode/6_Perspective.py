"""
6_Perspective.py
=================
Step 6: Perspective shift detection — second vs third person.

Metrics:
  pct_second_person — fraction of sentences containing you/your/yourself
  pct_third_person  — fraction of sentences containing they/them/the victim/etc.

Runs on both T1 and T2 responses (Exp-Append) and T1 only (Exp-Organic).
Reads pkl output from 3b_MythAlignment.py.

Usage:
  python 3f_Perspective.py --append path/ExpAppend_WithAlignment.pkl
                            --organic path/ExpOrganic_WithAlignment.pkl
                            --task advice
"""

import argparse
import re
import numpy as np
import pandas as pd

from 0_Config import MYTH_TYPES, get_output_dir
from 0_Stats import paired_test

SECOND_PERSON_RE = re.compile(r"\b(you|your|yourself)\b", re.IGNORECASE)
THIRD_PERSON_RE  = re.compile(
    r"\b(they|them|their|the victim|the person|the survivor|he|she|his|her)\b",
    re.IGNORECASE,
)

def perspective_score(text: str) -> dict:
    sentences = re.split(r"[.!?\n]", str(text))
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
    if not sentences:
        return {"pct_second_person": np.nan, "pct_third_person": np.nan}
    n = len(sentences)
    return {
        "pct_second_person": sum(1 for s in sentences if SECOND_PERSON_RE.search(s)) / n,
        "pct_third_person":  sum(1 for s in sentences if THIRD_PERSON_RE.search(s)) / n,
    }

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--append",  required=True)
parser.add_argument("--organic", required=True)
parser.add_argument("--task",    default="advice",
                    choices=["advice", "summarization"])
args = parser.parse_args()

OUT_DIR = get_output_dir(args.task, "3f_Perspective")
print(f"Output dir: {OUT_DIR}")

# ── Load inputs ───────────────────────────────────────────────────────────────
print(f"Loading Exp-Append from {args.append}")
append_df = pd.read_pickle(args.append)

print(f"Loading Exp-Organic from {args.organic}")
organic_df = pd.read_pickle(args.organic)

# ── Score perspective ─────────────────────────────────────────────────────────
print("\nScoring perspective (Exp-Append)...")
for col in ["response_t1", "response_t2"]:
    scores = append_df[col].apply(perspective_score)
    append_df[f"pct_second_{col[-2:]}"] = scores.apply(lambda x: x["pct_second_person"])
    append_df[f"pct_third_{col[-2:]}"]  = scores.apply(lambda x: x["pct_third_person"])

print("Scoring perspective (Exp-Organic)...")
org_scores = organic_df["response_t1"].apply(perspective_score)
organic_df["pct_second_person"] = org_scores.apply(lambda x: x["pct_second_person"])
organic_df["pct_third_person"]  = org_scores.apply(lambda x: x["pct_third_person"])

# ── Save scored DataFrames ────────────────────────────────────────────────────
scalar_cols = [
    "narrative_idx", "model", "myth_type", "myth_pair", "frame", "dose",
    "is_pair", "prompt_variant",
    "pct_second_t1", "pct_third_t1", "pct_second_t2", "pct_third_t2",
]
out = OUT_DIR / "ExpAppend_PerspectiveScores.csv"
append_df[[c for c in scalar_cols if c in append_df.columns]].to_csv(out, index=False)
print(f"  Saved: {out.name}")

out = OUT_DIR / "ExpOrganic_PerspectiveScores.csv"
organic_df[["narrative_idx", "model", "myth_type", "narrative_nli_label",
             "prompt_variant", "pct_second_person", "pct_third_person"]
           ].to_csv(out, index=False)
print(f"  Saved: {out.name}")

# ── Paired T1 vs T2 perspective shift tests ───────────────────────────────────
print("\nRunning paired perspective shift tests...")
stat_rows = []
for myth in MYTH_TYPES:
    sub = append_df[append_df["myth_type"] == myth]
    for metric in ["pct_second", "pct_third"]:
        t1_col = f"{metric}_t1"
        t2_col = f"{metric}_t2"
        if t1_col not in sub.columns or t2_col not in sub.columns:
            continue
        vals = sub[[t1_col, t2_col]].dropna()
        if len(vals) < 5:
            continue
        result = paired_test(vals[t1_col].tolist(), vals[t2_col].tolist())
        result.update({"myth": myth, "metric": metric})
        stat_rows.append(result)

if stat_rows:
    out = OUT_DIR / "ExpAppend_PerspectiveShift_StatTests.csv"
    pd.DataFrame(stat_rows).to_csv(out, index=False)
    print(f"  Saved: {out.name}")

print("\nStep 6 complete.")