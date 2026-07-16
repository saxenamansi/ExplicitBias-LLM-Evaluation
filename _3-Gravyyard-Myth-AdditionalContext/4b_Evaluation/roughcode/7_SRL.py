"""
7_SRL.py
=========
Step 7: Semantic Role Labeling and psycholinguistic framing features.

SRL (AllenNLP BERT-based):
  For each response, identify sentences where victim/perpetrator tokens
  appear as ARG0 (agent) or ARG1 (patient).
  Metrics:
    victim_agent_ratio      — fraction of victim mentions as ARG0
    victim_patient_ratio    — fraction of victim mentions as ARG1
    perpetrator_agent_ratio — fraction of perpetrator mentions as ARG0

Connotation frames (Rashkin et al. 2016):
  Average writer_perspective, reader_value, reader_effect, reader_mental_state
  over verb tokens found in the response.

Power & agency (Sap et al. 2017):
  Average agency and power scores over verb tokens.

Install AllenNLP before running:
  pip install allennlp allennlp-models

Download lexicons before running (run once):
  cd LexiconDictionaries
  wget https://raw.githubusercontent.com/hrashkin/connotation-frames/master/data/connotation_frames_verb.csv
  wget https://raw.githubusercontent.com/maartensap/power-agency/master/agency_power.csv

Entity detection:
  Victim:      you/your/yourself (advice); victim/survivor/she/he/they (summary)
  Perpetrator: perpetrator/abuser/attacker/him/her/them
  CHECK: review VICTIM_KEYWORDS and PERPETRATOR_KEYWORDS in 0_Config.py
         against your actual model outputs before running.

Usage:
  python 3g_SRL.py --append path/ExpAppend_WithAlignment.pkl
                   --organic path/ExpOrganic_WithAlignment.pkl
                   --task advice
                   --sample_n 360
"""

import argparse
import re
import numpy as np
import pandas as pd
from pathlib import Path

from 0_Config import (
    MYTH_TYPES, LEXICON_DIR, get_output_dir,
    VICTIM_KEYWORDS, PERPETRATOR_KEYWORDS,
)
from 0_Stats import paired_test, independent_test

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--append",   required=True)
parser.add_argument("--organic",  required=True)
parser.add_argument("--task",     default="advice",
                    choices=["advice", "summarization"])
parser.add_argument("--sample_n", type=int, default=360)
args = parser.parse_args()

OUT_DIR = get_output_dir(args.task, "3g_SRL")
print(f"Output dir: {OUT_DIR}")

# ── Load AllenNLP SRL predictor ───────────────────────────────────────────────
print("Loading AllenNLP SRL predictor...")
try:
    from allennlp_models.pretrained import load_predictor
    srl_predictor = load_predictor("structured-prediction-srl-bert", cuda_device=0)
    print("  SRL predictor loaded.")
except Exception as e:
    print(f"  ERROR loading AllenNLP: {e}")
    print("  Install with: pip install allennlp allennlp-models")
    raise

# ── Load connotation frames lexicon ──────────────────────────────────────────
def load_connframes() -> pd.DataFrame:
    path = LEXICON_DIR / "connotation_frames_verb.csv"
    if not path.exists():
        print(f"  WARNING: {path.name} not found. Download with:")
        print("  wget https://raw.githubusercontent.com/hrashkin/connotation-frames/master/data/connotation_frames_verb.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
    verb_col = df.select_dtypes(exclude=[np.number]).columns[0]
    df[verb_col] = df[verb_col].str.lower().str.strip()
    df = df.set_index(verb_col).select_dtypes(include=[np.number])
    # Shorten column names
    rename = {}
    for c in df.columns:
        short = (c.replace("writer.sentiment.on.subject", "writer_persp")
                  .replace("reader.perception.of.subject.", "")
                  .replace("reader.perception.of.object.", "obj_")
                  .replace(".", "_"))
        rename[c] = short
    return df.rename(columns=rename)

# ── Load power & agency lexicon ───────────────────────────────────────────────
def load_power_agency() -> pd.DataFrame:
    path = LEXICON_DIR / "agency_power.csv"
    if not path.exists():
        print(f"  WARNING: {path.name} not found. Download with:")
        print("  wget https://raw.githubusercontent.com/maartensap/power-agency/master/agency_power.csv")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    verb_col = df.select_dtypes(exclude=[np.number]).columns[0]
    df[verb_col] = df[verb_col].str.lower().str.strip()
    return df.set_index(verb_col).select_dtypes(include=[np.number])

print("Loading connotation frames lexicon...")
connframes_df = load_connframes()
print(f"  {len(connframes_df)} verbs loaded" if not connframes_df.empty else "  Skipped.")

print("Loading power & agency lexicon...")
power_df = load_power_agency()
print(f"  {len(power_df)} verbs loaded" if not power_df.empty else "  Skipped.")

# ── SRL helpers ───────────────────────────────────────────────────────────────
def is_victim_token(token: str) -> bool:
    return token.lower() in VICTIM_KEYWORDS

def is_perpetrator_token(token: str) -> bool:
    return token.lower() in PERPETRATOR_KEYWORDS

def srl_framing_scores(text: str) -> dict:
    """
    Run SRL on text. For each predicate-argument structure, check whether
    victim/perpetrator tokens appear as ARG0 (agent) or ARG1 (patient).
    Returns ratio scores.
    CHECK: if responses are very long, consider sentence-level SRL instead of
           full-text SRL to avoid memory issues.
    """
    victim_arg0 = victim_arg1 = victim_total = 0
    perp_arg0   = perp_arg1   = perp_total   = 0

    try:
        result = srl_predictor.predict(sentence=str(text))
    except Exception:
        return {
            "victim_agent_ratio":      np.nan,
            "victim_patient_ratio":    np.nan,
            "perpetrator_agent_ratio": np.nan,
        }

    for verb_entry in result.get("verbs", []):
        tags   = verb_entry.get("tags", [])
        words  = result.get("words", [])
        # Group tokens by ARG label
        arg_spans: dict[str, list] = {}
        current_label = "O"
        current_span  = []
        for word, tag in zip(words, tags):
            if tag.startswith("B-"):
                if current_label != "O" and current_span:
                    arg_spans.setdefault(current_label, []).append(" ".join(current_span))
                current_label = tag[2:]
                current_span  = [word]
            elif tag.startswith("I-"):
                current_span.append(word)
            else:
                if current_label != "O" and current_span:
                    arg_spans.setdefault(current_label, []).append(" ".join(current_span))
                current_label = "O"
                current_span  = []

        for arg_label, spans in arg_spans.items():
            for span in spans:
                span_tokens = span.lower().split()
                has_victim = any(is_victim_token(t) for t in span_tokens)
                has_perp   = any(is_perpetrator_token(t) for t in span_tokens)
                if has_victim:
                    victim_total += 1
                    if arg_label == "ARG0":
                        victim_arg0 += 1
                    elif arg_label == "ARG1":
                        victim_arg1 += 1
                if has_perp:
                    perp_total += 1
                    if arg_label == "ARG0":
                        perp_arg0 += 1

    return {
        "victim_agent_ratio":      victim_arg0 / victim_total if victim_total > 0 else np.nan,
        "victim_patient_ratio":    victim_arg1 / victim_total if victim_total > 0 else np.nan,
        "perpetrator_agent_ratio": perp_arg0   / perp_total   if perp_total   > 0 else np.nan,
    }

def verb_lexicon_scores(text: str) -> dict:
    """
    Average connotation frame and power/agency scores over verb tokens in text.
    """
    words  = str(text).lower().split()
    result = {}

    if not connframes_df.empty:
        scores = {col: [] for col in connframes_df.columns}
        for w in words:
            if w in connframes_df.index:
                for col in connframes_df.columns:
                    scores[col].append(float(connframes_df.at[w, col]))
        for col, vals in scores.items():
            result[f"connframe_{col}"] = float(np.mean(vals)) if vals else np.nan

    if not power_df.empty:
        scores = {col: [] for col in power_df.columns}
        for w in words:
            if w in power_df.index:
                for col in power_df.columns:
                    scores[col].append(float(power_df.at[w, col]))
        for col, vals in scores.items():
            result[f"power_agency_{col}"] = float(np.mean(vals)) if vals else np.nan

    return result

# ── Load inputs ───────────────────────────────────────────────────────────────
print(f"\nLoading Exp-Append from {args.append}")
append_df = pd.read_pickle(args.append)

print(f"Loading Exp-Organic from {args.organic}")
organic_df = pd.read_pickle(args.organic)

# Sample for SRL (expensive)
sample_append = append_df[append_df["myth_type"].notna()].sample(
    min(args.sample_n, len(append_df)), random_state=42
)
print(f"Running SRL on {len(sample_append)} Exp-Append rows...")

# ── Score Exp-Append ──────────────────────────────────────────────────────────
append_rows = []
for i, (_, row) in enumerate(sample_append.iterrows()):
    if i % 50 == 0:
        print(f"  {i}/{len(sample_append)}")

    t1_srl  = srl_framing_scores(row["response_t1"])
    t2_srl  = srl_framing_scores(row["response_t2"])
    t1_verb = verb_lexicon_scores(row["response_t1"])
    t2_verb = verb_lexicon_scores(row["response_t2"])

    append_rows.append({
        "narrative_idx":  row["narrative_idx"],
        "model":          row["model"],
        "myth_type":      row["myth_type"],
        "frame":          row["frame"],
        "dose":           row["dose"],
        "prompt_variant": row.get("prompt_variant"),
        **{f"t1_{k}": v for k, v in t1_srl.items()},
        **{f"t2_{k}": v for k, v in t2_srl.items()},
        **{f"t1_{k}": v for k, v in t1_verb.items()},
        **{f"t2_{k}": v for k, v in t2_verb.items()},
    })

srl_append_df = pd.DataFrame(append_rows)
out = OUT_DIR / "ExpAppend_SRL_ConnFrames_PowerAgency_T1andT2.csv"
srl_append_df.to_csv(out, index=False)
print(f"  Saved: {out.name}")

# ── Score Exp-Organic ─────────────────────────────────────────────────────────
print(f"\nRunning SRL on {min(args.sample_n, len(organic_df))} Exp-Organic rows...")
sample_organic = organic_df.sample(min(args.sample_n, len(organic_df)), random_state=42)
organic_rows   = []
for i, (_, row) in enumerate(sample_organic.iterrows()):
    if i % 50 == 0:
        print(f"  {i}/{len(sample_organic)}")
    t1_srl  = srl_framing_scores(row["response_t1"])
    t1_verb = verb_lexicon_scores(row["response_t1"])
    organic_rows.append({
        "narrative_idx":       row["narrative_idx"],
        "model":               row["model"],
        "myth_type":           row["myth_type"],
        "narrative_nli_label": row["narrative_nli_label"],
        "prompt_variant":      row.get("prompt_variant"),
        **t1_srl, **t1_verb,
    })

srl_organic_df = pd.DataFrame(organic_rows)
out = OUT_DIR / "ExpOrganic_SRL_ConnFrames_PowerAgency_T1.csv"
srl_organic_df.to_csv(out, index=False)
print(f"  Saved: {out.name}")

# ── Paired T1 vs T2 SRL shift tests ──────────────────────────────────────────
srl_metrics = [
    "victim_agent_ratio", "victim_patient_ratio", "perpetrator_agent_ratio"
]
stat_rows = []
for myth in MYTH_TYPES:
    sub = srl_append_df[srl_append_df["myth_type"] == myth]
    for metric in srl_metrics:
        t1_col, t2_col = f"t1_{metric}", f"t2_{metric}"
        if t1_col not in sub.columns or t2_col not in sub.columns:
            continue
        vals = sub[[t1_col, t2_col]].dropna()
        if len(vals) < 5:
            continue
        result = paired_test(vals[t1_col].tolist(), vals[t2_col].tolist())
        result.update({"myth": myth, "metric": metric})
        stat_rows.append(result)

if stat_rows:
    out = OUT_DIR / "ExpAppend_SRL_PairedTests_T1vsT2.csv"
    pd.DataFrame(stat_rows).to_csv(out, index=False)
    print(f"  Saved: {out.name}")

print("\nStep 7 complete.")