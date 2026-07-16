"""
5_Affect.py
============
Step 5: Affect, emotion, moral, and psycholinguistic feature analysis.

Lexicons:
  Existing: ANEW, NRC-EIL, NRC-VAD, WWBP-Affect, WWBP-Empathy,
            MPQA, MFD, Prosocial, VADER
  New (Xi & Singh 2024):
    Connotation frames (Rashkin et al. 2016) — writer perspective,
      reader value/effect/mental-state per verb
    Power & agency (Sap et al. 2017) — agency/power per verb
    Hedge words (Hyland 2018) — hardcoded
    Modal words — hardcoded

Analyses:
  - Score all T1 and T2 responses
  - VIF multicollinearity check
  - Correlation matrix
  - Paired T1 vs T2 tests per myth per feature

Reads pkl output from 2_MythAlignment.py.
Images saved to ResultAnalysis/{task}/5_Affect/figures/

Usage:
  python 5_Affect.py --append path/ExpAppend_WithAlignment.pkl
                      --organic path/ExpOrganic_WithAlignment.pkl
                      --task advice
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from 0_Config import MYTH_TYPES, AFFECT_PREFIXES, DEMOGRAPHIC_FEATURES, get_output_dir
from 0_LexiconLoaders import score_affect
from 0_Stats import compute_vif, paired_test

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--append",  required=True)
parser.add_argument("--organic", required=True)
parser.add_argument("--task",    choices=["advice", "summarization"])
args = parser.parse_args()

OUT_DIR     = get_output_dir(args.task, "5_Affect")
FIG_DIR     = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output dir: {OUT_DIR}")

# ── Load inputs ───────────────────────────────────────────────────────────────
print(f"Loading Exp-Append from {args.append}")
append_df = pd.read_pickle(args.append)

print(f"Loading Exp-Organic from {args.organic}")
organic_df = pd.read_pickle(args.organic)

# ── Score affect for all responses ───────────────────────────────────────────
def score_dataframe(df: pd.DataFrame, response_col: str,
                    meta_cols: list, turn_label: str) -> pd.DataFrame:
    """Score affect for one column of responses. Returns flat DataFrame."""
    rows = []
    for _, row in df.iterrows():
        text = str(row.get(response_col, ""))
        rows.append({
            **{c: row.get(c) for c in meta_cols},
            "turn": turn_label,
            **score_affect(text),
        })
    return pd.DataFrame(rows)

META = ["narrative_idx", "model", "myth_type", "myth_pair",
        "frame", "dose", "is_pair", "prompt_variant"]

print("\nScoring T1 and T2 affect (Exp-Append)...")
t1_affect = score_dataframe(append_df, "response_t1", META, "t1")
t2_affect = score_dataframe(append_df, "response_t2", META, "t2")
affect_df  = pd.concat([t1_affect, t2_affect], ignore_index=True)

out = OUT_DIR / "ExpAppend_AffectScores_AllLexicons_T1andT2.csv"
affect_df.to_csv(out, index=False)
print(f"  Saved: {out.name}")

print("\nScoring T1 affect (Exp-Organic)...")
org_meta = ["narrative_idx", "model", "myth_type", "narrative_nli_label", "prompt_variant"]
org_affect = score_dataframe(organic_df, "response_t1", org_meta, "t1")
out = OUT_DIR / "ExpOrganic_AffectScores_AllLexicons_T1.csv"
org_affect.to_csv(out, index=False)
print(f"  Saved: {out.name}")

# ── Identify affect columns ───────────────────────────────────────────────────
aff_cols = [
    c for c in affect_df.columns
    if any(c.startswith(p) for p in AFFECT_PREFIXES)
    and affect_df[c].std(skipna=True) > 0
]
print(f"\nAffect columns found: {len(aff_cols)}")

# ── VIF check ─────────────────────────────────────────────────────────────────
t2_aff = affect_df[affect_df["turn"] == "t2"].copy()
vif_df  = compute_vif(t2_aff[aff_cols].dropna())
out = OUT_DIR / "AffectFeatures_VIF.csv"
vif_df.to_csv(out, index=False)
print(f"  Saved: {out.name}")

high_vif = vif_df[vif_df["VIF"] > 10]["feature"].tolist()
if high_vif:
    print(f"  High-VIF features (>10): {high_vif}")

# ── Correlation matrix ────────────────────────────────────────────────────────
proj_cols  = [c for c in ["proj_subspace_delta"] if c in append_df.columns]
if proj_cols:
    t2_aff = t2_aff.merge(
        append_df[["narrative_idx", "model", "myth_type", "frame", "dose"]
                  + proj_cols].drop_duplicates(),
        on=["narrative_idx", "model", "myth_type", "frame", "dose"],
        how="left",
    )
corr_cols   = aff_cols + [c for c in proj_cols if c in t2_aff.columns]
corr_matrix = t2_aff[corr_cols].corr()
out = OUT_DIR / "AffectFeatures_CorrelationMatrix.csv"
corr_matrix.to_csv(out)
print(f"  Saved: {out.name}")

sz  = max(10, len(aff_cols) // 2)
fig, ax = plt.subplots(figsize=(sz, sz))
im = ax.imshow(corr_matrix.values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=90, fontsize=6)
ax.set_yticklabels(corr_matrix.columns, fontsize=6)
plt.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig(FIG_DIR / "AffectFeatures_CorrelationMatrix.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved figure: AffectFeatures_CorrelationMatrix.png")

# ── Paired T1 vs T2 tests per myth per feature ────────────────────────────────
print("\nRunning paired T1 vs T2 affect tests...")
t1_af = affect_df[affect_df["turn"] == "t1"]
t2_af = affect_df[affect_df["turn"] == "t2"]
merged = t1_af.merge(
    t2_af,
    on=["narrative_idx", "model", "myth_type", "frame", "dose"],
    suffixes=("_t1", "_t2"),
)

stat_rows = []
for myth in MYTH_TYPES:
    sub = merged[merged["myth_type"] == myth]
    for feat in aff_cols:
        a_col, b_col = f"{feat}_t1", f"{feat}_t2"
        if a_col not in sub.columns or b_col not in sub.columns:
            continue
        vals = sub[[a_col, b_col]].dropna()
        if len(vals) < 5:
            continue
        result = paired_test(vals[a_col].tolist(), vals[b_col].tolist())
        result.update({"myth": myth, "feature": feat})
        stat_rows.append(result)

if stat_rows:
    affect_stats_df = pd.DataFrame(stat_rows)
    out = OUT_DIR / "ExpAppend_AffectShift_StatTests_T1vsT2.csv"
    affect_stats_df.to_csv(out, index=False)
    print(f"  Saved: {out.name}")

    # Heatmap: p-values across myths × features
    pivot = affect_stats_df.pivot(index="feature", columns="myth", values="p")
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) // 3)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=0.1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=6)
    plt.colorbar(im, ax=ax, label="p-value")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "AffectShift_Pvalues_Heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figure: AffectShift_Pvalues_Heatmap.png")

print("\nStep 5 complete.")