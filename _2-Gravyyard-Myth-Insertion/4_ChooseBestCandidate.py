import pandas as pd
import ast
import math

# ── Load NSP results ──────────────────────────────────────────────────────────
df = pd.read_csv(
    "Results/Reddit_candidates-(SV)-NSP-all.csv",
    converters={"modified_sentences": ast.literal_eval}
)

# ── Compute delta NSP ─────────────────────────────────────────────────────────
def compute_delta_nsp(row):
    """
    delta NSP = mean(prev_prob, next_prob) - orig_prob
    Higher = myth sentence fits more naturally at this insertion point.
    Returns NaN if any required probability is missing.
    """
    prev = row["prev_prob"]
    next_ = row["next_prob"]
    orig = row["orig_prob"]

    # Need at least one of prev/next, and orig for comparison
    if math.isnan(orig):
        return float("nan")
    
    available = [p for p in [prev, next_] if not math.isnan(p)]
    if not available:
        return float("nan")
    
    return sum(available) / len(available) - orig

df["delta_nsp"] = df.apply(compute_delta_nsp, axis=1)

# ── Select best candidate per (narrative_idx, myth_type, myth_variation, dose) ─
# Group by narrative + myth configuration, pick insertion with highest delta NSP
best_candidates = (
    df
    .sort_values("delta_nsp", ascending=False)
    .groupby(["narrative_idx", "myth_type", "myth_variation", "dose"], dropna=False)
    .first()
    .reset_index()
)

# ── Rename for summarization pipeline compatibility ───────────────────────────
best_candidates = best_candidates.rename(columns={
    "original_narrative": "narrative_original",
    "modified_narrative": "narrative_modified",
})

# Add a clean unique ID for the summarization pipeline resume logic
best_candidates["id"] = range(len(best_candidates))

# ── Report ────────────────────────────────────────────────────────────────────
print(f"Total candidates:      {len(df)}")
print(f"Best candidates kept:  {len(best_candidates)}")
print(f"Delta NSP stats:\n{best_candidates['delta_nsp'].describe().round(4)}")
print(f"\nRows with NaN delta NSP: {best_candidates['delta_nsp'].isna().sum()}")

# ── Save ──────────────────────────────────────────────────────────────────────
best_candidates.to_csv(
    "Results/ModifiedNarrative-(SV)-bestCandidates.csv",
    index=False
)
print("Saved → Results/ModifiedNarrative-(SV)-bestCandidates.csv")