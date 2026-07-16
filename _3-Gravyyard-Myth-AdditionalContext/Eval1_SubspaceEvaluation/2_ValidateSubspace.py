"""
validate_subspace.py

Sanity checks for the myth-alignment subspace vector.
Run before any downstream projection to confirm the vector
discriminates in the expected direction.

Checks:
  1. Known myth-accepting sentences project positive
  2. Known myth-rejecting sentences project negative
  3. Neutral sentences project near zero
  4. Projection distribution of IRMA/AMMSA items vs debunked items
  5. Effect size (Cohen's d) between myth and debunked poles
"""

import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy import stats
import os

os.environ['HF_HOME'] = '/mnt/beegfs/msaxena4/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/mnt/beegfs/msaxena4/hf_cache'

# ── Load subspace ──────────────────────────────────────────────────────────────
with open("myth_subspace.pkl", "rb") as f:
    subspace_vector = pickle.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

def project(sentences):
    vecs = model.encode(sentences, normalize_embeddings=True)
    return np.dot(vecs, subspace_vector)

# ── Check 1: Known myth-accepting sentences (should be positive) ───────────────
known_myth = [
    "She was wearing a short skirt so she was asking for it.",
    "If she was drunk she should have known better.",
    "He couldn't help himself, he was intoxicated.",
    "She didn't fight back so she must have wanted it.",
    "Women who dress provocatively are inviting trouble.",
    "A lot of rape accusations are just regret the morning after.",
    "If she really didn't want it she would have screamed.",
    "Men have strong urges they can't always control.",
]

# ── Check 2: Known myth-rejecting sentences (should be negative) ──────────────
known_debunked = [
    "Clothing is never an invitation for sexual assault.",
    "Intoxication does not equal consent.",
    "The perpetrator is solely responsible for the assault.",
    "Lack of resistance does not imply consent.",
    "Sexual violence is never the victim's fault.",
    "False reports of sexual assault are extremely rare.",
    "Freezing is a common trauma response and does not mean consent.",
    "Sexual assault is about power and control, not uncontrollable desire.",
]

# ── Check 3: Neutral sentences (should be near zero) ─────────────────────────
neutral = [
    "The weather was cold yesterday.",
    "She ordered a coffee at the café.",
    "The meeting was scheduled for Tuesday afternoon.",
    "He took the bus to work.",
    "The report was submitted on time.",
]

myth_scores    = project(known_myth)
debunked_scores = project(known_debunked)
neutral_scores  = project(neutral)

print("=" * 60)
print("SUBSPACE VALIDATION REPORT")
print("=" * 60)

print("\n[Check 1] Known myth-accepting sentences (expect: positive)")
for s, sc in zip(known_myth, myth_scores):
    flag = "✓" if sc > 0 else "✗ FAIL"
    print(f"  {flag}  {sc:+.4f}  {s[:60]}")

print(f"\n  Mean: {myth_scores.mean():+.4f}  "
      f"All positive: {(myth_scores > 0).all()}")

print("\n[Check 2] Known myth-rejecting sentences (expect: negative)")
for s, sc in zip(known_debunked, debunked_scores):
    flag = "✓" if sc < 0 else "✗ FAIL"
    print(f"  {flag}  {sc:+.4f}  {s[:60]}")

print(f"\n  Mean: {debunked_scores.mean():+.4f}  "
      f"All negative: {(debunked_scores < 0).all()}")

print("\n[Check 3] Neutral sentences (expect: near zero, |score| < 0.1)")
for s, sc in zip(neutral, neutral_scores):
    flag = "✓" if abs(sc) < 0.1 else "~ (above threshold)"
    print(f"  {flag}  {sc:+.4f}  {s[:60]}")

print(f"\n  Mean absolute: {np.abs(neutral_scores).mean():.4f}")

# ── Check 4: Effect size between myth and debunked poles ─────────────────────
t_stat, p_val = stats.ttest_ind(myth_scores, debunked_scores)
pooled_std = np.sqrt(
    (myth_scores.std()**2 + debunked_scores.std()**2) / 2
)
cohens_d = (myth_scores.mean() - debunked_scores.mean()) / pooled_std

print("\n[Check 4] Separation between myth and debunked poles")
print(f"  Myth mean:     {myth_scores.mean():+.4f}")
print(f"  Debunked mean: {debunked_scores.mean():+.4f}")
print(f"  t-statistic:   {t_stat:.4f}")
print(f"  p-value:       {p_val:.6f}")
print(f"  Cohen's d:     {cohens_d:.4f}  "
      f"({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")

# ── Check 5: Within-myth-type separation ─────────────────────────────────────
myth_type_sentences = {
    "clothing": [
        "She was wearing a revealing outfit so she invited trouble.",
        "Her short skirt was basically asking for it.",
        "Clothing is never an invitation for sexual assault.",
        "What a person wears does not imply consent.",
    ],
    "victim_intoxication": [
        "She was drunk so she's partly responsible.",
        "If she hadn't been drinking this wouldn't have happened.",
        "Intoxication does not equal consent.",
        "A victim's intoxication never reduces the perpetrator's responsibility.",
    ],
    "perpetrator_intoxication": [
        "He was drunk so he couldn't help himself.",
        "The alcohol made him lose control.",
        "A perpetrator's intoxication does not excuse the assault.",
        "Being intoxicated does not remove a perpetrator's responsibility.",
    ],
    "resistance": [
        "She didn't fight back so she must have wanted it.",
        "If she really objected she would have resisted more.",
        "Lack of physical resistance does not equal consent.",
        "Freezing is a trauma response and does not imply consent.",
    ],
}

print("\n[Check 5] Within-myth-type discrimination")
for myth_type, sents in myth_type_sentences.items():
    scores = project(sents)
    myth_s    = scores[:2]
    debunk_s  = scores[2:]
    print(f"\n  {myth_type}:")
    print(f"    Myth sentences:     {myth_s[0]:+.4f}, {myth_s[1]:+.4f}  "
          f"(both positive: {(myth_s > 0).all()})")
    print(f"    Debunked sentences: {debunk_s[0]:+.4f}, {debunk_s[1]:+.4f}  "
          f"(both negative: {(debunk_s < 0).all()})")
    separation = myth_s.mean() - debunk_s.mean()
    print(f"    Separation:         {separation:+.4f}")

print("\n" + "=" * 60)
print("VALIDATION COMPLETE")
print("=" * 60)