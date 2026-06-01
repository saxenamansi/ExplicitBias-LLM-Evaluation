"""Subspace validation: bipole evaluation, effect sizes, myth-type checks, scale projections."""

import pickle, numpy as np, pandas as pd
from scipy import stats
from itertools import combinations
from sentence_transformers import SentenceTransformer

from Embeddings import get_w2v, get_glove, mean_word_embedding
from SubspaceValidationSentences import KNOWN_MYTH, KNOWN_DEBUNKED, NEUTRAL, MYTH_TYPE_SENTENCES
from RMAScaleSentences import SCALES

# ── Core utilities ─────────────────────────────────────────────────────────────
def load_subspace(name):
    with open(f"MythSubspaces/{name}.pkl", "rb") as f:
        return pickle.load(f)

def project(sentences, subspace_vector, encode_fn):
    vecs = np.array([encode_fn(s) for s in sentences])
    return np.dot(vecs, subspace_vector)

def cohens_d_two_random_variables(a, b):
    pooled_std = np.sqrt((a.std()**2 + b.std()**2) / 2)
    d = (a.mean() - b.mean()) / pooled_std
    label = "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
    return d, label

# ── Validation functions ───────────────────────────────────────────────────────
def bipole_evaluation(embedding_name, subspace_vector, encode_fn):
    """Project known myth, debunked, and neutral sentences onto the subspace."""
    results = []
    for label, sents in [
        ("myth",     KNOWN_MYTH),
        ("debunked", KNOWN_DEBUNKED),
        ("neutral",  NEUTRAL),
    ]:
        scores = project(sents, subspace_vector, encode_fn)
        for sent, score in zip(sents, scores):
            results.append({
                "embedding": embedding_name, 
                "scale": "sanity_check",
                "label": label, 
                "sentence": sent, 
                "score": score
            })
    return results

def rma_scale_projection(embedding_name, subspace_vector, encode_fn, scales=SCALES):
    """
    Project scale items and their debunked counterparts onto subspace.
    scales: list of (scale_name, myth_sents, debunked_sents)
    """
    results = []
    for scale_name, myth_sents, debunked_sents in scales:
        for label, sents in [
            ("myth",     myth_sents),
            ("debunked", debunked_sents)
        ]:
            scores = project(sents, subspace_vector, encode_fn)
            for sent, score in zip(sents, scores):
                results.append({
                    "embedding": embedding_name, 
                    "scale": scale_name,
                    "label": label, 
                    "sentence": sent, 
                    "score": score
                })
    return results

def effect_size_bipole(embedding_name, subspace_vector, encode_fn):
    """Cohen's d between known myth and debunked poles."""
    myth_scores     = project(KNOWN_MYTH,     subspace_vector, encode_fn)
    debunked_scores = project(KNOWN_DEBUNKED, subspace_vector, encode_fn)
    t_stat, p_val   = stats.ttest_ind(myth_scores, debunked_scores)
    d, d_label      = cohens_d_two_random_variables(myth_scores, debunked_scores)
    return [{
        "embedding": embedding_name, "check": "bipole",
        "myth_type": "common",
        "myth_mean": myth_scores.mean(),
        "debunked_mean": debunked_scores.mean(),
        "mean_separation": myth_scores.mean() - debunked_scores.mean(),
        "t_stat": t_stat, 
        "p_value": p_val,
        "cohens_d": d, 
        "effect_size": d_label
    }]

def effect_size_rma_scales(embedding_name, subspace_vector, encode_fn, scales=SCALES):
    """Cohen's d between myth and debunked poles for each scale."""
    results = []
    for scale_name, myth_sents, debunked_sents in scales:
        myth_scores     = project(myth_sents,     subspace_vector, encode_fn)
        debunked_scores = project(debunked_sents, subspace_vector, encode_fn)
        t_stat, p_val   = stats.ttest_ind(myth_scores, debunked_scores)
        d, d_label      = cohens_d_two_random_variables(myth_scores, debunked_scores)
        results.append({
            "embedding": embedding_name, "check": scale_name,
            "myth_type": "all",
            "myth_mean": myth_scores.mean(),
            "debunked_mean": debunked_scores.mean(),
            "mean_separation": myth_scores.mean() - debunked_scores.mean(),
            "t_stat": t_stat, 
            "p_value": p_val,
            "cohens_d": d, 
            "effect_size": d_label
        })
    return results

# ── Runner ─────────────────────────────────────────────────────────────────────
def run_validation(embeddings, scales=SCALES):
    """
    embeddings: {"SBERT": (vector, encode_fn), "W2V": (vector, encode_fn), ...}
    Writes BipoleEvaluation.csv and EffectSize.csv.
    """
    
    BIPOLE_PATH     = "ProjectionValidation/BipoleEvaluation.csv"
    EFFECT_SIZE_PATH = "ProjectionValidation/EffectSize.csv"
    
    bipole_rows, effect_rows = [], []
    for name, (vec, fn) in embeddings.items():
        
        bipole_rows += bipole_evaluation(name, vec, fn)
        bipole_rows += rma_scale_projection(name, vec, fn)
        
        effect_rows += effect_size_bipole(name, vec, fn)
        effect_rows += effect_size_rma_scales(name, vec, fn)

    pd.DataFrame(bipole_rows).to_csv(BIPOLE_PATH, index=False)
    pd.DataFrame(effect_rows).to_csv(EFFECT_SIZE_PATH, index=False)

# Load Models
sbert = SentenceTransformer("all-mpnet-base-v2")
w2v   = get_w2v()
glove = get_glove()

embeddings = {
    "SBERT": (load_subspace("SBERT"), lambda s: sbert.encode([s], normalize_embeddings=True)[0]),
    "W2V":   (load_subspace("W2V"),   lambda s: mean_word_embedding(s, w2v)),
    "GLOVE": (load_subspace("GLOVE"), lambda s: mean_word_embedding(s, glove)),
}

# Main
run_validation(embeddings)
