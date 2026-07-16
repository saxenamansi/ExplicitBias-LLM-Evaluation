"""
Embeddings.py
===============
Load embedding models once and expose helpers used across step scripts.
Import this module; do not re-load models in individual step scripts.

Models loaded:
  SBERT  — all-mpnet-base-v2
  W2V    — word2vec-google-news-300
  GloVe  — glove-wiki-gigaword-300

All models are loaded lazily via get_*() functions so importing this module
does not trigger loading until a step script actually needs them.
"""

import pickle
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from Config import SBERT_MODEL, SUBSPACE_PKL, MYTH_UNIT_SENTENCES

# ── Lazy singletons ───────────────────────────────────────────────────────────
_sbert   = None
_w2v     = None
_glove   = None
_subspace = None
_myth_unit_vecs = None


def get_sbert() -> SentenceTransformer:
    global _sbert
    if _sbert is None:
        print(f"Loading SBERT ({SBERT_MODEL})...")
        _sbert = SentenceTransformer(SBERT_MODEL)
    return _sbert


def get_w2v():
    global _w2v
    if _w2v is None:
        import gensim.downloader as gensim_api
        print("Loading Word2Vec (word2vec-google-news-300)...")
        _w2v = gensim_api.load("word2vec-google-news-300")
    return _w2v


def get_glove():
    global _glove
    if _glove is None:
        import gensim.downloader as gensim_api
        print("Loading GloVe (glove-wiki-gigaword-300)...")
        _glove = gensim_api.load("glove-wiki-gigaword-300")
    return _glove


def get_subspace() -> np.ndarray:
    """
    Load myth-alignment subspace vector from pkl.
    Returns None if file missing or wrong dimension.
    """
    global _subspace
    if _subspace is not None:
        return _subspace
    if not SUBSPACE_PKL.exists():
        print(f"WARNING: {SUBSPACE_PKL} not found. Subspace projection will be skipped.")
        return None
    with open(SUBSPACE_PKL, "rb") as f:
        vec = pickle.load(f)
    vec = np.array(vec)
    if vec.shape[0] != 768:
        print(f"WARNING: Subspace vector is {vec.shape[0]}-dim, expected 768. "
              f"Rebuild with 0_BuildSubspace.py using {SBERT_MODEL}.")
        return None
    print(f"Loaded subspace vector ({vec.shape[0]}-dim).")
    _subspace = vec
    return _subspace


def get_myth_unit_vecs() -> dict:
    """
    Encode myth unit sentences with SBERT.
    Returns {myth_type: unit_vector (768-dim, normalized)}.
    """
    global _myth_unit_vecs
    if _myth_unit_vecs is None:
        sbert = get_sbert()
        _myth_unit_vecs = {
            myth: sbert.encode([sent], normalize_embeddings=True)[0]
            for myth, sent in MYTH_UNIT_SENTENCES.items()
        }
    return _myth_unit_vecs


# ── Encoding helpers ──────────────────────────────────────────────────────────

def encode_sbert(texts: list, batch_size: int = 64) -> np.ndarray:
    """Encode list of strings with SBERT. Returns (N, 768) normalized array."""
    return get_sbert().encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=batch_size,
    )

 
def mean_word_embedding(text: str, model) -> np.ndarray:
    """Mean pool word vectors for tokens found in model vocabulary."""
    words = str(text).lower().split()
    vecs  = [model[w] for w in words if w in model]
    if not vecs:
        return np.zeros(model.vector_size)
    return np.mean(vecs, axis=0)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0][0]