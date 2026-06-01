"""
BuildMythSubspace.py
"""

import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from Embeddings import get_w2v, get_glove, mean_word_embedding

from RMAScaleSentences import IRMA_MYTHS, IRMA_DEBUNKED, AMMSA_MYTHS, AMMSA_DEBUNKED

all_myths    = IRMA_MYTHS + AMMSA_MYTHS
all_debunked = IRMA_DEBUNKED + AMMSA_DEBUNKED

# Loading models: 
sbert = SentenceTransformer("all-mpnet-base-v2")
w2v, glove = get_w2v(), get_glove()

def get_encoding(isStatic, myths = all_myths, debunked = all_debunked, model = None, encode_fn = None):
    if isStatic:
        if not encode_fn: raise ValueError("Please pass static encoding function")
        else:
            myth_vecs = [encode_fn(s) for s in myths]
            debunked_vecs = [encode_fn(s) for s in debunked]
    else:
        if not model: raise ValueError("Please pass contextual encoding model")
        else:
            myth_vecs = model.encode(myths, normalize_embeddings=True)
            debunked_vecs = model.encode(debunked, normalize_embeddings=True)
    return myth_vecs, debunked_vecs
            
def build_subspace(isStatic, model = None, encode_fn = None):  
    """Normalized mean-difference subspace vector (myth-accepting → myth-rejecting)."""
    if isStatic:
        myth_vecs, debunked_vecs = get_encoding(isStatic, encode_fn = encode_fn)
    else:
        myth_vecs, debunked_vecs = get_encoding(isStatic, model = model)
    vec = np.mean(myth_vecs, axis=0) - np.mean(debunked_vecs, axis=0)
    return vec / np.linalg.norm(vec)

subspaces  = {
    "SBERT": build_subspace(False, model = sbert),
    "W2V": build_subspace(True, encode_fn =  lambda s: mean_word_embedding(s, w2v)),
    "GLOVE": build_subspace(True, encode_fn =  lambda s: mean_word_embedding(s, glove))
}

for name, vec in subspaces.items():
    with open(f"MythSubspaces/{name}.pkl", "wb") as f:
        pickle.dump(vec, f)
print("Saved MythSubspace{SBERT|W2V|GloVe}.pkl")
