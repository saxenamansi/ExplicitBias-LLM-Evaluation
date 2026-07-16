"""Encode original narratives and save projection scores per embedding."""

import pickle, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from Embeddings import get_w2v, get_glove, mean_word_embedding

NARRATIVES_PATH = "../Data/Reddit-OriginalNarratives-SV-Data.csv"
OUT_PATH        = "ProjectionValidation/NarrativeProjections.csv"

# ── Core utilities ─────────────────────────────────────────────────────────────
def load_subspace(name):
    with open(f"MythSubspaces/{name}.pkl", "rb") as f:
        return pickle.load(f)

def project_narratives(narratives, subspace_vector, encode_fn):
    return np.dot(np.array([encode_fn(t) for t in narratives]), subspace_vector)

    
# ── Load Modelsn ───────────────────────────────────────────────────────────────
sbert = SentenceTransformer("all-mpnet-base-v2")
w2v   = get_w2v()
glove = get_glove()

embeddings = {
    "SBERT": (load_subspace("SBERT"), lambda s: sbert.encode([s], normalize_embeddings=True)[0]),
    "W2V":   (load_subspace("W2V"),   lambda s: mean_word_embedding(s, w2v)),
    "GLOVE": (load_subspace("GLOVE"), lambda s: mean_word_embedding(s, glove)),
}

def main():
    narr_df = pd.read_csv(NARRATIVES_PATH).reset_index().rename(
        columns={"index": "narrative_index"}
    )
    narr_df["narrative"] = narr_df["Title"] + " [SEP] " + narr_df["Text"]
    out = narr_df[["narrative_index"]].copy()
    for name, (vec, fn) in embeddings.items():
        out[f"projection_{name.lower()}"] = project_narratives(
            narr_df["narrative"].tolist(), vec, fn
        )
    out.to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()