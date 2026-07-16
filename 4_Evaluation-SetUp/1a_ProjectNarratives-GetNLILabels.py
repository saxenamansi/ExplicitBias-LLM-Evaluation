"""Encode original narratives, project onto myth subspaces, and join NLI demographic labels."""
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from Embeddings import get_w2v, get_glove, mean_word_embedding

NARRATIVES_PATH = "../Data/Reddit-OriginalNarratives-SV-Data.csv"
NLI_PATH        = "../Data/SentenceNLI-SV-Full.csv"
OUT_PATH        = "StatisticalTestResults/NarrativeProjections-withDemographics.csv"
DEMO_CATS       = ["AGE", "GENDER", "PERSPECTIVE", "RELATIONSHIP", "MYTH"]

def load_subspace(name):
    with open(f"MythSubspaces/{name}.pkl", "rb") as f:
        return pickle.load(f)

def project_narratives(narratives, subspace_vector, encode_fn):
    return np.dot(np.array([encode_fn(t) for t in narratives]), subspace_vector)

def build_wide_labels(nli):
    """Pivot NLI labels to one row per narrative, one column per myth_type."""
    rows = []
    for cat in DEMO_CATS:
        cat_df = nli[nli["myth_category"] == cat][["narrative_index", "myth_type", "overall_label"]]
        rows.append(cat_df.pivot(index="narrative_index", columns="myth_type", values="overall_label"))
    return pd.concat(rows, axis=1).reset_index()

def main():
    narr_df = pd.read_csv(NARRATIVES_PATH).reset_index().rename(columns={"index": "narrative_index"})
    narr_df["narrative"] = narr_df["Title"] + " [SEP] " + narr_df["Text"]

    sbert = SentenceTransformer("all-mpnet-base-v2")
    # w2v   = get_w2v()
    # glove = get_glove()
    embeddings = {
        "SBERT": (load_subspace("SBERT"), lambda s: sbert.encode([s], normalize_embeddings=True)[0]),
        # "W2V":   (load_subspace("W2V"),   lambda s: mean_word_embedding(s, w2v)),
        # "GLOVE": (load_subspace("GLOVE"), lambda s: mean_word_embedding(s, glove)),
    }

    out = narr_df[["narrative_index"]].copy()
    for name, (vec, fn) in embeddings.items():
        out[f"projection_{name.lower()}"] = project_narratives(narr_df["narrative"].tolist(), vec, fn)

    nli  = pd.read_csv(NLI_PATH)
    wide = build_wide_labels(nli)
    out.merge(wide, on="narrative_index").to_csv(OUT_PATH, index=False)

if __name__ == "__main__":
    main()