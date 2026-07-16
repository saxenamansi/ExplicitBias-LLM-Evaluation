"""Shapiro-Wilk normality tests on paired narrative-T1 differences (cosine and myth alignment)."""
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import shapiro
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["Summarization", "AdviceGeneration"], required=True)
args = parser.parse_args()

MODELS          = ["gemma", "llama", "mistral", "qwen"]
EMB             = "SBERT"
EMB_DIR         = Path(f"../Results/{args.task}/1_Embeddings/")
EVAL_DIR        = Path("MythSubspaces")
NARRATIVES_PATH = Path("../Data/Reddit-OriginalNarratives-SV-Data.csv")

with open(EVAL_DIR / f"{EMB}.pkl", "rb") as f:
    subspace = pickle.load(f)

sbert    = SentenceTransformer("all-mpnet-base-v2")
narr_df  = pd.read_csv(NARRATIVES_PATH).reset_index().rename(columns={"index": "narrative_index"})
narr_df["narrative"] = narr_df["Title"] + " [SEP] " + narr_df["Text"]
narr_embs = sbert.encode(narr_df["narrative"].tolist(), normalize_embeddings=True)
narr_proj = np.dot(narr_embs, subspace)
narr_df["proj_narr"] = narr_proj

results = []
for model in MODELS:
    t1_embs = pd.read_pickle(EMB_DIR / EMB / f"{model}_t1.pkl")
    meta    = pd.read_csv(EMB_DIR / f"{model}_t1_metadata.csv")
    if "prompt_variant" not in meta.columns:
        meta["prompt_variant"] = "default"
    meta["narrative_idx"] = meta["narrative_idx"].astype(int)
    meta["proj_t1"] = t1_embs.apply(lambda v: np.dot(v, subspace)).values

    # cosine similarity between narrative and T1 embeddings
    t1_emb_matrix   = np.vstack(t1_embs.values)
    narr_emb_subset = narr_embs[meta["narrative_idx"].values]
    meta["cosine"]  = (narr_emb_subset * t1_emb_matrix).sum(axis=1)  # dot of normalized = cosine

    meta = meta.merge(narr_df[["narrative_index", "proj_narr"]],
                      left_on="narrative_idx", right_on="narrative_index", how="left")
    meta["proj_diff"] = meta["proj_t1"] - meta["proj_narr"]

    for pv, grp in meta.groupby("prompt_variant"):
        for metric, col in [("cosine", "cosine"), ("myth_alignment", "proj_diff")]:
            vals = grp[col].dropna()
            if len(vals) < 3:
                continue
            stat, p = shapiro(vals.sample(min(len(vals), 5000), random_state=42))
            results.append({"model": model, "prompt_variant": pv, "metric": metric,
                            "n": len(vals), "mean": vals.mean(), "std": vals.std(),
                            "skewness": vals.skew(), "kurtosis": vals.kurtosis(), "shapiro_p": p})

shapiro_df = pd.DataFrame(results)
shapiro_df["significant"], shapiro_df["p_bh_corrected"], _, shapiro_df["alpha"] = multipletests(shapiro_df["shapiro_p"], method="fdr_bh")
shapiro_df.to_csv(f"StatisticalTestResults/T1-{args.task}-CosineDist-Normality.csv", index=False)