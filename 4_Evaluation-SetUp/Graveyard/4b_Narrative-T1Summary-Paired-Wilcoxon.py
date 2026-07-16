"""Paired Wilcoxon signed-rank tests on narrative-T1 cosine similarity and myth alignment."""
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import wilcoxon
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

    t1_emb_matrix   = np.vstack(t1_embs.values)
    narr_emb_subset = narr_embs[meta["narrative_idx"].values]
    meta["cosine"]  = (narr_emb_subset * t1_emb_matrix).sum(axis=1)

    meta = meta.merge(narr_df[["narrative_index", "proj_narr"]],
                      left_on="narrative_idx", right_on="narrative_index", how="left")
    meta["proj_diff"] = meta["proj_t1"] - meta["proj_narr"]

    for pv, grp in meta.groupby("prompt_variant"):
        for metric, col in [("cosine", "cosine"), ("myth_alignment", "proj_diff")]:
            vals = grp[col].dropna().values
            if len(vals) < 3:
                continue
            stat, p = wilcoxon(vals)
            n  = len(vals)
            dz = np.mean(vals) / np.std(vals, ddof=1)
            cf = 1 - (3 / (4 * (n - 1) - 1))
            results.append({"model": model, "prompt_variant": pv, "metric": metric,
                            "stat": stat, "p": p, "n": n,
                            "mean_diff": np.mean(vals), "cohens_dz": dz, "hedges_g": dz * cf})

proj_df = pd.DataFrame(results)
proj_df["significant"], proj_df["p_bh_corrected"], _, proj_df["alpha"] = multipletests(proj_df["p"], method="fdr_bh")
proj_df.to_csv(f"StatisticalTestResults/T1-{args.task}-MythAlignment-WilcoxonSignedRank.csv", index=False)