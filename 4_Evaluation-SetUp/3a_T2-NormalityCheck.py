"""Shapiro-Wilk normality tests on T2 projection differences."""
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import shapiro
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["Summarization", "AdviceGeneration"], required=True)
parser.add_argument("--source", choices=["reddit", "JustDetention", "SurvivorStories"], default="reddit")
args = parser.parse_args()

MODELS   = ["gemma", "llama", "mistral", "qwen"]
EMBS     = ["SBERT"]
EVAL_DIR = Path("MythSubspaces")

EMB_DIR = {
    "reddit":          Path(f"../Results/{args.task}/1_Embeddings_1stPOV/"),
    "JustDetention":   Path("../Results/JustDetention/1_Embeddings/"),
    "SurvivorStories": Path("../Results/SurvivorStories/1_Embeddings/"),
}[args.source]

suffix  = f"-{args.task}_1stPOV" if args.source == "reddit" else f"-{args.source}-{args.task}"
out_dir = Path("StatisticalTestResults/1stPOV/MYTHS")
out_dir.mkdir(parents=True, exist_ok=True)

def load_subspace(emb):
    with open(EVAL_DIR / f"{emb}.pkl", "rb") as f:
        return pickle.load(f)

subspaces = {emb: load_subspace(emb) for emb in EMBS}
results = []

for emb in EMBS:
    for model in MODELS:
        pairs = pd.read_pickle(EMB_DIR / emb / f"{model}_t2_pairs.pkl")
        meta  = pd.read_csv(EMB_DIR / f"{model}_t2_metadata.csv")
        if "prompt_variant" not in meta.columns:
            meta["prompt_variant"] = "default"
        meta["proj_t1"]   = pairs[f"{emb}_t1"].apply(lambda v: np.dot(v, subspaces[emb])).values
        meta["proj_t2"]   = pairs[f"{emb}_t2"].apply(lambda v: np.dot(v, subspaces[emb])).values
        assert meta["proj_t1"].notna().all() and meta["proj_t2"].notna().all(), f"Null projections for {model}"
        meta["proj_diff"] = meta["proj_t2"] - meta["proj_t1"]
        meta["myth"]      = meta["myth_type"].fillna(meta["myth_pair"])
        for keys, grp in meta[meta["myth"].notna()].groupby(["myth", "frame", "dose", "prompt_variant"]):
            vals = grp["proj_diff"].values
            if len(vals) < 3:
                raise Error
            stat, p = shapiro(vals)
            results.append({"model": model, "emb": emb,
                            "myth": keys[0], "frame": keys[1], "dose": keys[2], "prompt_variant": keys[3],
                            "n": len(vals), "mean": vals.mean(), "std": vals.std(),
                            "skewness": pd.Series(vals).skew(), "kurtosis": pd.Series(vals).kurtosis(),
                            "p_value": p})

shapiro_df = pd.DataFrame(results)
shapiro_df["non_normal"], shapiro_df["p_bh_corrected"], _, shapiro_df["alpha"] = multipletests(shapiro_df["p_value"], method="fdr_bh")
shapiro_df.to_csv(out_dir / f"T2{suffix}-NormalityCheck.csv", index=False)