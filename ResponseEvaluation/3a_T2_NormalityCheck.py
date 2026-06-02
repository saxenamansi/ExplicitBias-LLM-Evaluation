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
args = parser.parse_args()

MODELS   = ["gemma", "llama", "mistral", "qwen"]
EMBS     = ["SBERT"]
EMB_DIR  = Path(f"../Results/{args.task}/1_Embeddings/") # Change to your path
EVAL_DIR = Path("MythSubspaces") # Change to your path

def load_subspace(emb):
    with open(EVAL_DIR / f"{emb}.pkl", "rb") as f: # Change to your path
        return pickle.load(f)

subspaces = {emb: load_subspace(emb) for emb in EMBS}
results = []

for emb in EMBS:
    for model in MODELS:
        pairs = pd.read_pickle(EMB_DIR / emb / f"{model}_t2_pairs.pkl") # Change to your path
        meta  = pd.read_csv(EMB_DIR / f"{model}_t2_metadata.csv") # Change to your path
      
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
shapiro_df.to_csv(f"StatisticalTestResults/T2-{args.task}-NormalityCheck.csv", index=False)
