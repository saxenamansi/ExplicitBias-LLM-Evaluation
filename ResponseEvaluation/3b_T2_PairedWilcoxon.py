"""Wilcoxon signed-rank tests on T2 projection differences."""
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import wilcoxon
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
            result = wilcoxon(vals, method='approx')
            n = len(vals)
            r = result.zstatistic / np.sqrt(n)
            results.append({"model": model, "emb": emb,
                            "myth": keys[0], "frame": keys[1], "dose": keys[2], "prompt_variant": keys[3],
                            "stat": result.statistic, "p_value": result.pvalue, "z": result.zstatistic,
                            "n": n, "mean_diff": np.mean(vals), "rank_biserial": round(r, 4)})

proj_df = pd.DataFrame(results)
proj_df["significant"], proj_df["p_bh_corrected"], _, proj_df["alpha"] = multipletests(proj_df["p_value"], method="fdr_bh")
proj_df.to_csv(f"StatisticalTestResults/T2-{args.task}-MythAlignment.csv", index=False)
