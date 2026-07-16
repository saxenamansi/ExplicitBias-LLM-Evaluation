import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import wilcoxon
from scipy.spatial.distance import cosine
from statsmodels.stats.multitest import multipletests

MODELS = ["gemma", "llama", "mistral", "qwen"]
EMBS   = ["SBERT"]
# "W2V", "GLOVE"]

EMB_DIR  = Path("../Results/Summarization/1_Embeddings/")
EVAL_DIR = Path("MythSubspaces")

def load_subspace(emb):
    with open(EVAL_DIR / f"{emb}.pkl", "rb") as f:
        return pickle.load(f)

subspaces = {emb: load_subspace(emb) for emb in EMBS}

results = []
for emb in EMBS:
    for model in MODELS:
        pairs = pd.read_pickle(EMB_DIR / emb / f"{model}_t2_pairs.pkl")
        meta  = pd.read_csv(EMB_DIR / f"{model}_t2_metadata.csv")
        meta["cosine_dist"] = pairs.apply(lambda r: cosine(r[f"{emb}_t1"], r[f"{emb}_t2"]), axis=1).values
        
        for keys, grp in meta.groupby(["myth_type", "frame", "dose"], dropna=True):
            vals = grp["cosine_dist"].dropna().values
            if len(vals) < 3:
                continue
            stat, p = wilcoxon(vals, alternative="greater")
            n  = len(vals)
            dz = np.mean(vals) / np.std(vals, ddof=1)
            cf = 1 - (3 / (4 * (n - 1) - 1))
            results.append({
                "model": model, "emb": emb,
                "myth_type": keys[0], "frame": keys[1], "dose": keys[2],
                "stat": stat, "p": p, "n": n,
                "mean_dist": np.mean(vals), "cohens_dz": dz, "hedges_g": dz * cf
            })

cosine_df = pd.DataFrame(results)
cosine_df["significant"], cosine_df["p_bh"], _, cosine_df["alpha"] = multipletests(cosine_df["p"], method="fdr_bh")
cosine_df.to_csv("ContextAppend-Summarization-SemanticShift.csv", index=False)