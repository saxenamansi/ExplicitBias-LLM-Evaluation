"""Paired Wilcoxon signed-rank tests on narrative-T1 cosine and myth alignment for real stories."""
import argparse
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import wilcoxon
from sentence_transformers import SentenceTransformer
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["Summarization", "AdviceGeneration"], required=True)
args = parser.parse_args()

MODELS      = ["gemma", "llama", "mistral", "qwen"]
SOURCES     = ["JustDetention", "SurvivorStories"]
EMB         = "SBERT"
RESULTS_DIR = Path("../SV-Org-RealStories/Results")
EVAL_DIR    = Path("MythSubspaces")

JD_PATH = "Data/JustDetention.csv"
SS_PATH = "Data/SurvivorStories.csv"

with open(EVAL_DIR / f"{EMB}.pkl", "rb") as f:
    subspace = pickle.load(f)

sbert = SentenceTransformer("all-mpnet-base-v2")

jd = pd.read_csv(JD_PATH).reset_index().rename(columns={"index": "narrative_index"})
jd["narrative"] = jd["blurb"]

ss = pd.read_csv(SS_PATH).reset_index().rename(columns={"index": "narrative_index"})
ss["narrative"] = ss["title"] + " [SEP] " + ss["text"]
ss["narrative_index"] = ss["narrative_index"] + len(jd)

narr_lookup = pd.concat([jd[["narrative_index", "narrative"]],
                         ss[["narrative_index", "narrative"]]], ignore_index=True)
narr_embs   = sbert.encode(narr_lookup["narrative"].tolist(), normalize_embeddings=True)
narr_lookup["proj_narr"] = np.dot(narr_embs, subspace)

source_offset = {"JustDetention": 0, "SurvivorStories": len(jd)}

results = []
for source in SOURCES:
    task_label = "advice" if args.task == "AdviceGeneration" else "summary"
    offset     = source_offset[source]
    for model in MODELS:
        csv_path = RESULTS_DIR / source / f"{model}_{task_label}_t1.csv"
        meta     = pd.read_csv(csv_path)
        meta["source"] = source
        if "prompt_variant" not in meta.columns:
            meta["prompt_variant"] = "default"

        t1_embs        = pd.read_pickle(RESULTS_DIR / source / f"{EMB}" / f"{model}_{task_label}_t1_embs.pkl")
        t1_emb_matrix  = np.vstack(t1_embs.values)
        meta["proj_t1"] = np.dot(t1_emb_matrix, subspace)

        adj_idx                = meta["narrative_idx"] + offset
        narr_emb_subset        = narr_embs[adj_idx.values]
        meta["cosine"]         = (narr_emb_subset * t1_emb_matrix).sum(axis=1)

        meta = meta.merge(narr_lookup[["narrative_index", "proj_narr"]],
                          left_on=adj_idx, right_on="narrative_index", how="left")
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
                results.append({"source": source, "model": model, "prompt_variant": pv,
                                "metric": metric, "stat": stat, "p": p, "n": n,
                                "mean_diff": np.mean(vals), "cohens_dz": dz, "hedges_g": dz * cf})

proj_df = pd.DataFrame(results)
proj_df["significant"], proj_df["p_bh_corrected"], _, proj_df["alpha"] = multipletests(proj_df["p"], method="fdr_bh")
proj_df.to_csv(f"StatisticalTest/RealStories-T1-{args.task}-Faithfulness-WilcoxonSignedRank.csv", index=False)