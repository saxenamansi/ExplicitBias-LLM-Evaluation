"""
For Advice Generated Responses:
Compute cosine similarity and projection scores, run paired statistical tests.
H0: mean(cosine_distance(t1,t2)) = 0  [semantic shift]
H0: mean(proj(t2)) = mean(proj(t1))     [myth alignment shift]
Outputs: 2_SemanticShift_Cosine/{emb}_{singles|pairs}.csv
         3_MythShift_Projection/{emb}_{singles|pairs}.csv
"""
import sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial.distance import cosine
from statsmodels.stats.multitest import multipletests
import importlib.util

def load_module(name, rel_path):
    path = Path(__file__).parent.parent / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

StatisticalTests  = load_module("StatisticalTests",    "4_Evaluation-SetUp/StatisticalTests.py")
paired_test    = StatisticalTests.paired_test

MODELS     = ["gemma", "llama", "mistral", "phi", "qwen"]
EMBS       = ["SBERT", "W2V", "GLOVE"]
EVAL_DIR   = Path(__file__).parent.parent / "4_Evaluation-SetUp"
EMB_DIR    = Path(__file__).parent / "SampleResults/1_Embeddings"
COS_DIR    = Path(__file__).parent / "SampleResults/2_SemanticShift_Cosine"
PROJ_DIR   = Path(__file__).parent / "SampleResults/3_MythShift_Projection"
META_COLS  = ["narrative_idx", "model", "prompt_variant", "myth_type", "myth_pair", "frame", "dose"]
SINGLE_KEY = ["model", "prompt_variant", "myth_type", "frame", "dose"]
PAIR_KEY   = ["model", "prompt_variant", "myth_pair",  "frame", "dose"]

for d in [COS_DIR, PROJ_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def load_subspace(emb):
    fname = f"MythSubspaces/{emb}.pkl" 
    with open(EVAL_DIR / fname, "rb") as f:
        return pickle.load(f)

def run_tests_onesample(df, key_cols, scalar_col):
    """For cosine: one-sample against zero."""
    rows = []
    for keys, grp in df.groupby(key_cols, dropna=False):
        d = grp[scalar_col].dropna().values
        if len(d) < 3: continue
        result = paired_test(d.tolist(), [0.0] * len(d))
        row = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(result)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out

def run_tests_paired(df, key_cols):
    """For projection: true paired t2 vs t1."""
    rows = []
    for keys, grp in df.groupby(key_cols, dropna=False):
        t1 = grp["proj_t1"].dropna().values
        t2 = grp["proj_t2"].dropna().values
        if len(t1) < 3: continue
        result = paired_test(t2.tolist(), t1.tolist())
        row = dict(zip(key_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(result)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    return out

for emb in EMBS:
    subspace = load_subspace(emb)
    frames   = []
    for model in MODELS:
        pairs = pd.read_pickle(EMB_DIR / emb / f"{model}_t2_pairs.pkl")
        meta  = pd.read_csv(EMB_DIR / f"{model}_t2_metadata.csv")[META_COLS]
        meta  = meta.reset_index(drop=True)

        # cosine distance (semantic shift)
        meta[f"cosine_dist"] = [
            cosine(pairs[f"{emb}_t1"].iloc[i], pairs[f"{emb}_t2"].iloc[i])
            for i in range(len(pairs))
        ]
        # projection shift (myth alignment)
        meta["proj_t1"] = [np.dot(pairs[f"{emb}_t1"].iloc[i], subspace) for i in range(len(pairs))]
        meta["proj_t2"] = [np.dot(pairs[f"{emb}_t2"].iloc[i], subspace) for i in range(len(pairs))]

        frames.append(meta)

    df      = pd.concat(frames, ignore_index=True)
    singles = df[df["myth_pair"].isna()]
    pairs   = df[df["myth_pair"].notna()]

    for label, grp, key in [("singles", singles, SINGLE_KEY), ("pairs", pairs, PAIR_KEY)]:

        # Cosine Similarity 
        run_tests_onesample(grp, key, "cosine_dist").to_csv(COS_DIR / f"{emb}_{label}.csv", index=False)

        # Projection
        run_tests_paired(grp, key).to_csv(PROJ_DIR / f"{emb}_{label}.csv", index=False)

    print(f"Done: {emb}")