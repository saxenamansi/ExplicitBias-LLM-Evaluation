"""Shapiro-Wilk normality tests on T1 subspace projections for real stories."""
import argparse
import ast
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import shapiro
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["Summarization", "AdviceGeneration"], required=True)
args = parser.parse_args()

MODELS    = ["gemma", "llama", "mistral", "qwen"]
SOURCES   = ["JustDetention", "SurvivorStories"]
EMB       = "SBERT"
RESULTS_DIR = Path("../SV-Org-RealStories/Results")
EVAL_DIR    = Path("MythSubspaces")
NARR_PATH   = Path("ProjectionValidation/RealStories-NarrativeProjections-withDemographics.csv")
MYTH_TYPES  = ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]
DEMO_COLS   = ["childhood_abuse", "adult_victim", "victim_female", "victim_male",
               "perpetrator_female", "perpetrator_male", "first_person_victim",
               "third_person_victim", "first_person_perpetrator", "third_person_perpetrator",
               "stranger_assault", "acquaintance_assault", "family_member", "intimate_partner"]
NLI_COLS    = MYTH_TYPES + DEMO_COLS

with open(EVAL_DIR / f"{EMB}.pkl", "rb") as f:
    subspace = pickle.load(f)
narr_proj = pd.read_csv(NARR_PATH)
results   = []

for source in SOURCES:
    task_label = "advice" if args.task == "AdviceGeneration" else "summary"
    for model in MODELS:
        csv_path = RESULTS_DIR / source / f"{model}_{task_label}_t1.csv"
        meta     = pd.read_csv(csv_path)
        meta["source"] = source
        if "prompt_variant" not in meta.columns:
            meta["prompt_variant"] = "default"

        t1_embs = pd.read_pickle(
            RESULTS_DIR / source / f"{EMB}" / f"{model}_{task_label}_t1_embs.pkl"
        )
        meta["proj_t1"] = t1_embs.apply(lambda v: np.dot(v, subspace)).values
        meta = meta.merge(narr_proj[["narrative_index"] + NLI_COLS],
                          left_on="narrative_idx", right_on="narrative_index", how="left")

        for col in NLI_COLS:
            for pv, grp in meta.groupby("prompt_variant"):
                for label, g in grp.groupby(col):
                    vals = g["proj_t1"].dropna().values
                    if len(vals) < 3:
                        continue
                    v = pd.Series(vals)
                    stat, p = shapiro(v.sample(min(len(v), 5000), random_state=42))
                    results.append({"source": source, "model": model, "col": col,
                                    "nli_label": label, "prompt_variant": pv,
                                    "n": len(vals), "mean": v.mean(), "std": v.std(),
                                    "skewness": v.skew(), "kurtosis": v.kurtosis(),
                                    "stat": stat, "p": p})

df = pd.DataFrame(results)
df["significant"], df["p_bh_corrected"], _, df["alpha"] = multipletests(df["p"].fillna(1.0), method="fdr_bh")
df.to_csv(f"StatisticalTest/RealStories-T1-{args.task}-Normality.csv", index=False)