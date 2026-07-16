"""Shapiro-Wilk normality tests on T1 subspace projections, per label group."""
import argparse
import ast
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import shapiro
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["summary", "advice"], required=True)
parser.add_argument("--dataset", choices=["JustDetention", "SurvivorStories"], required=True)
parser.add_argument("--analysis", choices=["MYTHS", "DEMO"], default="MYTHS")
args = parser.parse_args()

EMB_DIR = Path(f"../Results/{args.dataset}/1_Embeddings/")
NARR_PATH = Path(f"../StatisticalTestResults/NarrativeProjections-{args.dataset}.csv")

MODELS     = ["gemma", "llama", "mistral", "qwen"]
EMB        = "SBERT"
EVAL_DIR   = Path("../../4_Evaluation-SetUp/MythSubspaces")
MYTH_TYPES = ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]
DEMO_COLS  = ["childhood_abuse", "adult_victim", "victim_female", "victim_male",
              "perpetrator_female", "perpetrator_male", "first_person_victim",
              "third_person_victim", "first_person_perpetrator", "third_person_perpetrator",
              "stranger_assault", "acquaintance_assault", "family_member", "intimate_partner"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]

features = MYTH_TYPES if args.analysis == "MYTHS" else DEMO_COLS
out_dir  = Path(f"../StatisticalTestResults/{args.analysis}")
out_dir.mkdir(parents=True, exist_ok=True)

with open(EVAL_DIR / f"{EMB}.pkl", "rb") as f:
    subspace = pickle.load(f)
narr_proj = pd.read_csv(NARR_PATH)
results   = []

for model in MODELS:
    # t1_embs = pd.read_pickle(EMB_DIR / EMB / f"{model}_t1.pkl")
    # meta    = pd.read_csv(EMB_DIR / f"{model}_t1_metadata.csv")
    
    t1_embs = pd.read_pickle(EMB_DIR / "SBERT" / f"{model}_{args.task}_t1.pkl")
    meta    = pd.read_csv(EMB_DIR / f"{model}_{args.task}_t1_metadata.csv")
    
    if "prompt_variant" not in meta.columns:
        meta["prompt_variant"] = "default"
    meta["narrative_idx"] = meta["narrative_idx"].astype(int)
    meta["proj_t1"] = t1_embs.apply(lambda v: np.dot(v, subspace)).values
    assert meta["proj_t1"].notna().all(), f"Null projections found for {model}"
    parsed = meta["narrative_nli_label"].apply(ast.literal_eval)
    check  = meta[["narrative_idx"]].copy()
    for m in MYTH_TYPES:
        check[m] = parsed.apply(lambda d: d.get(m, None))
    check = check.merge(narr_proj[["narrative_index"] + MYTH_TYPES],
                        left_on="narrative_idx", right_on="narrative_index", how="left")
    for m in MYTH_TYPES:
        assert (check[f"{m}_x"] == check[f"{m}_y"]).all(), f"NLI mismatch for {m}"
    meta = meta.merge(narr_proj[["narrative_index"] + MYTH_TYPES + DEMO_COLS],
                      left_on="narrative_idx", right_on="narrative_index", how="left")

    for col in features:
        for pv, grp in meta.groupby("prompt_variant"):
            for label in NLI_LABELS:
                vals = grp.loc[grp[col] == label, "proj_t1"].values
                v = pd.Series(vals)
                if len(v) < 3:
                    results.append({"model": model, "col": col, "nli_label": label, "prompt_variant": pv,
                                    "n": len(v), "mean": None, "std": None,
                                    "skewness": None, "kurtosis": None, "stat": None, "p_value": None})
                    continue
                stat, p = shapiro(v)
                results.append({"model": model, "col": col, "nli_label": label, "prompt_variant": pv,
                                "n": len(v), "mean": v.mean(), "std": v.std(),
                                "skewness": v.skew(), "kurtosis": v.kurtosis(), "stat": stat, "p_value": p})

df = pd.DataFrame(results)
valid = df["p_value"].notna()
df.loc[valid, ["significant", "p_bh_corrected", "_", "alpha"]] = \
    np.array(multipletests(df.loc[valid, "p_value"], method="fdr_bh"), dtype=object).T
df.to_csv(out_dir / f"T1-Normality-{args.dataset}-{args.task}.csv", index=False)