"""Kruskal-Wallis tests on T1 subspace projections, per grouping column."""
import argparse
import ast
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["Summarization", "AdviceGeneration"], required=True)
args = parser.parse_args()

MODELS     = ["gemma", "llama", "mistral", "qwen"]
EMB        = "SBERT"
EMB_DIR    = Path(f"../Results/{args.task}/1_Embeddings/") # Change to your path
EVAL_DIR   = Path("MythSubspaces") # Change to your path
NARR_PATH  = Path("ProjectionValidation/NarrativeProjections-withDemographics.csv") # Change to your path
MYTH_TYPES = ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]
DEMO_COLS  = ["childhood_abuse", "adult_victim", "victim_female", "victim_male",
              "perpetrator_female", "perpetrator_male", "first_person_victim",
              "third_person_victim", "first_person_perpetrator", "third_person_perpetrator",
              "stranger_assault", "acquaintance_assault", "family_member", "intimate_partner"]
NLI_COLS   = MYTH_TYPES + DEMO_COLS
NLI_LABELS = ["entailment", "neutral", "contradiction"]

with open(EVAL_DIR / f"{EMB}.pkl", "rb") as f: # Change to your path
    subspace = pickle.load(f)
narr_proj = pd.read_csv(NARR_PATH)
results   = []

for model in MODELS:
    t1_embs = pd.read_pickle(EMB_DIR / EMB / f"{model}_t1.pkl") # Change to your path
    meta    = pd.read_csv(EMB_DIR / f"{model}_t1_metadata.csv") # Change to your path

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

    meta = meta.merge(narr_proj[["narrative_index"] + NLI_COLS],
                      left_on="narrative_idx", right_on="narrative_index", how="left")

    for col in NLI_COLS:
        grouping = "nli" if col in MYTH_TYPES else "demographic"
        for pv, grp in meta.groupby("prompt_variant"):
            label_groups = {k: grp.loc[grp[col] == k, "proj_t1"].values for k in NLI_LABELS}
            stat, p = kruskal(*label_groups.values())
            k = len(label_groups)
            n = sum(len(v) for v in label_groups.values())
            eta2 = max((stat - k + 1) / (n - k), 0)
            results.append({"model": model, "grouping": grouping, "col": col,
                             "prompt_variant": pv, "n": n, "k": k,
                             "stat": stat, "p_value": p, "eta2": eta2})

df = pd.DataFrame(results)
df["significant"], df["p_bh_corrected"], _, df["alpha"] = multipletests(df["p_value"], method="fdr_bh")
df.to_csv(f"StatisticalTestResults/T1-{args.task}-Kruskal.csv", index=False)
