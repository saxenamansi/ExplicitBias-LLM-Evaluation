"""Pairwise Dunn tests on T1 subspace projections, per grouping column."""
import argparse
import ast
import numpy as np
import pandas as pd
import pickle
import scikit_posthocs as sp
from pathlib import Path
from scipy import stats
from itertools import combinations
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
results = []

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
            data   = np.concatenate(list(label_groups.values()))
            labels = np.concatenate([[k] * len(v) for k, v in label_groups.items()])
            dunn_df = sp.posthoc_dunn(
                pd.DataFrame({'value': data, 'group': labels}),
                val_col='value', group_col='group', p_adjust=None
            )
            n_total = len(data)
            for l1, l2 in combinations(label_groups.keys(), 2):
                p = dunn_df.loc[l1, l2]
                z = np.abs(stats.norm.ppf(p / 2))
                r = z / np.sqrt(n_total)
                results.append({"model": model, "grouping": grouping, "col": col,
                                 "prompt_variant": pv, "comparison": f"{l1}_vs_{l2}",
                                 "n": n_total, "p_value": p, "rank_biserial": round(r, 4)})

df = pd.DataFrame(results)
df["significant"], df["p_bh_corrected"], _, df["alpha"] = multipletests(df["p_value"], method="fdr_bh")
df.to_csv(f"StatisticalTestResults/T1-{args.task}-Dunn.csv", index=False)
