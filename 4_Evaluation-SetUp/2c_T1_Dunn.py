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
parser.add_argument("--source", choices=["reddit", "JustDetention", "SurvivorStories"], default="reddit")
parser.add_argument("--analysis", choices=["MYTHS", "DEMO"], default="MYTHS")
args = parser.parse_args()

SOURCE_PATHS = {
    "reddit":          (Path(f"../Results/{args.task}/1_Embeddings_1stPOV/"),
                        Path("ProjectionValidation/NarrativeProjections-withDemographics.csv")),
    "JustDetention":   (Path(f"../Results/JustDetention/1_Embeddings/"),
                        Path("ProjectionValidation/RealStories-NarrativeProjections-withDemographics.csv")),
    "SurvivorStories": (Path(f"../Results/SurvivorStories/1_Embeddings/"),
                        Path("ProjectionValidation/RealStories-NarrativeProjections-withDemographics.csv")),
}
EMB_DIR, NARR_PATH = SOURCE_PATHS[args.source]

MODELS     = ["gemma", "llama", "mistral", "qwen"]
EMB        = "SBERT"
EVAL_DIR   = Path("MythSubspaces")
MYTH_TYPES = ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]
DEMO_COLS  = ["childhood_abuse", "adult_victim", "victim_female", "victim_male",
              "perpetrator_female", "perpetrator_male", "first_person_victim",
              "third_person_victim", "first_person_perpetrator", "third_person_perpetrator",
              "stranger_assault", "acquaintance_assault", "family_member", "intimate_partner"]
NLI_LABELS = ["entailment", "neutral", "contradiction"]

features = MYTH_TYPES if args.analysis == "MYTHS" else DEMO_COLS
suffix   = f"-{args.task}_1stPOV" if args.source == "reddit" else f"-{args.source}-{args.task}"
out_dir  = Path(f"StatisticalTestResults/1stPOV/{args.analysis}")
out_dir.mkdir(parents=True, exist_ok=True)

with open(EVAL_DIR / f"{EMB}.pkl", "rb") as f:
    subspace = pickle.load(f)
narr_proj = pd.read_csv(NARR_PATH)
results = []

for model in MODELS:
    t1_embs = pd.read_pickle(EMB_DIR / EMB / f"{model}_t1.pkl")
    meta    = pd.read_csv(EMB_DIR / f"{model}_t1_metadata.csv")
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
                med1 = np.median(label_groups[l1])
                med2 = np.median(label_groups[l2])
                r = (1 if med1 > med2 else -1) * z / np.sqrt(n_total)
                results.append({"model": model, "col": col,
                                 "prompt_variant": pv, "comparison": f"{l1}_vs_{l2}",
                                 "n": n_total, "p_value": p, "rank_biserial": round(r, 4)})

df = pd.DataFrame(results)
df["significant"], df["p_bh_corrected"], _, df["alpha"] = multipletests(df["p_value"], method="fdr_bh")
df.to_csv(out_dir / f"T1{suffix}-Dunn.csv", index=False)