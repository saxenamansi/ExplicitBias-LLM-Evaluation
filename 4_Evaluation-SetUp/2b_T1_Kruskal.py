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
results   = []

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
            stat, p = kruskal(*label_groups.values())
            k = len(label_groups)
            n = sum(len(v) for v in label_groups.values())
            eta2 = max((stat - k + 1) / (n - k), 0)
            results.append({"model": model, "col": col, "prompt_variant": pv,
                             "n": n, "k": k, "stat": stat, "p_value": p, "eta2": eta2})

df = pd.DataFrame(results)
df["significant"], df["p_bh_corrected"], _, df["alpha"] = multipletests(df["p_value"], method="fdr_bh")
df.to_csv(out_dir / f"T1{suffix}-Kruskal.csv", index=False)