import argparse
import ast
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.stats import shapiro, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument("--task", choices=["Summarization", "AdviceGeneration"], required=True)
args = parser.parse_args()

MODELS     = ["gemma", "llama", "mistral", "qwen"]
EMB        = "SBERT"
EMB_DIR    = Path(f"../Results/{args.task}/1_Embeddings/")
EVAL_DIR   = Path("MythSubspaces")
NARR_PATH  = Path("ProjectionValidation/NarrativeProjections-withDemographics.csv")
MYTH_TYPES = ["clothing", "victim_intoxication", "perpetrator_intoxication", "resistance"]
DEMO_COLS  = ["childhood_abuse", "adult_victim", "victim_female", "victim_male",
              "perpetrator_female", "perpetrator_male", "first_person_victim",
              "third_person_victim", "first_person_perpetrator", "third_person_perpetrator",
              "stranger_assault", "acquaintance_assault", "family_member", "intimate_partner"]
NLI_COLS   = MYTH_TYPES + DEMO_COLS

def load_subspace():
    with open(EVAL_DIR / f"{EMB}.pkl", "rb") as f:
        return pickle.load(f)

subspace  = load_subspace()
narr_proj = pd.read_csv(NARR_PATH)
results   = []

for model in MODELS:
    t1_embs = pd.read_pickle(EMB_DIR / EMB / f"{model}_t1.pkl")
    meta    = pd.read_csv(EMB_DIR / f"{model}_t1_metadata.csv")

    if "prompt_variant" not in meta.columns:
        meta["prompt_variant"] = "default"

    meta["narrative_idx"] = meta["narrative_idx"].astype(int)
    meta["proj_t1"] = t1_embs.apply(lambda v: np.dot(v, subspace)).values

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
            label_groups = {k: g["proj_t1"].dropna().values
                            for k, g in grp.groupby(col) if len(g) >= 3}
            if len(label_groups) < 2:
                continue

            # normality per label Shapiro-Wilk
            for label, vals in label_groups.items():
                v = pd.Series(vals)
                stat, p = shapiro(v.sample(min(len(v), 5000), random_state=42))
                results.append({"model": model, "test": "normality", "grouping": grouping,
                                "col": col, "nli_label": label, "prompt_variant": pv,
                                "comparison": None, "n": len(vals), "mean": v.mean(),
                                "std": v.std(), "skewness": v.skew(), "kurtosis": v.kurtosis(),
                                "stat": stat, "p": p})

            # Kruskal Wallis
            stat, p = kruskal(*label_groups.values())
            k    = len(label_groups)
            n    = sum(len(v) for v in label_groups.values())
            eta2 = max((stat - k + 1) / (n - k), 0)
            results.append({"model": model, "test": "kruskal", "grouping": grouping,
                            "col": col, "nli_label": None, "prompt_variant": pv,
                            "comparison": None, "n": sum(len(v) for v in label_groups.values()),
                            "mean": None, "std": None, "skewness": None, "kurtosis": None,
                            "stat": stat, "p": p, "eta2": eta2})

            # Mann-Whitney (pairwise)
            for (l1, v1), (l2, v2) in combinations(label_groups.items(), 2):
                stat, p = mannwhitneyu(v1, v2, alternative="two-sided")
                rb = 1 - (2 * stat) / (len(v1) * len(v2))
                results.append({"model": model, "test": "mannwhitney", "grouping": grouping,
                                "col": col, "nli_label": None, "prompt_variant": pv,
                                "comparison": f"{l1}_vs_{l2}", "n": len(v1) + len(v2),
                                "mean": None, "std": None, "skewness": None, "kurtosis": None,
                                "stat": stat, "p": p, "rank_biserial": rb})

df = pd.DataFrame(results)
df["significant"], df["p_bh"], _, df["alpha"] = multipletests(df["p"].fillna(1.0), method="fdr_bh")
df.to_csv(f"T1-{args.task}.csv", index=False)