import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
from itertools import combinations
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["JustDetention", "SurvivorStories"], required=True)
parser.add_argument("--analysis", choices=["MYTHS", "DEMO"], required=True)
args = parser.parse_args()

df = pd.read_csv(f'../StatisticalTestResults/NarrativeProjections-{args.dataset}.csv')
assert df['projection_sbert'].notna().all(), "Null values found in projection_sbert"

MYTHS = ['clothing', 'perpetrator_intoxication', 'resistance', 'victim_intoxication']
DEMO  = ['childhood_abuse', 'adult_victim', 'victim_female', 'victim_male',
         'perpetrator_female', 'perpetrator_male', 'first_person_victim',
         'third_person_victim', 'first_person_perpetrator', 'third_person_perpetrator',
         'stranger_assault', 'acquaintance_assault', 'family_member', 'intimate_partner']
NLI_LABELS = ['entailment', 'neutral', 'contradiction']

features = MYTHS if args.analysis == "MYTHS" else DEMO
out_dir  = Path(f"../StatisticalTestResults/{args.analysis}")
out_dir.mkdir(parents=True, exist_ok=True)
       
# Shapiro-Wilk with Benjamini-Hochberg (BH)
results = []
for feat in features:
    for label in NLI_LABELS:
        vals = df.loc[df[feat] == label, 'projection_sbert']
        if len(vals) < 3:
            results.append({'feature': feat, 'label': label, 'n': len(vals),
                            'mean': None, 'std': None, 'skewness': None, 'kurtosis': None, 'p_value': None})
            continue
        stat, p = stats.shapiro(vals)
        results.append({'feature': feat, 'label': label, 'n': len(vals),
                        'mean': round(vals.mean(), 3), 'std': round(vals.std(), 3),
                        'skewness': round(vals.skew(), 3), 'kurtosis': round(vals.kurtosis(), 3),
                        'p_value': p})
shapiro_df = pd.DataFrame(results)
shapiro_df["significant"], shapiro_df["p_bh_corrected"], _, shapiro_df["alpha"] = multipletests(shapiro_df["p_value"], method="fdr_bh")
shapiro_df.to_csv(out_dir / f"Narrative-NormalityCheck-{args.dataset}.csv", index=False)