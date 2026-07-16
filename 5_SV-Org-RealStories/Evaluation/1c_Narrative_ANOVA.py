"""One-way ANOVA with BH correction on narrative projection scores."""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import levene, f_oneway
from statsmodels.stats.multitest import multipletests
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", choices=["JustDetention", "SurvivorStories"], required=True)
parser.add_argument("--analysis", choices=["MYTHS", "DEMO"], required=True)
args = parser.parse_args()

df = pd.read_csv(f'../StatisticalTestResults/NarrativeProjections-{args.dataset}.csv')
assert df['projection_sbert'].notna().all()

MYTHS = ['clothing', 'perpetrator_intoxication', 'resistance', 'victim_intoxication']
DEMO  = ['childhood_abuse', 'adult_victim', 'victim_female', 'victim_male',
         'perpetrator_female', 'perpetrator_male',
         'stranger_assault', 'acquaintance_assault', 'family_member', 'intimate_partner']
NLI_LABELS = ['entailment', 'neutral', 'contradiction']

features = MYTHS if args.analysis == "MYTHS" else DEMO
out_dir  = Path(f"../StatisticalTestResults/{args.analysis}")
out_dir.mkdir(parents=True, exist_ok=True)

# ── Levene's Test ─────────────────────────────────────────────────────────────
levene_results = []
for feat in features:
    groups = [df.loc[df[feat] == label, 'projection_sbert'].values for label in NLI_LABELS]
    stat, p = levene(*groups)
    levene_results.append({'feature': feat, 'levene_stat': round(stat, 3), 'p_value': p})
levene_df = pd.DataFrame(levene_results)
levene_df['significant'], levene_df['p_bh_corrected'], _, levene_df['alpha'] = multipletests(levene_df['p_value'], method='fdr_bh')
levene_df.to_csv(out_dir / f"Narrative-LeveneTest-{args.dataset}.csv", index=False)

equal_var_features = set(levene_df.loc[~levene_df['significant'], 'feature'])

# ── ANOVA ─────────────────────────────────────────────────────────────────────
anova_results = []
for feat in features:
    groups = [df.loc[df[feat] == label, 'projection_sbert'].values for label in NLI_LABELS]
    ns     = [len(g) for g in groups]
    if any(n == 0 for n in ns):
        anova_results.append({'feature': feat, 'anova_type': None, 'stat': None, 'p_value': None,
                              'eta2': None, 'n_entailment': ns[0], 'n_neutral': ns[1], 'n_contradiction': ns[2]})
        continue
    n, k = sum(ns), len(groups)
    if feat in equal_var_features:
        stat, p    = f_oneway(*groups)
        eta2       = max((stat * (k - 1)) / (stat * (k - 1) + (n - k)), 0)
        anova_type = "one-way"
    else:
        stat, p    = stats.alexandergovern(*groups)
        eta2       = None
        anova_type = "welch"
    anova_results.append({'feature': feat, 'anova_type': anova_type, 'stat': round(stat, 3), 'p_value': p,
                          'eta2': round(eta2, 4) if eta2 is not None else None,
                          'n_entailment': ns[0], 'n_neutral': ns[1], 'n_contradiction': ns[2]})
    
anova_df = pd.DataFrame(anova_results)
anova_df['significant'], anova_df['p_bh_corrected'], _, anova_df['alpha'] = multipletests(anova_df['p_value'], method='fdr_bh')
anova_df.to_csv(out_dir / f"Narrative-ANOVA-{args.dataset}.csv", index=False)