import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import kruskal
from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp
from itertools import combinations
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--analysis", choices=["MYTHS", "DEMO"], required=True)
args = parser.parse_args()

df = pd.read_csv('ProjectionValidation/NarrativeProjections-withDemographics.csv')
assert df['projection_sbert'].notna().all(), "Null values found in projection_sbert"

MYTHS = ['clothing', 'perpetrator_intoxication', 'resistance', 'victim_intoxication']
DEMO  = ['childhood_abuse', 'adult_victim', 'victim_female', 'victim_male',
         'perpetrator_female', 'perpetrator_male', 
         # 'first_person_victim',
         # 'third_person_victim', 'first_person_perpetrator', 'third_person_perpetrator',
         'stranger_assault', 'acquaintance_assault', 'family_member', 'intimate_partner']
NLI_LABELS = ['entailment', 'neutral', 'contradiction']

features = MYTHS if args.analysis == "MYTHS" else DEMO
out_dir  = Path(f"StatisticalTestResults/{args.analysis}")
out_dir.mkdir(parents=True, exist_ok=True)

# Shapiro-Wilk with Benjamini-Hochberg (BH)
results = []
for feat in features:
    for label in NLI_LABELS:
        vals = df.loc[df[feat] == label, 'projection_sbert']
        stat, p = stats.shapiro(vals)
        results.append({'feature': feat, 'label': label, 'n': len(vals),
                        'mean': round(vals.mean(), 3), 'std': round(vals.std(), 3),
                        'skewness': round(vals.skew(), 3), 'kurtosis': round(vals.kurtosis(), 3),
                        'p_value': p})
shapiro_df = pd.DataFrame(results)
shapiro_df["significant"], shapiro_df["p_bh_corrected"], _, shapiro_df["alpha"] = multipletests(shapiro_df["p_value"], method="fdr_bh")
shapiro_df.to_csv(out_dir / "Narrative-NormalityCheck-noPOV.csv", index=False)

# Kruskal-Wallis with Benjamini-Hochberg (BH)
results = []
for feat in features:
    groups = [df.loc[df[feat] == label, 'projection_sbert'].values for label in NLI_LABELS]
    stat, p = kruskal(*groups)
    n, k = sum(len(g) for g in groups), len(groups)
    eta2 = max((stat - k + 1) / (n - k), 0)
    results.append({'feature': feat, 'kruskal_stat': round(stat, 3), 'p_value': p, 'eta2': round(eta2, 4)})
kruskal_df = pd.DataFrame(results)
kruskal_df['significant'], kruskal_df['p_bh_corrected'], _, kruskal_df['alpha'] = multipletests(kruskal_df['p_value'], method='fdr_bh')
kruskal_df.to_csv(out_dir / "Narrative-KruskalWallis-noPOV.csv", index=False)

# Dunn Test with Benjamini-Hochberg (BH)
results = []
for feat in features:
    groups = {label: df.loc[df[feat] == label, 'projection_sbert'].values for label in NLI_LABELS}
    data   = np.concatenate(list(groups.values()))
    labels = np.concatenate([[label] * len(v) for label, v in groups.items()])
    dunn_df = sp.posthoc_dunn(
        pd.DataFrame({'value': data, 'group': labels}),
        val_col='value', group_col='group', p_adjust=None
    )
    n_total = len(data)
    for l1, l2 in combinations(NLI_LABELS, 2):
        p = dunn_df.loc[l1, l2]
        z = np.abs(stats.norm.ppf(p / 2))
        med1 = np.median(groups[l1])
        med2 = np.median(groups[l2])
        r = (1 if med1 > med2 else -1) * z / np.sqrt(n_total)
        results.append({'feature': feat, 'pair': f'{l1} vs {l2}', 'p_value': p, 'rank_biserial': round(r, 4)})
dunn_res_df = pd.DataFrame(results)
dunn_res_df['significant'], dunn_res_df['p_bh_corrected'], _, dunn_res_df['alpha'] = multipletests(dunn_res_df['p_value'], method='fdr_bh')
dunn_res_df.to_csv(out_dir / "Narrative-DunnTest-noPOV.csv", index=False)