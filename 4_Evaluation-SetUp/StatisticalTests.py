"""
StatisticalTests.py
===================
Modular statistical test helpers.
"""

import warnings
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import wilcoxon, ttest_rel, ttest_ind, mannwhitneyu
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── Normality ─────────────────────────────────────────────────────────────────

def check_normality(x: list | np.ndarray, p_value_threshold: float = 0.05) -> bool:
    """Shapiro-Wilk normality test. Returns True if normal (p-value > threshold)."""
    x = np.array(x, dtype=float)
    if len(x) < 3: # dataset size must be at least 3
        return False
    _, p = stats.shapiro(x[:5000])
    return p > p_value_threshold


# ── Paired tests ──────────────────────────────────────────────────────────────

def paired_ttest(a: list, b: list) -> dict:
    """
    Paired t-test for two related samples.
    Returns stat, p, n, Cohen's dz/dav/drm, Hedges' g.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    stat, p = ttest_rel(a, b)
    return _paired_effect_sizes(a, b, stat, p, "paired_t_test")


def wilcoxon_signed_rank(a: list, b: list) -> dict:
    """
    Wilcoxon signed-rank test for two related non-normal samples.
    Returns stat, p, n, Cohen's dz/dav/drm, Hedges' g.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    diff = a - b
    stat, p = wilcoxon(diff)
    return _paired_effect_sizes(a, b, stat, p, "wilcoxon_signed_rank")


def paired_test(a: list, b: list) -> dict:
    """
    One-sample test of differences in paired values against null hypothesis H0: mu=0. (difference = 0)
    Auto-selects paired_ttest (normal differences) or wilcoxon_signed_rank.

    Here a is difference in paired scores (within subject), and b is 0.
    """
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    if check_normality(a_arr - b_arr): # a is
        return paired_ttest(a, b)
    return wilcoxon_signed_rank(a, b)


def _paired_effect_sizes(a, b, stat, p, test_name) -> dict:
    diff      = a - b
    n         = len(diff)
    mean_d    = np.mean(diff)
    sd_d      = np.std(diff, ddof=1)
    sd_a      = np.std(a, ddof=1)
    sd_b      = np.std(b, ddof=1)
    sd_pooled = np.sqrt((sd_a**2 + sd_b**2) / 2)
    r_ab      = np.corrcoef(a, b)[0, 1]
    dz        = mean_d / sd_d if sd_d > 0 else np.nan
    dav       = mean_d / ((sd_a + sd_b) / 2) if (sd_a + sd_b) > 0 else np.nan
    drm       = mean_d / (sd_pooled * np.sqrt(2 * (1 - r_ab))) if sd_pooled > 0 else np.nan
    cf        = 1 - (3 / (4 * (n - 1) - 1))
    g         = dz * cf if not np.isnan(dz) else np.nan
    se        = sd_d / np.sqrt(n)
    ci        = stats.t.interval(0.95, df=n-1, loc=mean_d, scale=se)
    cl        = stats.norm.cdf(dz / np.sqrt(2)) if not np.isnan(dz) else np.nan
    glass_delta = mean_d / sd_a if sd_a > 0 else np.nan  # sd_a = sd of t1 (pre)
    return dict(
        test=test_name, stat=stat, p=p,
        normal=check_normality(a - b), n=n,
        mean1=np.mean(a), mean2=np.mean(b), mdiff=mean_d,
        sd1=sd_a, sd2=sd_b, sdiff=sd_d, r=r_ab, se_diff=se,
        cohens_dz=dz, cohens_dav=dav, cohens_drm=drm, hedges_g=g,
        ci_low=ci[0], ci_high=ci[1], cl_effect_size=cl, glass_delta = glass_delta
    )


# ── Independent tests ─────────────────────────────────────────────────────────

def independent_ttest(a: list, b: list) -> dict:
    """
    Independent samples t-test.
    Returns stat, p, n_a, n_b, Cohen's d, Hedges' g.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    stat, p = ttest_ind(a, b)
    return _independent_effect_sizes(a, b, stat, p, "independent_t_test")


def mann_whitney(a: list, b: list) -> dict:
    """
    Mann-Whitney U (Wilcoxon rank-sum) test for two independent non-normal samples.
    Returns stat, p, n_a, n_b, Cohen's d, Hedges' g.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    return _independent_effect_sizes(a, b, stat, p, "wilcoxon_rank_sum")


def independent_test(a: list, b: list) -> dict:
    """
    Auto-selects independent_ttest (both normal) or mann_whitney.
    """
    a_arr, b_arr = np.array(a, dtype=float), np.array(b, dtype=float)
    if check_normality(a_arr) and check_normality(b_arr):
        return independent_ttest(a, b)
    return mann_whitney(a, b)


def _independent_effect_sizes(a, b, stat, p, test_name) -> dict:
    na, nb    = len(a), len(b)
    sd_a      = np.std(a, ddof=1)
    sd_b      = np.std(b, ddof=1)
    sd_pooled = (
        np.sqrt(((na - 1) * sd_a**2 + (nb - 1) * sd_b**2) / (na + nb - 2))
        if na + nb > 2 else np.nan
    )
    d  = (np.mean(a) - np.mean(b)) / sd_pooled if sd_pooled and sd_pooled > 0 else np.nan
    cf = 1 - (3 / (4 * (na + nb - 2) - 1)) if na + nb > 4 else np.nan
    g  = d * cf if not (np.isnan(d) or np.isnan(cf)) else np.nan
    return dict(
        test=test_name, stat=stat, p=p,
        normal=(check_normality(a) and check_normality(b)),
        n_a=na, n_b=nb, cohens_d=d, hedges_g=g,
    )

# ── Benjamini–Hochberg correction ──────────────────────────────────────────────

def apply_bh_correction(results: list[dict], p_col: str = "p",
                         alpha: float = 0.05) -> list[dict]:
    """
    Apply Benjamini-Hochberg FDR correction to a list of test result dicts.
    Adds two fields to each dict:
      p_bh       — BH-corrected p-value
      significant — True if p_bh < alpha
    Returns the same list with fields added in-place.
    """
    if not results:
        return results
    pvals = [r[p_col] for r in results]
    _, p_corrected, _, _ = multipletests(pvals, alpha=alpha, method="fdr_bh")
    for r, pc in zip(results, p_corrected):
        r["p_bh"]        = pc
        r["significant"] = pc < alpha
    return results
    
# ── VIF ───────────────────────────────────────────────────────────────────────

def compute_vif(df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance inflation factors.
    Casts to float64 before passing to statsmodels to avoid TypeError crash.
    """
    df = df_features.copy().dropna()
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    df = df.loc[:, df.std() > 0]
    if df.shape[1] < 2:
        return pd.DataFrame({"feature": df.columns,
                             "VIF":     [np.nan] * df.shape[1]})
    vif_data = pd.DataFrame()
    vif_data["feature"] = df.columns
    vif_data["VIF"] = [
        variance_inflation_factor(df.values.astype(np.float64), i)
        for i in range(df.shape[1])
    ]
    return vif_data


# ── ANOVA ─────────────────────────────────────────────────────────────────────

def run_anova(df: pd.DataFrame, metric_col: str,
              factor_cols: list, demo_cols: list) -> pd.DataFrame | None:
    """
    Type-II ANOVA. factor_cols treated as categorical; demo_cols included
    if nunique > 1. Returns anova table or None if failed.
    CHECK: verify factor_cols and demo_cols match your actual DataFrame column names.
    """
    sub = df.dropna(subset=[metric_col]).copy()
    if len(sub) < 10:
        return None
    formula_parts = [f"C({c})" for c in factor_cols]
    for dc in demo_cols:
        if dc in sub.columns and sub[dc].nunique() > 1:
            formula_parts.append(f"C({dc})")
    formula = f"{metric_col} ~ " + " + ".join(formula_parts)
    try:
        lm = ols(formula, data=sub).fit()
        return anova_lm(lm, typ=2)
    except Exception as e:
        print(f"    ANOVA failed for {metric_col}: {e}")
        return None


# ── Linear regression ─────────────────────────────────────────────────────────

def run_regression(reg_df: pd.DataFrame, feat_cols: list,
                   effect_col: str, metric_name: str, vif_threshold: int = 10) -> pd.DataFrame | None:
    """
    Linear regression predicting effect_col from feat_cols.
    Drops high-VIF features (>10) before fitting.
    Returns coefficient summary or None if insufficient data.
    CHECK: confirm feat_cols are all numeric and present in reg_df.
    """
    sub = reg_df.dropna(subset=[effect_col]).copy()
    if len(sub) < 5:
        return None
    feat_cols = [c for c in feat_cols
                 if c in sub.columns and sub[c].std() > 0]
    if not feat_cols:
        return None
    X        = sub[feat_cols].fillna(0)
    vif_df   = compute_vif(X)
    high_vif = vif_df[vif_df["VIF"] > vif_threshold]["feature"].tolist()
    X        = X.drop(columns=high_vif)
    feat_cols = [c for c in feat_cols if c not in high_vif]
    if X.shape[1] == 0:
        return None
    X_scaled = StandardScaler().fit_transform(X.astype(np.float64))
    reg      = LinearRegression().fit(X_scaled, sub[effect_col])
    summary  = pd.DataFrame({"feature": feat_cols, "coefficient": reg.coef_})
    summary["metric"]    = metric_name
    summary["target"]    = effect_col
    summary["r_squared"] = reg.score(X_scaled, sub[effect_col])
    summary["n"]         = len(sub)
    return summary