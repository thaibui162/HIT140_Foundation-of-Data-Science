# 2_data_analyse.py - Preparation, descriptive statistics, confidence interval and two-sample t-test.

import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).resolve().parent
gk = pd.read_csv(ROOT / "processed" / "gk_wrangled.csv")

ALPHA = 0.05

# PART 1 - DATA PREPARATION AND SAMPLING
# ----------------------------------------------------------------
print("=" * 60)
print("PART 1 - DATA PREPARATION AND SAMPLING")
print("=" * 60)
print("starting rows:", len(gk))

# Filter 1: save% needs at least one shot on target, or it has no meaning.
no_shots = ~gk["has_save_pct"]
print("\nremoved - faced no shots on target:", no_shots.sum())
print(gk.loc[no_shots, ["player", "team", "gk_minutes"]].to_string(index=False))

# Filter 2: one goalkeeper per team, the one with the most minutes.
# Most minutes, then most shots faced
print("\nremoved - backup keepers:", int((~gk["is_primary_gk"] & ~no_shots).sum()))

gk = gk[gk["has_save_pct"] & gk["is_primary_gk"]].copy()

print("\nfinal sample:", len(gk))
print(gk["group_label"].value_counts())
print("shortest time played in the sample:", gk["gk_minutes"].min(), "minutes")

advanced = gk.loc[gk["group_label"] == "Advanced", "gk_save_pct"]
eliminated = gk.loc[gk["group_label"] == "Eliminated", "gk_save_pct"]

gk.to_csv(ROOT / "processed" / "gk_sample.csv", index=False, encoding="utf-8-sig")
print("\nSaved: processed/gk_sample.csv")


# PART 2 - DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 2 - DESCRIPTIVE STATISTICS (save %)")
print("=" * 60)

summary = gk.groupby("group_label")["gk_save_pct"].agg(
    n="count", mean="mean", median="median", std="std",
    min="min", max="max")
summary["IQR"] = gk.groupby("group_label")["gk_save_pct"].quantile(0.75) - \
                 gk.groupby("group_label")["gk_save_pct"].quantile(0.25)
print(summary.round(2))


# PART 3 - CHECKING THE t-TEST ASSUMPTIONS
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 3 - ASSUMPTION CHECKS")
print("=" * 60)

print("Shapiro-Wilk normality test (p > 0.05 means normal enough): ")
print("Advanced: p = ", round(stats.shapiro(advanced).pvalue, 4))
print("Eliminated: p = ", round(stats.shapiro(eliminated).pvalue, 4))

levene_p = stats.levene(advanced, eliminated).pvalue
print("Levene equal-variance test: p = ", round(levene_p, 4))
equal_var = levene_p > 0.05
print("->", "equal variances, use Student's t-test" 
      if equal_var
      else "unequal variances, use Welch's t-test")


# PART 4 - CONFIDENCE INTERVALS
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 4 - 95% CONFIDENCE INTERVALS")
print("=" * 60)

for name, group in [("Advanced", advanced), ("Eliminated", eliminated)]:
    n = len(group)
    se = group.std(ddof=1) / np.sqrt(n)         
    margin = stats.t.ppf(0.975, n - 1) * se     
    print(name.ljust(11), "mean =", round(group.mean(), 2), " 95% CI = [", round(group.mean() - margin, 2), ",",
          round(group.mean() + margin, 2), "]  n =", n)

# Confidence interval for the difference between the two means
n1, n2 = len(advanced), len(eliminated)
v1, v2 = advanced.var(ddof=1), eliminated.var(ddof=1)
difference = advanced.mean() - eliminated.mean()

pooled_var = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
if equal_var:
    se_diff = np.sqrt(pooled_var * (1 / n1 + 1 / n2))
    df = n1 + n2 - 2
else:
    se_diff = np.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
margin_diff = stats.t.ppf(0.975, df) * se_diff

print("\nDifference (Advanced - Eliminated) =", round(difference, 2))
print("95% CI = [", round(difference - margin_diff, 2), ",", round(difference + margin_diff, 2), "]")
print("The interval does not contain 0, so the two groups differ."
      if (difference - margin_diff) * (difference + margin_diff) > 0
      else "The interval contains 0, so we cannot say the groups differ.")


# PART 5 - TWO-SAMPLE t-TEST
# ----------------------------------------------------------------
print("\n" + "=" * 60)
print("PART 5 - TWO-SAMPLE t-TEST")
print("=" * 60)
print("H0: the two groups have the same mean save percentage")
print("H1: they do not (two-tailed, alpha = 0.05)")

t_stat, p_value = stats.ttest_ind(advanced, eliminated, equal_var=equal_var)
cohens_d = difference / np.sqrt(pooled_var)

print("\nt =", round(t_stat, 4))
print("p =", round(p_value, 6))
print("Cohen's d =", round(cohens_d, 3))
print("\nResult:", "REJECT H0" if p_value < ALPHA else "DO NOT reject H0")