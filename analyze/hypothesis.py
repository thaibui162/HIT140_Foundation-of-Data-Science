import os
import statistics as stats
import math
import pandas as pd
import scipy.stats as st

# one seed for every sample drawn in this project, so the code, the output
# and the slides all agree
SEED = 2026

# read data 

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")

distance = pd.read_csv(os.path.join(DATA, "processed", "wc2026_team_distance_CLEAN.csv"))
duration = pd.read_csv(os.path.join(DATA, "reference", "match_time_played.csv"))

df = distance[["match_id", "result_id", "date", "stage", "team", "distance_km"]].merge(
    duration[["match_id", "time_played_min"]], on="match_id", how="left")

# distance per 90 minutes
df["distance_per_90"] = df["distance_km"] / df["time_played_min"] * 90

# split the two stages. Anything that is not "First Stage" is knockout,
# including the third-place play-off.
group_stage = df[df["stage"] == "First Stage"]
elim_stage = df[df["stage"] != "First Stage"]

group_dist = group_stage["distance_per_90"]
elim_dist = elim_stage["distance_per_90"]

# DESCRIPTIVE STATISTICS PER GROUP 

def show_stats(label, values, unit="km/90"):
    """Print mean, median, range, quartiles, IQR, variance, sd and outliers."""

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = values[(values < lower_bound) | (values > upper_bound)]

    print("\n%s  (n = %d)" % (label, len(values)))
    print("  mean            : %.2f %s" % (values.mean(), unit))
    print("  median          : %.2f %s" % (values.median(), unit))
    print("  range           : %.2f %s (min %.2f, max %.2f)"
          % (values.max() - values.min(), unit, values.min(), values.max()))
    print("  Q1 / Q3         : %.2f / %.2f" % (q1, q3))
    print("  IQR             : %.2f" % iqr)
    # .var() and .std() divide by n-1, the sample formula we want
    print("  variance (n-1)  : %.2f" % values.var())
    print("  std dev  (n-1)  : %.2f" % values.std())
    print("  outlier bounds  : below %.2f or above %.2f" % (lower_bound, upper_bound))
    print("  outliers        : %d %s"
          % (len(outliers), sorted(round(v, 2) for v in outliers)))

print("\nDESCRIPTIVE STATISTICS (full data, per group)")
show_stats("Group stage", group_dist)
show_stats("Elimination stage", elim_dist)

# We keep the outliers. Both group-stage ones are real team-matches, and
# dropping low work rates would push the group mean up, which is the direction
# the test is predicting.

sig_lvl = 0.05


# NORMALITY CHECK

# How far the mean may sit from the median before we call a sample skewed.
# These samples spread about 4 km/90 either side of their mean, so half a
# kilometre is roughly an eighth of one standard deviation: a gap that small
# does not show as a lean on the histogram, anything bigger does.
SYMMETRY_GAP = 0.5


def check_normality(label, values):
    """Check the shape of a sample the way this unit teaches, then cross-check.

    Primary evidence is the mean against the median, the Week 2 diagnostic.
    Shapiro-Wilk is printed underneath as a formal test that goes beyond the
    syllabus, so we can see whether the two ways of looking agree.
    """

    mean = values.mean()
    median = values.median()
    gap = mean - median

    # the verdict is worked out from the numbers, not typed in
    if abs(gap) <= SYMMETRY_GAP:
        verdict = "mean and median sit together, so the sample looks roughly symmetric"
        looks_normal = True
    elif gap > 0:
        verdict = "the mean sits above the median, so the sample is right-skewed"
        looks_normal = False
    else:
        verdict = "the mean sits below the median, so the sample is left-skewed"
        looks_normal = False

    shap_stat, shap_p = st.shapiro(values)
    passes_shapiro = shap_p >= sig_lvl

    print("\nNORMALITY CHECK - %s" % label)
    print("  PRIMARY - mean vs median, the check taught in this unit")
    print("    mean            : %.2f km/90" % mean)
    print("    median          : %.2f km/90" % median)
    print("    gap             : %+.2f km/90 (mean - median), skewed if larger than %.2f"
          % (gap, SYMMETRY_GAP))
    print("    shape           : %s" % verdict)
    if looks_normal:
        print("    condition       : MET, close enough to normal to use the t-test")
    else:
        # the test still runs, the p-value is just no longer exact
        print("    condition       : NOT MET, so the p-value from this sample is")
        print("                      reported as approximate")
    print("  SECONDARY - Shapiro-Wilk, a formal test beyond the unit syllabus")
    print("    p-value         : %.4f" % shap_p)
    print("    how to read it  : here a p ABOVE %.2f is the good outcome, the"
          % sig_lvl)
    print("                      opposite way round to a t-test p-value")
    print("    agreement       : %s"
          % ("agrees with the primary check" if passes_shapiro == looks_normal
             else "DISAGREES with the primary check - check the histogram"))

    return looks_normal


# computed from the data, not typed in, so it stays correct if the file changes
mu_0 = round(stats.mean(group_dist), 2)

# We sample MATCHES, not rows. Every match has two rows, one per team, and
# those two share the same pitch, the same referee and the same extra time,
# so they are not two independent observations. Drawing whole matches keeps
# the sampling unit honest. 13 matches gives n = 26 rows; 25 is not reachable
# because rows come in pairs.
n_matches = 13
match_ids = elim_stage["match_id"].drop_duplicates().sample(
    n=n_matches, random_state=SEED)
sample = elim_stage[elim_stage["match_id"].isin(match_ids)]["distance_per_90"]
n = len(sample)

x_bar = stats.mean(sample)
s = stats.stdev(sample)

# t-statistic: t* = (x_bar - mu_0) / (s / sqrt(n))
std_err = s / math.sqrt(n)
t_hand = (x_bar - mu_0) / std_err
deg_free = n - 1

# p-value
t_stats, p_val = st.ttest_1samp(sample, mu_0, alternative='less')

print("ONE-SAMPLE t-TEST")
print("H0: mu = %.2f km/90, H1: mu < %.2f km/90" % (mu_0, mu_0))

check_normality("elimination sample (%d rows from %d matches)" % (n, n_matches),
                sample)

print("\nSample size: %d rows from %d distinct matches" % (n, n_matches))
print("Sample mean: %.2f km/90" % x_bar)
print("Sample standard deviation: %.2f" % s)
print("Standard error: %.2f" % std_err)
print("t* by hand: %.2f" % t_hand)
print("t* from scipy: %.2f" % t_stats)
print("Degrees of freedom: %d" % deg_free)
print("p-value: %.4f" % p_val)

print("\nConclusion:")
if p_val < sig_lvl:
    print("\t We reject the null hypothesis.")
    print("\t Teams work at a lower rate than %.2f km/90 in elimination matches." % mu_0)
else:
    print("\t We dont have enough evidence to reject the null hypothesis.")

# TWO-SAMPLE t-TEST 

# solve
# Same idea as above: draw matches, then take both team rows from each one.
# 25 group matches give n1 = 50 rows and 15 elimination matches give n2 = 30,
# the same sample sizes as before but without splitting any match.
m1 = 25
m2 = 15
g_ids = group_stage["match_id"].drop_duplicates().sample(n=m1, random_state=SEED)
e_ids = elim_stage["match_id"].drop_duplicates().sample(n=m2, random_state=SEED)

t1 = group_stage[group_stage["match_id"].isin(g_ids)]["distance_per_90"]
t2 = elim_stage[elim_stage["match_id"].isin(e_ids)]["distance_per_90"]

n1 = len(t1)
n2 = len(t2)

# basic statistics
x_bar1 = stats.mean(t1)
s1 = stats.stdev(t1)
x_bar2 = stats.mean(t2)
s2 = stats.stdev(t2)

# t-statistic: t* = (x_bar1 - x_bar2) / sqrt(s1^2/n1 + s2^2/n2)
denom = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
t_hand2 = (x_bar1 - x_bar2) / denom
# Two degrees of freedom are in play and they are not the same number.
# The conservative rule taught in the unit is the smaller sample minus one.
deg_free_simple = min(n1, n2) - 1

# scipy's Welch test does not use that. It uses the Welch-Satterthwaite
# formula below, and that is the df its p-value belongs to, so it is the one
# reported with the p-value.
v1 = s1 ** 2 / n1
v2 = s2 ** 2 / n2
deg_free2 = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))

# equal_var=False is Welch's t-test, which does not assume the two groups have
# the same variance. The two sds are close here, so pooling would give a
# similar answer, but Welch is safe either way.
t_stats2, p_val2 = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2,
                                           equal_var=False, alternative='two-sided')

print("\n\nTWO-SAMPLE t-TEST")
print("H0: mu1 = mu2, H1: mu1 != mu2")

# normality, checked on each sample separately
check_normality("group stage sample (%d rows from %d matches)" % (n1, m1), t1)
check_normality("elimination stage sample (%d rows from %d matches)" % (n2, m2), t2)

print("\nGroup stage - size: %d rows from %d distinct matches, mean: %.2f km/90, standard deviation: %.2f" % (n1, m1, x_bar1, s1))
print("Elimination stage - size: %d rows from %d distinct matches, mean: %.2f km/90, standard deviation: %.2f" % (n2, m2, x_bar2, s2))
# group stage minus elimination stage, the same order as t* below
print("Difference in means (group - elimination): %.2f km/90" % (x_bar1 - x_bar2))
print("\nt* by hand: %.2f" % t_hand2)
print("t* from scipy: %.2f" % t_stats2)
print("Degrees of freedom (Welch-Satterthwaite, used for the p-value): %.1f" % deg_free2)
print("Degrees of freedom (conservative rule, min(n1,n2)-1): %d" % deg_free_simple)
print("p-value: %.4f" % p_val2)

# Conclude
print("\nConclusion:")
if p_val2 < sig_lvl:
    print("\t We reject the null hypothesis.")
    print("\t The two stages do differ in work rate.")
else:
    print("\t We dont have enough evidence to reject the null hypothesis.")

