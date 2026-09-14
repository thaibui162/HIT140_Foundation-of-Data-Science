import os
import statistics as stats
import numpy as np
import math
import pandas as pd
import scipy.stats as st
import statsmodels.stats.weightstats as stm

# one seed for every sample drawn in this project, so the code, the output
# and the slides all agree
SEED = 2026

# load data

# the CSVs live in the Data folder next to this one
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")

distance = pd.read_csv(os.path.join(DATA, "processed", "wc2026_team_distance_CLEAN.csv"))
duration = pd.read_csv(os.path.join(DATA, "reference", "match_time_played.csv"))

df = distance[["match_id", "result_id", "date", "stage", "team", "distance_km"]].merge(
    duration[["match_id", "time_played_min"]], on="match_id", how="left")

# distance per 90 minutes, so a longer match does not look like harder work
df["distance_per_90"] = df["distance_km"] / df["time_played_min"] * 90

# split the two stages. Anything that is not "First Stage" is knockout,
# including the third-place play-off.
group_stage = df[df["stage"] == "First Stage"]
elim_stage = df[df["stage"] != "First Stage"]

dist = df["distance_per_90"]


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


# The intervals below come from these groups, so the spread is shown first.
print("\nDESCRIPTIVE STATISTICS (per group)")
show_stats("Group stage", group_stage["distance_per_90"])
show_stats("Elimination stage", elim_stage["distance_per_90"])

# We keep the outliers, they are all real team-matches.

pop_mean = stats.mean(dist)
pop_median = stats.median(dist)
pop_mode = stats.mode(dist)
pop_max = np.max(dist)

# pvariance and pstdev divide by N, which is right when the rows in front of us
# are the whole population.
pop_var = stats.pvariance(dist)
pop_std = stats.pstdev(dist)

print("Population size:", len(dist), "team-match records")
print("Population mean: %.2f km/90" % pop_mean)
print("Population median: %.2f km/90" % pop_median)
print("Population max: %.2f km/90" % pop_max)
print("Population variance: %.2f" % pop_var)
print("Population standard deviation: %.2f km/90" % pop_std)

# Distance is continuous and nearly every value is different, so the mode only
# picks up whichever number happened to repeat after rounding. It is printed
# because the brief asks for it.
print("Population mode: %.2f km/90  (not meaningful - continuous measurement)" % pop_mode)


#2 sample confidence intervals
# does the size of a sample change how wide its confidence interval is?
# n >= 30 counts as large, so the z distribution is used for it.
# n < 30 is small, so the t distribution is used instead.

# We sample MATCHES, not rows. Each match has two rows, one per team, and
# those two share the same match length and the same extra time, so they are
# not independent observations. Drawing whole matches keeps the sampling unit
# honest. 20 matches give 40 rows; 8 matches give 16, since rows come in pairs
# and 15 is not reachable.
m_z = 20
m_t = 8

ids_z = df["match_id"].drop_duplicates().sample(n=m_z, random_state=SEED)
ids_t = df["match_id"].drop_duplicates().sample(n=m_t, random_state=SEED)

sample_z = df[df["match_id"].isin(ids_z)]["distance_per_90"]
sample_t = df[df["match_id"].isin(ids_t)]["distance_per_90"]

n_z = len(sample_z)
n_t = len(sample_t)

# sample mean
x_bar_z = stats.mean(sample_z)
x_bar_t = stats.mean(sample_t)

# sample standard deviation - divide by n-1 here, these really are samples
s_z = stats.stdev(sample_z)
s_t = stats.stdev(sample_t)

# standard error = how far a sample mean of this size usually sits from the
# true mean. It gets smaller as n grows, which is the point of the comparison.
se_z = s_z / math.sqrt(n_z)
se_t = s_t / math.sqrt(n_t)

print("\nLarge sample  n=%d rows from %d distinct matches  mean: %.2f km/90  sd: %.2f  standard error: %.2f" % (n_z, m_z, x_bar_z, s_z, se_z))
print("Small sample  n=%d rows from %d distinct matches  mean: %.2f km/90  sd: %.2f  standard error: %.2f" % (n_t, m_t, x_bar_t, s_t, se_t))


# confidence intervals 
conf_lvl = 0.95
sig_lvl = 1 - conf_lvl        # alpha
deg_free = n_t - 1            # degrees of freedom for the t interval

# large sample: critical value -> margin of error -> bounds
z_score = st.norm.ppf(q=1 - sig_lvl / 2)
mrg_err_z = z_score * se_z
ci_low_z = x_bar_z - mrg_err_z
ci_upp_z = x_bar_z + mrg_err_z

# small sample: the same three steps, but t* replaces z* and is bigger, because
# with only 15 values the standard deviation is itself only an estimate
t_score = st.t.ppf(q=1 - sig_lvl / 2, df=deg_free)
mrg_err_t = t_score * se_t
ci_low_t = x_bar_t - mrg_err_t
ci_upp_t = x_bar_t + mrg_err_t

print("\nz* = %.3f, margin of error = %.2f km/90" % (z_score, mrg_err_z))
print("t* = %.3f (df=%d), margin of error = %.2f km/90" % (t_score, deg_free, mrg_err_t))

# the same two intervals from statsmodels, as a check on the arithmetic above
chk_low_z, chk_upp_z = stm._zconfint_generic(x_bar_z, se_z, alpha=sig_lvl, alternative="two-sided")
chk_low_t, chk_upp_t = stm._tconfint_generic(x_bar_t, se_t, deg_free, alpha=sig_lvl, alternative="two-sided")
print("statsmodels check - z: %.2f to %.2f, t: %.2f to %.2f" % (chk_low_z, chk_upp_z, chk_low_t, chk_upp_t))


# comparison of the two intervals, which is the point of this exercise. 
# The small sample's interval is wider, because the standard error is bigger and the critical value is bigger.
width_z = ci_upp_z - ci_low_z
width_t = ci_upp_t - ci_low_t

print("\n%d%% CI from the large sample (n=%d, z): %.2f to %.2f km/90" % (conf_lvl * 100, n_z, ci_low_z, ci_upp_z))
print("Interval width: %.2f km/90. Contains the population mean:" % width_z, ci_low_z <= pop_mean <= ci_upp_z)
print("\n%d%% CI from the small sample (n=%d, t): %.2f to %.2f km/90" % (conf_lvl * 100, n_t, ci_low_t, ci_upp_t))
print("Interval width: %.2f km/90. Contains the population mean:" % width_t, ci_low_t <= pop_mean <= ci_upp_t)
print("\nPopulation mean: %.2f km/90" % pop_mean)

