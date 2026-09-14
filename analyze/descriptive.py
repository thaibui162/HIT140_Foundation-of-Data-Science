import os
import pandas as pd

#load data

# the CSVs live in the Data folder next to this one
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "Data")

distance = pd.read_csv(os.path.join(DATA, "processed", "wc2026_team_distance_CLEAN.csv"))
duration = pd.read_csv(os.path.join(DATA, "reference", "match_time_played.csv"))

# Left join so a team-match with no duration would show up as NaN instead of
# being dropped without us noticing. Two team rows per match, one duration row.
df = distance[["match_id", "result_id", "date", "stage", "team", "distance_km"]].merge(
    duration[["match_id", "time_played_min"]], on="match_id", how="left")

# distance per 90 minutes, so a longer match does not look like harder work
df["distance_per_90"] = df["distance_km"] / df["time_played_min"] * 90

# split the two stages. Anything that is not "First Stage" is knockout,
# including the third-place play-off.
group = df[df["stage"] == "First Stage"]
elim = df[df["stage"] != "First Stage"]


def describe(label, unit, all_values, group_values, elim_values):
    """Print the same block of statistics for one measure."""

    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)

    # mean
    print("\nMean in all matches:", round(all_values.mean(), 2), unit)
    print("Mean in group stage:", round(group_values.mean(), 2), unit)
    print("Mean in elimination stage:", round(elim_values.mean(), 2), unit)

    # median
    print("\nMedian in all matches:", round(all_values.median(), 2), unit)
    print("Median in group stage:", round(group_values.median(), 2), unit)
    print("Median in elimination stage:", round(elim_values.median(), 2), unit)

    # range = biggest value - smallest value
    group_min = group_values.min()
    group_max = group_values.max()
    elim_min = elim_values.min()
    elim_max = elim_values.max()
    print("\nGroup stage range:", round(group_max - group_min, 2), unit)
    print("Group stage min:", group_min, unit + ", max:", group_max, unit)
    print("Elimination stage range:", round(elim_max - elim_min, 2), unit)
    print("Elimination stage min:", elim_min, unit + ", max:", elim_max, unit)

    # variance and standard deviation
  
    print("\nGroup stage variance:", round(group_values.var(), 2))
    print("Group stage standard deviation:", round(group_values.std(), 2))
    print("Elimination stage variance:", round(elim_values.var(), 2))
    print("Elimination stage standard deviation:", round(elim_values.std(), 2))

    # quartiles and IQR (IQR = 75th percentile - 25th percentile)
    g25 = group_values.quantile(0.25)
    g75 = group_values.quantile(0.75)
    e25 = elim_values.quantile(0.25)
    e75 = elim_values.quantile(0.75)
    print("\nGroup stage 25th:", round(g25, 2),
          "75th:", round(g75, 2), "IQR:", round(g75 - g25, 2))
    print("Elimination stage 25th:", round(e25, 2),
          "75th:", round(e75, 2), "IQR:", round(e75 - e25, 2))

    # outlier bounds, Week 2 rule: Q1 - 1.5*IQR and Q3 + 1.5*IQR
    g_low = g25 - 1.5 * (g75 - g25)
    g_high = g75 + 1.5 * (g75 - g25)
    g_out = group_values[(group_values < g_low) | (group_values > g_high)]
    print("\nGroup stage outlier bounds: below %.2f or above %.2f" % (g_low, g_high))
    print("Group stage outliers:", len(g_out), sorted(round(v, 2) for v in g_out))

    e_low = e25 - 1.5 * (e75 - e25)
    e_high = e75 + 1.5 * (e75 - e25)
    e_out = elim_values[(elim_values < e_low) | (elim_values > e_high)]
    print("\nElimination stage outlier bounds: below %.2f or above %.2f" % (e_low, e_high))
    print("Elimination stage outliers:", len(e_out), sorted(round(v, 2) for v in e_out))


    # comparison
    print("\nDifference in means:",
          round(elim_values.mean() - group_values.mean(), 2), unit)
    print("Difference in medians:",
          round(elim_values.median() - group_values.median(), 2), unit)


# Match duration ,  it is what makes the raw comparison misleading.
describe("MATCH DURATION (the confounder)", "min",
         df["time_played_min"],
         group["time_played_min"],
         elim["time_played_min"])

describe("RAW DISTANCE COVERED - not comparable across stages", "km",
         df["distance_km"],
         group["distance_km"],
         elim["distance_km"])

describe("DISTANCE PER 90 MINUTES - the comparable measure", "km/90",
         df["distance_per_90"],
         group["distance_per_90"],
         elim["distance_per_90"])
