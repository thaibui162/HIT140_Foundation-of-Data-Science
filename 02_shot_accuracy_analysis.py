
"""Shot accuracy gaps in group and knockout matches"""

import scipy.stats as st
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("world_cup_2026_team_match_sampled.csv")
colours = ["#52C4EA", "#36383B"]

# Select winning teams
winners = df[df["result"] == "Win"][["match_id", "phase", "shot_accuracy_pct"]]
winners = winners.rename(columns={"shot_accuracy_pct": "winner_accuracy"})

# Select losing teams
losers = df[df["result"] == "Loss"][["match_id", "phase", "shot_accuracy_pct"]]
losers = losers.rename(columns={"shot_accuracy_pct": "loser_accuracy"})

# Pair winner and loser from the same match
pairs = winners.merge(losers, on=["match_id", "phase"])

# Calculate winner–loser accuracy gap
pairs["accuracy_difference"] = pairs["winner_accuracy"] - pairs["loser_accuracy"]

# Descriptive statistics
summary = (pairs.groupby("phase")["accuracy_difference"]
           .agg(["count", "mean", "median", "std"])
           .reindex(["Group stage", "Knockout stage"]))
print(summary.round(2))

# 95% confidence intervals
for phase_name in ["Group stage", "Knockout stage"]:
    group = pairs[pairs["phase"] == phase_name]["accuracy_difference"]

    mean = group.mean()
    standard_error = st.sem(group)
    margin_error = st.t.ppf(0.975, len(group) - 1) * standard_error

    lower = mean - margin_error
    upper = mean + margin_error

    print(phase_name,
        "Mean:", round(mean, 2),
        "with 95% CI:", (round(lower, 2), round(upper, 2)))

# Bar chart
average_gap = summary["mean"]

average_gap.plot(kind="bar", color=colours)
plt.title("Mean Winner-Loser Shot Accuracy Gap")
plt.xlabel("Tournament Phase")
plt.ylabel("Accuracy Gap (percentage points)")
plt.xticks(rotation=0)
plt.grid(linestyle="--",alpha=0.5)
plt.show()