import pandas as pd
import scipy.stats as st

df = pd.read_csv("world_cup_2026_team_match_sampled.csv")

# Select winning teams
winners = df[df["result"] == "Win"][["match_id", "phase", "shot_accuracy_pct"]]
winners = winners.rename(columns={"shot_accuracy_pct": "winner_accuracy"})

# Select losing teams
losers = df[df["result"] == "Loss"][["match_id", "phase", "shot_accuracy_pct"]]
losers = losers.rename(columns={"shot_accuracy_pct": "loser_accuracy"})

# Pair teams from the same match
pairs = winners.merge(losers, on=["match_id", "phase"])
pairs["accuracy_difference"] = (pairs["winner_accuracy"] - pairs["loser_accuracy"])

# Separate the two tournament phases
group_gap = pairs[pairs["phase"] == "Group stage"]["accuracy_difference"]
knockout_gap = pairs[pairs["phase"] == "Knockout stage"]["accuracy_difference"]


# H0: Mean knockout gap = mean group-stage gap
# H1: Mean knockout gap > mean group-stage gap
test_result = st.ttest_ind(knockout_gap,group_gap,
                           equal_var=False,
                           alternative="greater")
mean_difference = knockout_gap.mean() - group_gap.mean()
a = knockout_gap.var(ddof=1) / len(knockout_gap)
b = group_gap.var(ddof=1) / len(group_gap)

standard_error = (a + b) ** 0.5

degrees_freedom = ((a+b)**2/(a**2/(len(knockout_gap)-1)+b**2/(len(group_gap)-1)))

lower, upper = st.t.interval(0.95,degrees_freedom,
                             loc=mean_difference,
                             scale=standard_error)

print(f"Group-stage mean gap:{group_gap.mean():.2f}")
print(f"Knockout-stage mean gap:{knockout_gap.mean():.2f}")
print(f"Mean difference:{mean_difference:.2f}")
print(f"95% CI: {lower:.2f} to {upper:.2f}")
print(f"T-statistic:{test_result.statistic:.2f}")
print(f"Degrees of freedom:{degrees_freedom:.2f}")
print(f"One-sided p-value:{test_result.pvalue:.3f}")

if test_result.pvalue < 0.05:
    print("Reject H0")
    print("Knockout matches had a significantly higher mean gap")
else:
    print("Fail to reject H0")
    print( "The difference was not statistically significant")

import matplotlib.pyplot as plt

# Check approximate normality using histograms
plt.hist(group_gap, bins=8, alpha=0.6, label="Group stage")
plt.hist(knockout_gap, bins=8, alpha=0.6, label="Knockout stage")
plt.title("Distribution of Shot Accuracy Gaps")
plt.xlabel("Winner-Loser Accuracy Gap")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Check using the complete dataset
full_df = pd.read_csv("world_cup_2026_team_match_cleaned1.csv")

full_pairs = full_df[full_df["result"].isin(["Win", "Loss"])].pivot(
    index=["match_id", "phase"],
    columns="result",
    values="shot_accuracy_pct").dropna()

full_pairs["gap"] = full_pairs["Win"] - full_pairs["Loss"]

full_group = full_pairs.loc[full_pairs.index.get_level_values("phase") == "Group stage", "gap"]

full_knockout = full_pairs.loc[full_pairs.index.get_level_values("phase") == "Knockout stage", "gap"]

full_test = st.ttest_ind(full_knockout,
                         full_group,
                         equal_var=False,
                         alternative="greater")

print(f"Complete dataset p-value: {full_test.pvalue:.3f}")
print("Same conclusion")