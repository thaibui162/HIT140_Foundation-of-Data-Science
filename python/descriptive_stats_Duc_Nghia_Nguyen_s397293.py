import pandas as pd
import statistics as stats
import scipy.stats as st
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("world_cup_2026_fouls_match_level.csv")
group_stage = df[df["Round"] == "First Stage"]
knockout_stage = df[df["Round"] != "First Stage"]

    # min max total
population_min = df["FoulsCommitted"].min()
group_stage_min = group_stage["FoulsCommitted"].min()
knockout_stage_min = knockout_stage["FoulsCommitted"].min()

population_max = df["FoulsCommitted"].max()
group_stage_max = group_stage["FoulsCommitted"].max()
knockout_stage_max = knockout_stage["FoulsCommitted"].max()

population_total = df["FoulsCommitted"].sum()
group_stage_total = group_stage["FoulsCommitted"].sum()
knockout_stage_total = knockout_stage["FoulsCommitted"].sum()

    # mode
population_mode = stats.mode(df["FoulsCommitted"])
group_stage_mode = stats.mode(group_stage["FoulsCommitted"])
knockout_stage_mode = stats.mode(knockout_stage["FoulsCommitted"])

    # median
population_median = stats.median(df["FoulsCommitted"])
group_stage_median = stats.median(group_stage["FoulsCommitted"])
knockout_stage_median = stats.median(knockout_stage["FoulsCommitted"])

    # mean
population_mean = stats.mean(df["FoulsCommitted"])
group_stage_mean = stats.mean(group_stage["FoulsCommitted"])
knockout_stage_mean = stats.mean(knockout_stage["FoulsCommitted"])

    # standard deviation
population_std = stats.stdev(df["FoulsCommitted"])
group_stage_std = stats.stdev(group_stage["FoulsCommitted"])
knockout_stage_std = stats.stdev(knockout_stage["FoulsCommitted"])
group_stage_n = len(group_stage["FoulsCommitted"])        #144 => 72 matches
knockout_stage_n = len(knockout_stage["FoulsCommitted"])  #64 => 32 matches


    # visualization - boxplot

boxplot_data = [df["FoulsCommitted"], group_stage["FoulsCommitted"], 
                knockout_stage["FoulsCommitted"]]
fig, ax = plt.subplots(figsize=(10, 6))

boxplot = ax.boxplot(boxplot_data, patch_artist=True, tick_labels=["All Matches", "Group Stage", "Knockout Stage"],
                     showmeans=True, meanline=True, showfliers=True,
                     medianprops=dict(color="red", linewidth=2),
                     meanprops=dict(color="blue", linewidth=2),
                     boxprops=dict(facecolor="lightblue", color="black"),
                     whiskerprops=dict(color="black"),
                    capprops=dict(color="black"))

colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(boxplot['boxes'], colors):
    patch.set_facecolor(color)

ax.set_title("Boxplot of Fouls Committed in World Cup 2026 Matches")
ax.set_ylabel("Number of Fouls Committed")
ax.set_xlabel("Match Stage")
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
fig.savefig("fouls_boxplot.png",dpi=300)




    # comparison
print("There are 72 group_stage matches & 32 knockout_matches")

print("\n--- Minimum ---")
print("All matches:", population_min)
print("Group stage:", group_stage_min)
print("Knockout stage:", knockout_stage_min)

print("\n--- Maximum ---")
print("All matches:", population_max)
print("Group stage:", group_stage_max)
print("Knockout stage:", knockout_stage_max)

print("\n--- Total ---")
print("All matches:", population_total)
print("Group stage:", group_stage_total)
print("Knockout stage:", knockout_stage_total)

print("\n--- Median ---")
print("\nMedian fouls committed in all matches: %.2f" % population_median)
print("Median fouls committed in group stage: %.2f" % group_stage_median)
print("Median fouls committed in knockout stage: %.2f" % knockout_stage_median)

print("\n--- Mean ---")
print("\nMean fouls committed in all matches: %.2f" % population_mean)
print("Mean fouls committed in group stage: %.2f" % group_stage_mean)
print("Mean fouls committed in knockout stage: %.2f" % knockout_stage_mean)

print("\n--- Standard deviation ---")
print("\nStandard deviation of fouls committed in all matches: %.2f" % population_std)
print("Standard deviation of fouls committed in group stage: %.2f" % group_stage_std)
print("Standard deviation of fouls committed in knockout stage: %.2f" % knockout_stage_std)

plt.show()
"""
    The results quite intersting, the mean fouls committed in knockout stage is higher than
      the mean fouls committed in group stage, which is what I expected. 
        The more competitive the match, the more fouls are committed?
    We could approach further to see in the group stage, the mean fouls committed in the matches of the teams that
      not advance (lose 1 and draw 1) 
    is higher than the mean fouls committed in the matches of the teams that advance (win 1 and draw 1) ? 

"""