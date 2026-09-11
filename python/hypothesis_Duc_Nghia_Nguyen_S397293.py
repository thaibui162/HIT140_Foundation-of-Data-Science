
# On average how many fouls per match got. 

# Question: Does the eliminated stage have a significant impact on the number of fouls committed in a match compared to the group stage?

import pandas as pd
import statistics as stats
import scipy.stats as st

df = pd.read_csv("world_cup_2026_fouls_match_level.csv")
group_stage = df[df["Round"] == "First Stage"]
knockout_stage = df[df["Round"] != "First Stage"]

n1 = 50
n2 = 30
t1 = group_stage.sample(n=n1, random_state=1)
t2 = knockout_stage.sample(n=n2, random_state=1)

    # the basic statistics:
x_bar1 = stats.mean(t1["FoulsCommitted"])
s1 = stats.stdev(t1["FoulsCommitted"])
x_bar2 = stats.mean(t2["FoulsCommitted"]) 
s2 = stats.stdev(t2["FoulsCommitted"])

    # perform two-sample t-test
    # null hypothesis: mean of group stage = mean of knockout stage - fouls committed in a match of 2 groups are equal
    # alternative hypothesis: mean of group stage < mean of knockout stage - fouls committed in a match of knockout stage is higher than that of group stage
    # H0: u1 = u2
    # H1: u1 < u2
t_stats, p_val = st.ttest_ind_from_stats(x_bar1, s1, n1, x_bar2, s2, n2, equal_var=False, alternative='less')

mean_difference =(x_bar1 - x_bar2)

print("s1 = %.2f, s2 = %.2f" % (s1, s2))

print(
    "The sampled group-stage matches had a mean of "
    "%.2f fouls per match." % x_bar1
)

print(
    "The sampled knockout-stage matches had a mean of "
    "%.2f fouls per match." % x_bar2
)

print(
    "The observed difference was %.2f fouls per match."
    % mean_difference
)


print("\n Computing t* ...")
print("\t t-statistic (t*): %.2f" % t_stats)

print("\n Computing p-value ...")
print("\t p-value: %.4f" % p_val)


print("\n Conclusion:")
if p_val < 0.05:
    print("%.4f < 0.05" % p_val)
    print("\t We reject the null hypothesis.")
else:
    print("%.4f > 0.05" % p_val)
    print("\t We dont have enough evidence to reject the null hypothesis.")


