import pandas as pd
import statistics as stats
import numpy as np
import math
import statsmodels.stats.weightstats as stm
import matplotlib.pyplot as plt

df_data = pd.read_csv("world_cup_2026_fouls_match_level.csv")

pop_mean = stats.mean(df_data["FoulsCommitted"])
pop_median = stats.median(df_data["FoulsCommitted"])
pop_mode = stats.mode(df_data["FoulsCommitted"])
pop_max = np.max(df_data["FoulsCommitted"])
pop_var = stats.variance(df_data["FoulsCommitted"])
pop_std = stats.stdev(df_data["FoulsCommitted"])


    # checking if the sample size affect to the CI compared with the population's
    # z and t sample sizes
n_z = 35 
n_t = 15

sample_z = df_data.sample(n=n_z, random_state=1)    # remove random_state=1 to get different sample each time
sample_t = df_data.sample(n=n_t, random_state=1)

sample_z.to_csv("sample_z.csv", index=False)
sample_t.to_csv("sample_t.csv", index=False)

    # mean
x_bar_z = stats.mean(sample_z["FoulsCommitted"])
x_bar_t = stats.mean(sample_t["FoulsCommitted"])

    # standard deviation    
s = stats.stdev(sample_t["FoulsCommitted"])


    # standard error
std_err_z = pop_std / math.sqrt(n_z)
std_err_t = pop_std / math.sqrt(n_t)

    # confidence level 95%, significance level, and degrees of freedom
conf_lvl = 0.95
sig_lvl = 1 - conf_lvl
df = n_t - 1


ci_low_z, ci_upp_z = stm._zconfint_generic(x_bar_z, std_err_z, alpha=0.05, alternative="two-sided")
ci_low_t, ci_upp_t = stm._zconfint_generic(x_bar_t, std_err_t, alpha=0.05, alternative="two-sided")

    # visualization

# Information for the two samples
means = [x_bar_z, x_bar_t]
lower_limits = [ci_low_z, ci_low_t]
upper_limits = [ci_upp_z, ci_upp_t]

labels = [
    "Sample n = 35",
    "Sample n = 15"
]

colours = ["blue", "orange"]
y_positions = [1, 0]

plt.figure(figsize=(10, 5))

for mean, lower, upper, label, colour, y in zip(
    means,
    lower_limits,
    upper_limits,
    labels,
    colours,
    y_positions
):
    # Confidence interval line
    plt.hlines(
        y=y,
        xmin=lower,
        xmax=upper,
        color=colour,
        linewidth=4
    )

    # End caps
    plt.plot(
        [lower, upper],
        [y, y],
        linestyle="None",
        marker="|",
        markersize=18,
        markeredgewidth=3,
        color=colour
    )

    # Sample mean
    plt.scatter(
        mean,
        y,
        color=colour,
        s=100,
        zorder=3
    )

    # Display values
    plt.text(
        mean,
        y + 0.13,
        f"Mean = {mean:.2f}",
        ha="center",
        color=colour
    )

    plt.text(lower, y - 0.15, f"{lower:.2f}", ha="center")
    plt.text(upper, y - 0.15, f"{upper:.2f}", ha="center")

# Known population mean
plt.axvline(
    x=pop_mean,
    color="red",
    linestyle="--",
    label=f"Population mean = {pop_mean:.2f}"
)

plt.yticks(y_positions, labels)
plt.xlabel("Total fouls committed per match")
plt.title("Comparison of 95% Confidence Intervals")
plt.legend()
plt.grid(axis="x", linestyle="--", alpha=0.4)
plt.ylim(-0.5, 1.5)
plt.tight_layout()
plt.savefig("CI_random_2_samples.png",dpi=300)



    #comparison
print("Mean of population: %.2f" % pop_mean)
print("Standard deviation of population: %.2f" % pop_std)

print("\nMean of z sample with size %d: %.2f" % (n_z, x_bar_z))
print("CI z*: %.2f to %.2f. Interval size: %.2f" % (ci_low_z, ci_upp_z, ci_upp_z - ci_low_z))
print(">> There is %.2f%% confident that the mean of population %.2f lies in this interval with sample size %d." % (conf_lvl * 100, pop_mean, n_z))

print("\nMean of t sample with size %d: %.2f" % (n_t, x_bar_t))
print("CI t*: %.2f to %.2f. Interval size: %.2f" % (ci_low_t, ci_upp_t, ci_upp_t - ci_low_t))
print(">> There is %.2f%% confident that the mean of population %.2f would include in this interval with sample size %d." % (conf_lvl * 100, pop_mean, n_t))

    # conclusion: the bigger sample the smaller the confidence interval, and the closer to the population mean? 
    # sample size bigger >> interval is more narrow.

plt.show()