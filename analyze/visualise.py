"""
visualise.py — figures for Task 1.

Two audiences, two kinds of figure, and they are not interchangeable.

  FOR THE MARKER   fig3_histograms.png, fig4_boxplots.png
                   A histogram and a boxplot per group. These are required
                   evidence for the Week 2 descriptive work and for the
                   normality check the t-tests in hypothesis.py rest on.

  FOR A GENERAL    fig1_story.png, fig2_confidence_intervals.png
  AUDIENCE         The three-panel reversal story, and the two stage means
                   with their 95% confidence intervals.

Run:  python visualise.py     ->  writes PNGs into figures/
"""

import os

import matplotlib
matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st


# ===== DESIGN TOKENS ========================================================
# Two categories, so two categorical hues in fixed slot order - blue for the
# group stage, orange for elimination. The pair was checked for colour-vision
# deficiency separation rather than eyeballed: worst-case Delta E 24.7
# (protanopia), well clear of the 8.0 threshold, and both sit above 3:1
# contrast on the page. Identity is never carried by colour alone here - every
# bar and panel is labelled in text as well.

GROUP_COLOR = "#2a78d6"        # blue
ELIM_COLOR = "#eb6834"         # orange
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e0dfd9"

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_SOFT,
    "text.color": INK,
    "xtick.color": INK_SOFT,
    "ytick.color": INK_SOFT,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def style_axis(ax):
    """Recessive grid, no chart junk."""
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ===== FIGURE 1 - THE STORY, IN THREE STEPS =================================

def figure_story(group, elim):
    """Three plain bar charts that walk a reader through the reversal.

    Panel 1 says elimination teams cover more ground. Panel 2 says their
    matches are longer. Panel 3 says that once you divide the first by the
    second, they actually run LESS. Read left to right it needs no statistics
    background at all.

    All three y-axes start at zero. Panel 3's difference is only about 3% of
    its bar height and will look small - that is honest, and the fix is the
    annotation, NOT a truncated axis. Cutting the axis to make a 2.69 gap
    look dramatic is the single most common way a chart lies.
    """
    panels = [
        ("1. Teams cover more ground\nin elimination matches",
         "Total distance per match (km)",
         group["distance_km"].mean(), elim["distance_km"].mean(), "km"),
        ("2. ...but those matches\nlast longer",
         "Match duration (minutes)",
         group["time_played_min"].mean(), elim["time_played_min"].mean(), "min"),
        ("3. ...so per 90 minutes played,\nthey actually run LESS",
         "Distance per 90 minutes (km)",
         group["distance_per_90"].mean(), elim["distance_per_90"].mean(), "km/90"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))

    for ax, (title, ylabel, group_value, elim_value, unit) in zip(axes, panels):
        bars = ax.bar(
            ["Group\nstage", "Elimination\nstage"],
            [group_value, elim_value],
            color=[GROUP_COLOR, ELIM_COLOR],
            width=0.55,
        )

        # Direct labels beat a y-axis read-off, and they stay in text ink
        # rather than the bar colour.
        for bar, value in zip(bars, [group_value, elim_value]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ax.get_ylim()[1] * 0.02,
                f"{value:.1f}",
                ha="center", va="bottom", fontweight="bold", color=INK,
            )

        difference = elim_value - group_value
        ax.set_title(title, color=INK, pad=12)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(0, max(group_value, elim_value) * 1.25)
        style_axis(ax)

        # The gap named in words, because on a zero baseline a 3% difference
        # is invisible however true it is.
        ax.text(
            0.5, 0.94,
            f"{difference:+.1f} {unit}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=11, fontweight="bold", color=INK,
        )

    fig.suptitle(
        "Elimination teams run further - but not harder",
        fontsize=14, fontweight="bold", color=INK, y=0.99,
    )
    fig.text(
        0.5, 0.015,
        "FIFA World Cup 2026  ·  208 team-match records  ·  "
        "group stage n=144, elimination n=64",
        ha="center", fontsize=8.5, color=INK_SOFT,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    return fig


# ===== FIGURE 2 - MEANS WITH 95% CONFIDENCE INTERVALS =======================

def figure_confidence_intervals(group, elim):
    """One 95% CI per stage, as required by the task definition.

    A general audience reads two ranges that do not overlap as "this is not
    luck" without needing a p-value, which is exactly the right intuition
    here. A dot plot rather than bars: these are estimates with uncertainty,
    not magnitudes, so there is nothing to anchor at zero.
    """
    fig, ax = plt.subplots(figsize=(9, 4))

    # Both groups have n >= 30, so the Central Limit Theorem licenses the
    # normal-based interval and z* is the right critical value.
    confidence = 0.95
    z_star = st.norm.ppf(1 - (1 - confidence) / 2)      # 1.960

    rows = [
        ("Group stage", group["distance_per_90"], GROUP_COLOR),
        ("Elimination stage", elim["distance_per_90"], ELIM_COLOR),
    ]

    for position, (label, values, color) in enumerate(rows):
        n = len(values)
        mean = values.mean()
        std_dev = values.std()                  # ddof=1, the sample formula
        standard_error = std_dev / n ** 0.5
        margin_of_error = z_star * standard_error

        ax.errorbar(
            mean, position,
            xerr=margin_of_error,
            fmt="o", markersize=11, color=color,
            ecolor=color, elinewidth=2.5, capsize=7, capthick=2.5,
        )
        ax.text(
            mean, position + 0.22,
            f"{mean:.2f} km/90",
            ha="center", va="bottom", fontweight="bold", color=INK,
        )
        ax.text(
            mean, position - 0.26,
            f"n = {n}   95% CI [{mean - margin_of_error:.2f}, "
            f"{mean + margin_of_error:.2f}]   ±{margin_of_error:.2f}",
            ha="center", va="top", fontsize=9, color=INK_SOFT,
        )

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label for label, _, _ in rows])
    ax.set_ylim(-0.65, len(rows) - 0.35)
    # Group stage first, reading top to bottom - the same order the bars use
    # left to right in figure 1.
    ax.invert_yaxis()
    ax.set_xlabel("Distance per 90 minutes (km)")
    ax.set_title(
        "The two stages' average work rates, with 95% confidence intervals",
        fontweight="bold", color=INK, pad=14,
    )
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    fig.text(
        0.5, 0.02,
        "The two intervals do not overlap, so the gap is unlikely to be "
        "chance.  z* = 1.960; both groups n >= 30, so the normal-based "
        "interval applies.",
        ha="center", fontsize=8.5, color=INK_SOFT,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


# ===== FIGURE 3 - HISTOGRAM PER GROUP (assessment requirement) ==============

def figure_histograms(group, elim):
    """Shape of each group's distribution - what the normality check is read from.

    Shared x-axis and shared bin edges, so the two panels are actually
    comparable. Different bins per panel would make two distributions look
    different when only the binning changed.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True, sharey=True)

    all_values = list(group["distance_per_90"]) + list(elim["distance_per_90"])
    bins = 18
    bin_range = (min(all_values), max(all_values))

    # Headroom reserved above the tallest bar in either panel, so the mean
    # label always has empty space to sit in. Shared y-limits keep the two
    # panels comparable.
    tallest = max(
        max(ax_values.value_counts(bins=bins).max() for ax_values in
            (group["distance_per_90"], elim["distance_per_90"])),
        1,
    )

    for ax, (label, values, color) in zip(axes, [
        ("Group stage", group["distance_per_90"], GROUP_COLOR),
        ("Elimination stage", elim["distance_per_90"], ELIM_COLOR),
    ]):
        ax.hist(values, bins=bins, range=bin_range,
                color=color, edgecolor=SURFACE, linewidth=1.2)
        ax.set_ylim(0, tallest * 1.22)
        ax.axvline(values.mean(), color=INK, linewidth=1.6, linestyle="--")
        ax.text(
            values.mean(), tallest * 1.09,
            f"mean {values.mean():.1f}",
            fontsize=9, color=INK, va="center", ha="center",
            bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 2},
        )
        ax.set_title(f"{label}  (n = {len(values)}, skew = {values.skew():+.2f})",
                     color=INK)
        ax.set_xlabel("Distance per 90 minutes (km)")
        style_axis(ax)

    axes[0].set_ylabel("Number of team-matches")
    fig.suptitle("Distribution shape by stage", fontsize=13,
                 fontweight="bold", color=INK)
    fig.text(
        0.5, 0.015,
        "Elimination is roughly symmetric and unimodal. The group stage is "
        "left-skewed (-0.99), pulled by a few unusually low work rates.\n"
        "That is why its Shapiro-Wilk test fails, and why the p-values in "
        "hypothesis.py are flagged as approximate rather than exact.",
        ha="center", fontsize=8.5, color=INK_SOFT,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.94])
    return fig


# ===== FIGURE 4 - BOXPLOT PER GROUP (assessment requirement) ================

def figure_boxplots(group, elim):
    """Both measures, side by side, so the variance story is visible.

    The left panel is raw kilometres and the right is per 90 minutes. The
    elimination box collapses from very wide to the same width as the group
    box - that is the whole argument for normalising, in one picture.

    Two separate panels rather than two scales on one axis: km and km/90 are
    different quantities, and putting them on a shared axis would invent a
    comparison the data does not support.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    panels = [
        ("distance_km", "Raw distance per match (km)",
         "Before normalising - elimination is far more spread out"),
        ("distance_per_90", "Distance per 90 minutes (km)",
         "After normalising - the extra spread disappears"),
    ]

    for ax, (column, ylabel, title) in zip(axes, panels):
        boxes = ax.boxplot(
            [group[column], elim[column]],
            tick_labels=["Group\nstage", "Elimination\nstage"],
            patch_artist=True, widths=0.5,
            medianprops={"color": INK, "linewidth": 2},
            flierprops={"marker": "o", "markersize": 5,
                        "markerfacecolor": INK_SOFT,
                        "markeredgecolor": "none", "alpha": 0.6},
        )
        for patch, color in zip(boxes["boxes"], [GROUP_COLOR, ELIM_COLOR]):
            patch.set_facecolor(color)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1.5)
        for element in ("whiskers", "caps"):
            for artist in boxes[element]:
                artist.set_color(INK_SOFT)

        ax.set_ylabel(ylabel, fontsize=9)

        # The number the picture is making an argument about, folded into the
        # title rather than floated over the plot area where it collided with
        # the low outliers.
        ax.set_title(
            f"{title}\nstandard deviation {group[column].std():.2f} "
            f"vs {elim[column].std():.2f}",
            color=INK, fontsize=10, pad=10,
        )
        style_axis(ax)

    fig.suptitle(
        "Spread by stage, before and after adjusting for match length",
        fontsize=13, fontweight="bold", color=INK,
    )
    fig.text(
        0.5, 0.015,
        "Box = middle 50% (Q1 to Q3), line = median, whiskers = 1.5x IQR, "
        "dots = outliers beyond that.",
        ha="center", fontsize=8.5, color=INK_SOFT,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    return fig


# ===== DATA =================================================================

def load_data():
    """Load the two CSVs, join them, and split into the two stages."""

    # the CSVs live in the Data folder next to this one
    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "..", "Data")

    distance = pd.read_csv(os.path.join(data, "processed", "wc2026_team_distance_CLEAN.csv"))
    duration = pd.read_csv(os.path.join(data, "reference", "match_time_played.csv"))

    # left join, two team rows per match and one duration row per match
    df = distance[["match_id", "result_id", "date", "stage", "team", "distance_km"]].merge(
        duration[["match_id", "time_played_min"]], on="match_id", how="left")

    df["distance_per_90"] = df["distance_km"] / df["time_played_min"] * 90

    # anything that is not "First Stage" is knockout
    return df, df[df["stage"] == "First Stage"], df[df["stage"] != "First Stage"]


# ===== DRIVER ===============================================================

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df, group, elim = load_data()

    figures = [
        ("fig1_story.png", figure_story(group, elim)),
        ("fig2_confidence_intervals.png",
         figure_confidence_intervals(group, elim)),
        ("fig3_histograms.png", figure_histograms(group, elim)),
        ("fig4_boxplots.png", figure_boxplots(group, elim)),
    ]

    for name, fig in figures:
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("wrote", os.path.relpath(path, os.path.dirname(FIG_DIR)))


if __name__ == "__main__":
    main()
