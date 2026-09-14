"""
charts.py - presentation figures for the WC2026 distance analysis.

Three figures, and they are NOT all built from the same data:

  fig_descriptive_sample.png   the analysis sample (50 group / 30 elimination)
  fig_normality_sample.png     the same analysis sample
  fig_population_context.png   all 208 team-match records

The samples are re-derived here with the same seed and the same match_id draw
that hypothesis.py uses, so every number on a figure matches the number that
script prints. Nothing is redrawn and no statistic is typed in by hand.

Run:  python charts.py     ->  writes PNGs into figures/
"""

import os

import matplotlib
matplotlib.use("Agg")          # write files, never open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st

# the same seed the analysis scripts use
SEED = 2026

# The same mean-vs-median rule hypothesis.py applies. Half a kilometre is about
# an eighth of one standard deviation here, too small to show as a lean on a
# histogram; anything larger shows.
SYMMETRY_GAP = 0.5


def shape_verdict(values):
    """The Week 2 normality check: where the mean sits against the median.

    Returns the gap and the verdict in words. Worked out from the numbers, so
    it stays right if the sample ever changes.
    """

    gap = values.mean() - values.median()
    if abs(gap) <= SYMMETRY_GAP:
        return gap, "roughly symmetric"
    if gap > 0:
        return gap, "right-skewed"
    return gap, "left-skewed"


# ===== SLIDE PALETTE ========================================================
# The two stage colours mean the same two stages in every figure.

GROUP_COLOR = "#2A78D6"        # blue, group stage
ELIM_COLOR = "#EB6834"         # orange, elimination stage
NAVY = "#1E2761"               # neutral emphasis
GREY = "#5A5F70"               # axis labels and annotations
WHITE = "#FFFFFF"
GRIDLINE = "#EEEEEE"

# projection sizes - nothing below 10
SIZE_TITLE = 13
SIZE_LABEL = 11
SIZE_TICK = 10
SIZE_NOTE = 10
SIZE_ANNOT = 12

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

plt.rcParams.update({
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "axes.edgecolor": GREY,
    "axes.labelcolor": GREY,
    "text.color": NAVY,
    "xtick.color": GREY,
    "ytick.color": GREY,
    "font.size": SIZE_TICK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def style_axis(ax):
    """Light horizontal gridlines only, no chart junk."""
    ax.grid(axis="y", color=GRIDLINE, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=SIZE_TICK)


# ===== DATA =================================================================

REQUIRED = ["match_id", "stage", "team", "distance_km"]


def load_data():
    """Load the two CSVs and build the analysis frame."""

    here = os.path.dirname(os.path.abspath(__file__))
    data = os.path.join(here, "..", "Data")

    distance_path = os.path.join(data, "processed", "wc2026_team_distance_CLEAN.csv")
    duration_path = os.path.join(data, "reference", "match_time_played.csv")

    for path in (distance_path, duration_path):
        if not os.path.exists(path):
            raise SystemExit("MISSING FILE: %s" % path)

    distance = pd.read_csv(distance_path)
    duration = pd.read_csv(duration_path)

    missing = [c for c in REQUIRED if c not in distance.columns]
    if missing:
        raise SystemExit(
            "MISSING COLUMNS in wc2026_team_distance_CLEAN.csv: %s" % missing)
    if "time_played_min" not in duration.columns:
        raise SystemExit(
            "MISSING COLUMN time_played_min in match_time_played.csv")

    df = distance[["match_id", "result_id", "date", "stage", "team", "distance_km"]].merge(
        duration[["match_id", "time_played_min"]], on="match_id", how="left")

    if df["time_played_min"].isna().any():
        raise SystemExit(
            "STOP: %d team-match rows got no duration from the join. "
            "Do not plot this." % int(df["time_played_min"].isna().sum()))

    df["distance_per_90"] = df["distance_km"] / df["time_played_min"] * 90

    group = df[df["stage"] == "First Stage"]
    elim = df[df["stage"] != "First Stage"]
    return df, group, elim


def draw_samples(group, elim):
    """Re-derive the exact samples hypothesis.py tests.

    Matches are drawn, not rows: every match has two team rows that share a
    pitch, a referee and the same extra time, so a match is the sampling unit.
    25 group matches give 50 rows, 15 elimination matches give 30.
    """

    m1 = 25
    m2 = 15

    g_ids = group["match_id"].drop_duplicates().sample(n=m1, random_state=SEED)
    e_ids = elim["match_id"].drop_duplicates().sample(n=m2, random_state=SEED)

    g_sample = group[group["match_id"].isin(g_ids)]["distance_per_90"]
    e_sample = elim[elim["match_id"].isin(e_ids)]["distance_per_90"]

    # every drawn match must contribute both of its rows
    for name, values, wanted in (("group", g_sample, m1 * 2),
                                 ("elimination", e_sample, m2 * 2)):
        if len(values) != wanted:
            raise SystemExit(
                "SAMPLE ERROR: %s sample has %d rows, expected %d"
                % (name, len(values), wanted))

    return g_sample, e_sample, m1, m2


# ===== FIGURE 1 - THE ANALYSIS SAMPLE, BOX PLOT =============================

def figure_descriptive_sample(g_sample, e_sample, m1, m2):
    """Box plot of the analysis sample, with every point shown."""

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    data = [g_sample.values, e_sample.values]
    colors = [GROUP_COLOR, ELIM_COLOR]
    labels = ["Group stage", "Elimination stage"]

    boxes = ax.boxplot(
        data, patch_artist=True, widths=0.45, showfliers=False,
        medianprops=dict(color=NAVY, linewidth=2.2),
        whiskerprops=dict(color=GREY, linewidth=1.4),
        capprops=dict(color=GREY, linewidth=1.4),
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.25)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.8)

    # every observation, jittered sideways so overlapping points separate.
    # A fixed generator keeps the jitter identical between runs; it affects
    # the drawing only, never a statistic.
    jitter_rng = np.random.default_rng(SEED)
    for position, (values, color) in enumerate(zip(data, colors), start=1):
        x = position + jitter_rng.uniform(-0.13, 0.13, size=len(values))
        ax.scatter(x, values, s=26, color=color, alpha=0.75,
                   edgecolor=WHITE, linewidth=0.6, zorder=3)

    # n and sd against each box
    top = max(g_sample.max(), e_sample.max())
    for position, (values, matches) in enumerate(
            zip([g_sample, e_sample], [m1, m2]), start=1):
        ax.text(
            position, top + 1.2,
            "n = %d  (%d matches)\nsd = %.2f" % (len(values), matches, values.std()),
            ha="center", va="bottom", fontsize=SIZE_NOTE, color=NAVY,
        )

    ax.set_xticklabels(labels, fontsize=SIZE_LABEL)
    ax.set_ylabel("Distance per 90 minutes (km)", fontsize=SIZE_LABEL)
    ax.set_ylim(min(g_sample.min(), e_sample.min()) - 1.5, top + 5.5)
    # keep the two boxes close together rather than marooned at the edges
    ax.set_xlim(0.45, 2.55)
    ax.set_title("Work rate by stage", fontsize=SIZE_TITLE,
                 fontweight="bold", color=NAVY, loc="left", pad=26)
    ax.text(
        0, 1.045,
        "Analysis sample only (%d + %d team-match rows), not all 208 records"
        % (len(g_sample), len(e_sample)),
        transform=ax.transAxes, fontsize=SIZE_NOTE, color=GREY,
    )
    style_axis(ax)

    fig.tight_layout()
    return fig


# ===== FIGURE 2 - NORMALITY OF THE SAMPLE ===================================

def label_text(name, value, side):
    """Pad a line label away from its line: 'left' means text to the right."""
    if side == "left":
        return "  %s %.2f" % (name, value)
    return "%s %.2f  " % (name, value)


def figure_normality_sample(g_sample, e_sample, shapiro_g, shapiro_e, alpha=0.05):
    """One histogram per stage, from the analysis sample.

    The mean AND the median are drawn on each panel, because the gap between
    them is the normality check this unit teaches. Shapiro-Wilk is a formal
    test from outside the syllabus and is kept small and grey to match.
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    low = min(g_sample.min(), e_sample.min())
    high = max(g_sample.max(), e_sample.max())
    bins = np.linspace(low, high, 11)          # shared bins, so shapes compare

    panels = [
        (axes[0], g_sample, "Group stage", GROUP_COLOR, shapiro_g),
        (axes[1], e_sample, "Elimination stage", ELIM_COLOR, shapiro_e),
    ]

    # one headroom for both panels, room for the labels above the tallest bar
    tallest = max(np.histogram(values, bins=bins)[0].max()
                  for _, values, _, _, _ in panels)

    skewed = []

    for ax, values, name, color, shapiro_p in panels:
        ax.hist(values, bins=bins, color=color, alpha=0.75,
                edgecolor=WHITE, linewidth=1.1)
        # headroom: the formal-check note sits at the top, the mean and
        # median labels below it, both clear of the tallest bar
        ax.set_ylim(0, tallest * 1.65)
        top = ax.get_ylim()[1]

        mean = values.mean()
        median = values.median()
        gap, shape = shape_verdict(values)
        if shape != "roughly symmetric":
            skewed.append((name, shape, gap))

        # Different line styles, different label heights, and each label is
        # written on the side facing away from the other line, so nothing is
        # struck through when the two lines nearly touch.
        if mean >= median:
            mean_side, median_side = "left", "right"
        else:
            mean_side, median_side = "right", "left"

        ax.axvline(mean, color=NAVY, linestyle="--", linewidth=2, zorder=4)
        ax.text(mean, top * 0.80, label_text("mean", mean, mean_side),
                ha=mean_side, va="top", fontsize=SIZE_NOTE,
                color=NAVY, fontweight="bold")

        ax.axvline(median, color=GREY, linestyle="-.", linewidth=2, zorder=4)
        ax.text(median, top * 0.68, label_text("median", median, median_side),
                ha=median_side, va="top", fontsize=SIZE_NOTE,
                color=GREY, fontweight="bold")

        # the shape verdict goes in the title, not the formal test's p-value
        ax.set_title("%s   n = %d   %s" % (name, len(values), shape),
                     fontsize=SIZE_TITLE, color=NAVY, pad=12)
        ax.set_xlabel("Distance per 90 minutes (km)", fontsize=SIZE_LABEL)

        ax.text(0.98, 0.99,
                "formal check (beyond unit):\nShapiro-Wilk p = %.4f" % shapiro_p,
                transform=ax.transAxes, ha="right", va="top",
                fontsize=SIZE_NOTE, color=GREY, linespacing=1.4)
        style_axis(ax)

    axes[0].set_ylabel("Number of team-matches", fontsize=SIZE_LABEL)

    # the caption is built from the values, so it cannot name the wrong sample
    if skewed:
        described = " and ".join(
            "%s is %s: its mean sits %.2f km/90 %s the median"
            % (name, shape, abs(gap), "above" if gap > 0 else "below")
            for name, shape, gap in skewed)
        tail = ("so its t-test p-value is reported as approximate."
                if len(skewed) == 1
                else "so those t-test p-values are reported as approximate.")
        caption = described + ", " + tail
    else:
        caption = ("Mean and median sit together in both samples, so both look "
                   "roughly symmetric and neither p-value needs a caveat.")

    fig.text(0.5, 0.02, caption, ha="center", fontsize=SIZE_NOTE, color=GREY)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ===== FIGURE 3 - THE FULL POPULATION, THREE PANELS =========================

def figure_population_context(group, elim):
    """The three-step story, from all 208 records."""

    panels = [
        ("Distance covered", "Kilometres per match", "km",
         group["distance_km"], elim["distance_km"]),
        ("Match length", "Minutes played", "min",
         group["time_played_min"], elim["time_played_min"]),
        ("Work rate", "Distance per 90 minutes (km)", "km/90",
         group["distance_per_90"], elim["distance_per_90"]),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for ax, (title, ylabel, unit, group_values, elim_values) in zip(axes, panels):
        g = group_values.mean()
        e = elim_values.mean()

        bars = ax.bar(["Group\nstage", "Elimination\nstage"], [g, e],
                      color=[GROUP_COLOR, ELIM_COLOR], width=0.55)

        # y starts at zero. Panel 3's gap is small and that is the honest
        # picture; the annotation carries it, not a cut axis.
        ax.set_ylim(0, max(g, e) * 1.28)

        for bar, value in zip(bars, [g, e]):
            ax.text(bar.get_x() + bar.get_width() / 2, value * 1.02,
                    "%.2f" % value, ha="center", va="bottom",
                    fontsize=SIZE_ANNOT, fontweight="bold", color=NAVY)

        ax.text(0.5, 0.95, "%+.2f %s" % (e - g, unit),
                transform=ax.transAxes, ha="center", va="top",
                fontsize=SIZE_ANNOT, fontweight="bold", color=NAVY)

        ax.set_title(title, fontsize=SIZE_TITLE, color=NAVY, pad=12)
        ax.set_ylabel(ylabel, fontsize=SIZE_LABEL)
        ax.tick_params(axis="x", labelsize=SIZE_LABEL)
        style_axis(ax)

    fig.text(0.5, 0.02,
             "All %d team-match records of the tournament (the full population), "
             "not a sample" % (len(group) + len(elim)),
             ha="center", fontsize=SIZE_NOTE, color=GREY)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ===== DRIVER ===============================================================

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df, group, elim = load_data()
    g_sample, e_sample, m1, m2 = draw_samples(group, elim)

    shapiro_g = st.shapiro(g_sample)[1]
    shapiro_e = st.shapiro(e_sample)[1]

    print("=" * 70)
    print("VALUES ON THE FIGURES - check these against the slides")
    print("=" * 70)

    print("\nfig_descriptive_sample.png   SOURCE: analysis sample")
    print("  Group stage       : n = %d rows from %d distinct matches" % (len(g_sample), m1))
    print("    mean            : %.2f km/90" % g_sample.mean())
    print("    median          : %.2f km/90" % g_sample.median())
    print("    sd              : %.2f" % g_sample.std())
    print("    min / max       : %.2f / %.2f" % (g_sample.min(), g_sample.max()))
    print("    Q1 / Q3         : %.2f / %.2f"
          % (g_sample.quantile(0.25), g_sample.quantile(0.75)))
    print("  Elimination stage : n = %d rows from %d distinct matches" % (len(e_sample), m2))
    print("    mean            : %.2f km/90" % e_sample.mean())
    print("    median          : %.2f km/90" % e_sample.median())
    print("    sd              : %.2f" % e_sample.std())
    print("    min / max       : %.2f / %.2f" % (e_sample.min(), e_sample.max()))
    print("    Q1 / Q3         : %.2f / %.2f"
          % (e_sample.quantile(0.25), e_sample.quantile(0.75)))

    print("\nfig_normality_sample.png     SOURCE: analysis sample (same draw)")
    for name, values, shapiro_p in (("Group stage", g_sample, shapiro_g),
                                    ("Elimination stage", e_sample, shapiro_e)):
        gap, shape = shape_verdict(values)
        agree = (shapiro_p >= 0.05) == (shape == "roughly symmetric")
        print("  %-18s: n = %d" % (name, len(values)))
        print("    mean / median   : %.2f / %.2f km/90"
              % (values.mean(), values.median()))
        print("    gap             : %+.2f km/90  ->  %s   (taught check, "
              "threshold %.2f)" % (gap, shape, SYMMETRY_GAP))
        print("    Shapiro-Wilk p  : %.4f  ->  %s   (formal check, beyond unit)"
              % (shapiro_p, "FAILS normality" if shapiro_p < 0.05
                 else "passes normality"))
        print("    the two checks  : %s" % ("agree" if agree else "DISAGREE"))

    print("\nfig_population_context.png   SOURCE: all %d records (population)" % len(df))
    for label, unit, col in (("Distance covered", "km", "distance_km"),
                             ("Match length", "min", "time_played_min"),
                             ("Work rate", "km/90", "distance_per_90")):
        g = group[col].mean()
        e = elim[col].mean()
        print("  %-18s: group %7.2f %-5s  elimination %7.2f %-5s  difference %+.2f"
              % (label, g, unit, e, unit, e - g))
    print("  Row counts        : group %d, elimination %d" % (len(group), len(elim)))

    figures = [
        ("fig_descriptive_sample.png",
         figure_descriptive_sample(g_sample, e_sample, m1, m2)),
        ("fig_normality_sample.png",
         figure_normality_sample(g_sample, e_sample, shapiro_g, shapiro_e)),
        ("fig_population_context.png",
         figure_population_context(group, elim)),
    ]

    print()
    for name, fig in figures:
        path = os.path.join(FIG_DIR, name)
        fig.savefig(path, dpi=200)
        plt.close(fig)
        print("wrote figures/%s" % name)


if __name__ == "__main__":
    main()
