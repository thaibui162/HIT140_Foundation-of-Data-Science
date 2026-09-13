# _1_data_wrangling.py - Build the goalkeeper dataset used by every later step.
# One row per goalkeeper, with labels and markers. No goalkeeper is dropped except those who never played.

import numpy as np
import pandas as pd
from pathlib import Path

# Step 1. Paths and Load
# -------------------------------------------------
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "Dataset"
PROCESSED = ROOT / "processed"
OUT = PROCESSED / "gk_wrangled.csv"

PROCESSED.mkdir(parents=True, exist_ok=True)
name_fix = {"Bosnia–Herz": "Bosnia and Herzegovina", "United States": "USA"}

# Every count line looks the same, so the number is easy to follow.
def report(label, n): print(f"[wrangling] {label}: {n} rows")
players = pd.read_csv(RAW / "players.csv")
matches = pd.read_csv(RAW / "team_match.csv")

report("loaded players", len(players))    
print("[wrangling] loaded matches:", matches.shape)

# Checks the structure of its own input.
assert matches["match_id"].nunique() == 104, "match file should hold 104 matches"
assert len(matches) == 208, "match file should hold 208 rows (104 x 2 teams)"

# Step 2. Standardise team names
# -------------------------------------------------
players["team"] = players["team"].replace(name_fix)
assert set(players["team"]) == set(matches["team"]), "team names still unmapped"

# Step 3. Team lookup
# -------------------------------------------------
# Reducing to 48 teams rows to makes the merge safer. 
# Merging two files straight away could multiply rows, as match file have 208 rows.
qualified_teams = set(matches.loc[matches["stage"] == "Round of 32", "team"].unique())

lookup = pd.DataFrame({"team": matches["team"].unique()})
lookup["qualified"] = lookup["team"].isin(qualified_teams)
lookup["group_label"] = np.where(lookup["qualified"], "Advanced", "Eliminated")
                    
print("[wrangling] qualified teams:", lookup["qualified"].sum())
assert len(lookup) == 48, f"lookup should hold 48 teams, got {len(lookup)}"
assert lookup["qualified"].sum() == 32, "32 teams should have qualified"
report("team lookup", len(lookup)) 


# Step 4. Keep goalkeepers who actually played
# -------------------------------------------------
# 'gk_games' records what a player actually played; while 'position' is only the role of a player.
# Every team must have at least one keeper.
gk = players[players["gk_games"].notna()].copy()
report("filter to keepers who played", len(gk))

assert gk["team"].nunique() == 48, f"only {gk['team'].nunique()} teams have a keeper"


# Step 6. Merge
# -------------------------------------------------
before = len(gk)
gk = gk.merge(lookup, on="team", how="left", validate="many_to_one")

# Checking for nulls to verify on any unmapped name.
assert len(gk) == before, "merge changed the row count"
assert gk["qualified"].isna().sum() == 0, "some keepers got no label"

report("merge with team lookup", len(gk))
print("[wrangling] split:", gk["group_label"].value_counts().to_dict())


# Step 7. Marker columns
# -------------------------------------------------
# Defined save percentage
gk["has_save_pct"] = gk["gk_save_pct"].notna()

# Team's main goalkeeper (meaning with most minutes played)
primary_rows = (gk.sort_values(["gk_minutes", "gk_shots_on_target_against"], ascending=False).groupby("team").head(1).index)
gk["is_primary_gk"] = gk.index.isin(primary_rows)

# 3. Recompute save% from gk_saves 
sot = gk["gk_shots_on_target_against"].replace(0, np.nan)
gk["save_pct_calc"] = (gk["gk_saves"] / sot * 100).round(2)
gk["counts_add_up"] = (gk["gk_saves"] + gk["gk_goals_against"] == gk["gk_shots_on_target_against"])

assert gk["is_primary_gk"].sum() == 48, f"expected 48 primary keepers, got {gk['is_primary_gk'].sum()}"

# Step 8. Select columns
# -------------------------------------------------
keep_cols = [
    "player", "team", "position", "age", "club",                            # identity
    "qualified", "group_label",                                             # grouping
    "games", "gk_games", "gk_games_starts", "gk_minutes",                   # appearances
    "gk_goals_against", "gk_shots_on_target_against",
    "gk_saves", "gk_save_pct", "gk_clean_sheets",                           # goalkeeping
    "has_save_pct", "is_primary_gk", "save_pct_calc", "counts_add_up",      # markers
]
gk = gk[keep_cols].sort_values(["team", "gk_minutes"], ascending=[True, False])


# Step 9. Export
# -------------------------------------------------
gk.to_csv(OUT, index=False, encoding="utf-8-sig")

print("\n[wrangling] SUMMARY")
print("Goalkeepers:", len(gk))
print("Advanced:", int((gk["group_label"] == "Advanced").sum()))
print("Eliminated:", int((gk["group_label"] == "Eliminated").sum()))
print("Primary keepers:", int(gk["is_primary_gk"].sum()))
print("With a save %:", int(gk["has_save_pct"].sum()))
print("Saved:", gk.shape, "->", OUT.name)
print("Counts not adding up:", int((~gk["counts_add_up"]).sum()))
print("Max gap, my save% vs supplied:", round((gk["save_pct_calc"] - gk["gk_save_pct"]).abs().max(), 2))