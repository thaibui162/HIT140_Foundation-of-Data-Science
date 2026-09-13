# 0_validate_sources.py - Check the Kaggle player file against the official FIFA file.

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

players = pd.read_csv(ROOT / "Dataset" / "players.csv")
matches = pd.read_csv(ROOT / "Dataset" / "team_match.csv")

print("players.csv (Kaggle):", players.shape)
print("team_match.csv (FIFA):", matches.shape)

results = []

def check(name, got, expect):
    ok = "OK" if got == expect else "CHECK"
    print(name, "->", got, "- expect", expect, "-", ok)
    results.append([name, got, expect, ok])


# Structure - does each file hold what it claims to hold?
# -------------------------------------------------------------
print("\nSTRUCTURE")
check("unique teams, Kaggle", players["team"].nunique(), 48)
check("unique teams, FIFA", matches["team"].nunique(), 48)
check("unique matches, FIFA", matches["match_id"].nunique(), 104)
check("rows, FIFA (104 x 2)", len(matches), 208)
check("opponent names not in team", len(set(matches["opponent"]) - set(matches["team"])), 0)


# Check team names
# -------------------------------------------------------------
print("\nTEAM NAMES")
print("in Kaggle only:", set(players["team"]) - set(matches["team"]))
print("in FIFA only:", set(matches["team"]) - set(players["team"]))

name_fix = {"Bosnia–Herz": "Bosnia and Herzegovina", "United States": "USA"}
players["team"] = players["team"].replace(name_fix)
assert set(players["team"]) == set(matches["team"]), "team names still unmapped"
print("all 48 names match")


# Check total goals and cards
# -------------------------------------------------------------
# Kaggle puts an own goal in the scorer's own_goals column, while FIFA credits it to the team that benefits.
print("\nTOTAL GOALS")
check("goals", int(players["goals"].sum() + players["own_goals"].sum()), int(matches["goals_for"].sum()))
check("yellow cards", int(players["cards_yellow"].sum()), int(matches["yellow_cards"].sum()))
check("red cards", int(players["cards_red"].sum()), int(matches["red_cards"].sum()))


# Check team by team the goals conceded and matches played
# -------------------------------------------------------------
print("\nTEAM BY TEAM")
conceded_kaggle = players.groupby("team")["gk_goals_against"].sum()
conceded_fifa = matches.groupby("team")["goals_against"].sum()
check("teams agreeing on goals conceded", int((conceded_kaggle == conceded_fifa).sum()), 48)

played_kaggle = players.groupby("team")["games"].max()
played_fifa = matches.groupby("team")["match_id"].nunique()
check("teams agreeing on matches played", int((played_kaggle == played_fifa).sum()), 48)


# Check which teams advanced
# -------------------------------------------------------------
# FIFA: the team appears in the Round of 32.
# Kaggle: a player has more than 3 appearances, so the team played beyond the group stage.
print("\nTEAMS THAT ADVANCED")
qualified_fifa = set(matches.loc[matches["stage"] == "Round of 32", "team"])
qualified_kaggle = set(played_kaggle[played_kaggle > 3].index)
check("teams the two sources disagree on", len(qualified_fifa ^ qualified_kaggle), 0)


# Check shots on target faced
# -------------------------------------------------------------
print("\nSHOTS ON TARGET FACED")
matches["sot_against"] = matches.groupby("match_id")["shots_on_target"].transform(lambda s: s[::-1].values)
faced_kaggle = players.groupby("team")["gk_shots_on_target_against"].sum()
faced_fifa = matches.groupby("team")["sot_against"].sum()
check("total SoT faced", int(faced_kaggle.sum()), int(faced_fifa.sum()))
check("teams agreeing on SoT faced", int((faced_kaggle == faced_fifa).sum()), 48)
print((faced_kaggle - faced_fifa)[lambda s: s != 0])


# Summary
# -------------------------------------------------------------
print("\nSUMMARY")
print(pd.DataFrame(results, columns=["test", "got", "expect", "result"]).to_string(index=False))