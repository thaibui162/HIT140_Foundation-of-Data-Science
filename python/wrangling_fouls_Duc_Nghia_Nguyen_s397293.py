import pandas as pd

df = pd.read_csv("world_cup_2026_fouls_raw.csv")

# Calculate total fouls committed in each match
df["FoulsCommitted"] = (
    df["HomeFoulsCommitted"]
    + df["AwayFoulsCommitted"]
)

print(df[[
    "MatchID",
    "HomeTeam",
    "AwayTeam",
    "HomeFoulsCommitted",
    "AwayFoulsCommitted",
    "FoulsCommitted"
]].head())

df.to_csv(
    "world_cup_2026_fouls_match_level.csv",
    index=False
)