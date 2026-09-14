import pandas as pd
import numpy as np

df = pd.read_csv("world_cup_2026_team_match_raw.csv")   #create data frame
df["date"] = pd.to_datetime(df["date"])                     #change datetime type

print(df.head())                    #show first 5 rows to check data
print(df.shape)                     #check the dimensions of a DataFrame
print(df["match_id"].nunique())     #check matches
print(df.isna().sum().sum())        #check missing value
print(df.duplicated(["match_id", "team"]).sum()) #check duplicate match
df.info()                           #check data type

#create shot accurancy variable

df["shot_accuracy_pct"] = np.where(df["shots"] > 0,
                                   df["shots_on_target"] / df["shots"] * 100,0).round(2)

df["pass_accuracy_pct"] = (df["passes_completed"] / df["passes"] * 100).round(2)
df["result"] = df["result"].str.strip().str.title()
df["won"] = (df["result"] == "Win").astype(int)

# Create tournament phase variable
df["phase"] = np.where(df["stage"].str.startswith("Group"), "Group stage","Knockout stage")

# Save complete cleaned dataset
df.to_csv("world_cup_2026_team_match_cleaned1.csv",index=False)

# Randomly sample 60% of matches
sampled_match_ids = df["match_id"].drop_duplicates().sample(frac=0.60, random_state=2026)

sample_df = df[df["match_id"].isin(sampled_match_ids)]

sample_df = sample_df.sort_values(["match_id", "team"]).reset_index(drop=True)

print("Sample shape:", sample_df.shape)
print("Sample matches:", sample_df["match_id"].nunique())
print(sample_df.groupby("phase")["match_id"].nunique())
print(sample_df["result"].value_counts())
print((df["shots_on_target"] > df["shots"]).sum())


# This file is created automatically
sample_df.to_csv("world_cup_2026_team_match_sampled.csv", index=False)