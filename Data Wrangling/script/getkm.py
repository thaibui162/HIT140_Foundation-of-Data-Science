import os

import requests
import pandas as pd
import time

# =====================================================
# 0. Where the output goes
# =====================================================
# Resolved from this file's location so the script can be run from anywhere.

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# NOTE: this script deliberately does NOT write to
#   datasets/wc2026_team_distance.csv
# That file is the ORIGINAL shipped extract. It is the raw evidence the
# wrangling report is written about - it is the file that carries the
# numeric team codes, and wrangle_wc2026.py reads it as its input. An
# earlier version of this script pointed here, which would have silently
# overwritten that evidence with an already-corrected extract and left the
# report describing a defect no longer present in its own source file.
#
# The re-fetch is written alongside it instead.

REFETCH_PATH = os.path.join(
    DATA_ROOT,
    "datasets",
    "wc2026_team_distance_REFETCH.csv"
)

# TimePlayed is a property of the MATCH, not of a team, so it is also
# emitted as a match-level reference table for wrangle_wc2026.py to join.

TIME_PATH = os.path.join(
    DATA_ROOT,
    "reference",
    "match_time_played.csv"
)

# =====================================================
# 1. Get World Cup 2026 match list
# =====================================================

matches_url = (
    "https://raw.githubusercontent.com/"
    "Bustami/efi-fifa-data-wc-2026/"
    "master/data/wc2026_matches.csv"
)

# keep_default_na=False is essential, not cosmetic.
#
# This file writes the literal string "NA" into home_team / away_team /
# home_team_id / away_team_id for every knockout fixture, because the bracket
# was not yet resolved when it was published. Pandas' default na_values list
# includes "NA", so the default read turns those cells into NaN. The team-name
# lookup below then finds nothing and silently falls back to the raw numeric
# team id - which is exactly how the knockout rows of the shipped dataset
# ended up reading "43883" instead of "South Africa".
#
# Reading NA as a plain string makes that case explicit and testable.

matches = pd.read_csv(
    matches_url,
    keep_default_na=False,
    na_values=[]
)

MISSING = {"", "NA", "N/A", "nan", "NaN", "None"}


def is_missing(value):
    return str(value).strip() in MISSING


# =====================================================
# 2. Build one team_id -> team_name table for the WHOLE tournament
# =====================================================
# The knockout rows of the match list carry no team names, but the 72
# group-stage rows name all 48 teams and give each one its FIFA team id.
# Those are the same ids the stats API returns for knockout matches, so a
# single table built once resolves every match in the tournament - including
# the ones whose own row says "NA".
#
# This is a direct lookup against FIFA's own identifiers. There is no
# inference from row order or fixture evidence anywhere in it.

team_names = {}

for _, match in matches.iterrows():

    for id_column, name_column in (
        ("home_team_id", "home_team"),
        ("away_team_id", "away_team"),
    ):

        team_id = match[id_column]
        team_name = match[name_column]

        if is_missing(team_id) or is_missing(team_name):
            continue

        team_names[str(int(float(team_id)))] = str(team_name).strip()

print(f"Resolved {len(team_names)} team ids from the match list")

# =====================================================
# 3. Store results
# =====================================================

def time_played_from_players(result_id):
    """Recover match duration from the player-level endpoint.

    The team-level response occasionally omits TimePlayed. The player
    endpoint carries the same field per player, and the maximum across
    players - the ever-present players, who were on the pitch for the
    whole match - equals the team-level match duration. That equality was
    checked against 12 matches whose team-level value IS present, spanning
    regulation and extra time, and held exactly in every one.

    Returns None if the endpoint gives nothing usable, so the caller can
    record a genuine gap rather than a fabricated number.
    """
    url = (
        f"https://fdh-api.fifa.com/"
        f"v1/stats/match/{result_id}/players.json"
    )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30
        )
        response.raise_for_status()
        data = response.json()

    except Exception as error:
        print(f"  player-level lookup failed for {result_id}: {error}")
        return None

    times = []

    for stats in data.values():
        values = {stat[0]: stat[1] for stat in stats}
        if "TimePlayed" in values:
            times.append(float(values["TimePlayed"]))

    if not times:
        return None

    return max(times)


results = []
unresolved = set()
missing_time = set()
recovered_time = {}

# =====================================================
# 4. Go through every match
# =====================================================

for _, match in matches.iterrows():
    if is_missing(match["result_id"]):
        continue

    result_id = int(float(match["result_id"]))

    url = (
        f"https://fdh-api.fifa.com/"
        f"v1/stats/match/{result_id}/teams.json"
    )

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

        # =================================================
        # Find TotalDistance and TimePlayed
        # =================================================
        # The response carries ~140 stat keys per team as [name, value]
        # pairs. Collapse it to a dict once and read the wanted keys by
        # name, rather than re-scanning the list for each one.
        #
        # TimePlayed is total elapsed playing time for the match in
        # minutes, INCLUDING stoppage time and extra time: a regulation
        # fixture reports ~101, a match taken to extra time ~140. It is
        # the same value for both teams in a match.
        #
        # Without it, distance_km is not comparable across stages. A
        # knockout team covering 160 km in 145 minutes is doing LESS work
        # per minute than a group-stage team covering 107 km in 101, but
        # on raw kilometres alone it reads as the tournament's hardest
        # workload. Every stage-to-stage comparison needs this column.

        for team_id, stats in data.items():

            stat_values = {stat[0]: stat[1] for stat in stats}

            if "TotalDistance" not in stat_values:
                continue

            distance_km = float(stat_values["TotalDistance"]) / 1000

            time_played = stat_values.get("TimePlayed")

            if time_played is None:
                # Fall back to the player-level endpoint before giving up.
                # Recorded either way - a silent 90 here would be
                # indistinguishable from a real 90-minute match.
                if result_id not in recovered_time:
                    recovered_time[result_id] = time_played_from_players(
                        result_id
                    )
                    if recovered_time[result_id] is not None:
                        print(
                            f"  recovered TimePlayed for {result_id} from "
                            f"player data: "
                            f"{round(recovered_time[result_id], 2)} min"
                        )

                time_played = recovered_time[result_id]

                if time_played is None:
                    missing_time.add(result_id)
                else:
                    time_played = round(float(time_played), 2)
            else:
                time_played = round(float(time_played), 2)

            team = team_names.get(str(team_id))

            if team is None:
                # Falling back to the raw id is what corrupted the
                # original extract. Keep the fallback so no row is
                # lost, but record it so it cannot pass unnoticed.
                team = str(team_id)
                unresolved.add(str(team_id))

            results.append({
                "match_id": match["match_id"],
                "result_id": result_id,
                "date": match["date"],
                "stage": match["stage"],
                "team": team,
                "distance_km": round(distance_km, 2),
                "time_played_min": time_played
            })

            print(
                match["match_id"],
                team,
                round(distance_km, 2),
                "km in",
                time_played,
                "min"
            )

        # Small delay between requests
        time.sleep(0.2)

    except Exception as e:
        print(
            "Failed:",
            result_id,
            match["home_team"],
            "vs",
            match["away_team"],
            e
        )

# =====================================================
# 5. Create DataFrame
# =====================================================

distance_df = pd.DataFrame(results)

print("\nTEAM DISTANCE DATA:")
print(distance_df)

# =====================================================
# 6. Report any team that could not be named
# =====================================================

if unresolved:
    print(
        f"\nWARNING: {len(unresolved)} team id(s) had no name and were "
        f"written as raw ids: {sorted(unresolved)}"
    )
    print("The `team` column is not fully resolved - do not analyse as is.")
else:
    print("\nAll team ids resolved to names.")

recovered = {k: v for k, v in recovered_time.items() if v is not None}

if recovered:
    print(
        f"\nNOTE: {len(recovered)} match(es) had no team-level TimePlayed "
        f"and were recovered from player data: {recovered}"
    )

if missing_time:
    print(
        f"\nWARNING: {len(missing_time)} match(es) returned no TimePlayed: "
        f"{sorted(missing_time)}"
    )
    print("Those rows carry a blank duration and cannot be normalised.")
else:
    print("All matches reported TimePlayed.")


# =====================================================
# 7. Build the match-level TimePlayed reference
# =====================================================
# TimePlayed describes the match, so the two rows of a match must agree.
# Checking that is not ceremony: a disagreement would mean the value is
# per-team (e.g. minutes on the pitch) rather than match duration, which
# would make the km/90 normalisation built on it wrong.

time_rows = []
inconsistent = []

for (match_id, result_id), group in distance_df.groupby(
    ["match_id", "result_id"], sort=False
):

    values = group["time_played_min"].dropna().unique()

    if len(values) > 1:
        inconsistent.append((match_id, sorted(values)))

    time_rows.append({
        "match_id": match_id,
        "result_id": result_id,
        "time_played_min": values[0] if len(values) else None
    })

time_df = pd.DataFrame(time_rows)

if inconsistent:
    print(
        f"\nWARNING: {len(inconsistent)} match(es) reported DIFFERENT "
        f"TimePlayed for their two teams: {inconsistent}"
    )
    print("TimePlayed is not match duration - do not normalise on it.")
else:
    print(
        f"TimePlayed agrees across both teams in all "
        f"{len(time_df)} matches."
    )

print(
    "\nMatch duration (min): "
    f"min {time_df['time_played_min'].min()}, "
    f"median {time_df['time_played_min'].median()}, "
    f"max {time_df['time_played_min'].max()}"
)

# =====================================================
# 8. Save CSVs
# =====================================================

for path in (REFETCH_PATH, TIME_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)

distance_df.to_csv(
    REFETCH_PATH,
    index=False
)

time_df.to_csv(
    TIME_PATH,
    index=False
)

print(f"\nWrote {len(distance_df)} rows -> {REFETCH_PATH}")
print(f"Wrote {len(time_df)} rows -> {TIME_PATH}")
