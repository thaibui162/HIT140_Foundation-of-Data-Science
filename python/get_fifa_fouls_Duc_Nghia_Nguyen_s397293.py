"""Download FIFA World Cup 2026 fouls as one raw row per match.

Output:
    world_cup_2026_fouls_raw.csv

The FIFA statistics field ``FoulsAgainst`` is retained as fouls committed,
matching the definition used for this analytic task. ``FoulsFor`` (fouls won)
is intentionally excluded because it is not part of the research question.
"""

from pathlib import Path
import time

import pandas as pd
import requests


COMPETITION_ID = "17"
SEASON_ID = "285023"

BASE_API = "https://api.fifa.com/api/v3"
STATS_API = "https://fdh-api.fifa.com/v1/stats"

OUTPUT_FILE = Path(__file__).with_name("world_cup_2026_fouls_raw.csv")


def get_team_name(team):
    """Return the best available English team name."""
    names = team.get("TeamName") or []

    if names:
        return names[0].get("Description", "Unknown")

    return team.get("ShortClubName") or team.get("Abbreviation") or "Unknown"


def get_description(value):
    """Extract Description from FIFA's list-based text fields."""
    if isinstance(value, list) and value:
        return value[0].get("Description")

    return None


def get_team_stat(team_statistics, team_id, stat_name):
    """Extract one statistic for one team from the FIFA team-stat response."""
    statistics = team_statistics.get(str(team_id), [])

    for statistic in statistics:
        if statistic[0] == stat_name:
            return statistic[1]

    return None


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    calendar_response = session.get(
        f"{BASE_API}/calendar/matches",
        params={
            "language": "en",
            "count": 500,
            "idSeason": SEASON_ID,
            "idCompetition": COMPETITION_ID,
        },
        timeout=30,
    )
    calendar_response.raise_for_status()

    matches = calendar_response.json().get("Results", [])
    print("Matches found:", len(matches))

    rows = []
    failed_matches = []

    for match in matches:
        match_id = match.get("IdMatch")
        stats_match_id = match.get("Properties", {}).get("IdIFES")
        home = match.get("Home") or {}
        away = match.get("Away") or {}

        if not match_id or not stats_match_id or not home or not away:
            failed_matches.append(match_id or "Unknown")
            continue

        home_team_id = home.get("IdTeam")
        away_team_id = away.get("IdTeam")
        home_name = get_team_name(home)
        away_name = get_team_name(away)

        stats_url = f"{STATS_API}/match/{stats_match_id}/teams.json"

        try:
            stats_response = session.get(stats_url, timeout=30)
            stats_response.raise_for_status()
            team_statistics = stats_response.json()
        except (requests.RequestException, ValueError) as error:
            print(f"Statistics unavailable for match {match_id}: {error}")
            failed_matches.append(match_id)
            continue

        rows.append(
            {
                "Date": (match.get("Date") or "")[:10],
                "Round": get_description(match.get("StageName")),
                "MatchID": match_id,
                "HomeTeam": home_name,
                "AwayTeam": away_name,
                "HomeFoulsCommitted": get_team_stat(
                    team_statistics, home_team_id, "FoulsAgainst"
                ),
                "AwayFoulsCommitted": get_team_stat(
                    team_statistics, away_team_id, "FoulsAgainst"
                ),
            }
        )

        print(f"Collected {match_id}: {home_name} vs {away_name}")
        time.sleep(0.2)

    raw_fouls = pd.DataFrame(rows)
    raw_fouls.to_csv(OUTPUT_FILE, index=False)

    print("\nRaw rows saved:", len(raw_fouls))
    print("Output file:", OUTPUT_FILE)

    if failed_matches:
        print("Matches not collected:", len(failed_matches))
        print(failed_matches)


if __name__ == "__main__":
    main()
