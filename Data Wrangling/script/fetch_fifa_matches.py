#!/usr/bin/env python3
"""
=============================================================================
 STAGE 1 - fetch_fifa_matches.py
 Acquire the 2026 FIFA World Cup match schedule from a live source and
 export it to CSV.
=============================================================================

WHY THIS SCRIPT EXISTS
----------------------
The dataset `wc2026_team_distance.csv` identifies teams by numeric code in
all knockout rows. To resolve those codes we need an authoritative reference
of "which match_id was which fixture" - ideally including each team's ID.

This script builds that reference by fetching it from source, so the mapping
is reproducible rather than hand-entered.

SOURCES (in priority order)
---------------------------
1. FIFA Data API  (api.fifa.com/api/v3)
   Competition 17 = FIFA World Cup, Season 285023 = 2026 edition.
   These IDs were read from FIFA's own match-centre URLs, e.g.
     fifa.com/en/match-centre/match/17/285023/289287/400021518
                                     ^^  ^^^^^^        ^^^^^^^^^
                                   comp  season         match_id
   BEST CASE: FIFA returns `IdTeam` for each side. FIFA's national-team IDs
   are 5-digit values in the same range as the codes in our dataset, so a
   successful FIFA fetch converts our positional inference into a DIRECT
   code -> name lookup. That is the strongest possible evidence.

2. ESPN Site API  (site.api.espn.com) - fallback
   Reliable and complete, but ESPN uses its OWN ids (3-digit team ids,
   6-digit event ids). Useful for fixtures/scores, useless for a direct
   code join. If this fallback is used, Stage 2 falls back to matching on
   home/away position instead.

Every output row records which source produced it, so the two are never
silently mixed.

USAGE
-----
  python3 scripts/fetch_fifa_matches.py                     # auto (FIFA, then ESPN)
  python3 scripts/fetch_fifa_matches.py --source fifa       # FIFA only
  python3 scripts/fetch_fifa_matches.py --source espn       # ESPN only
  python3 scripts/fetch_fifa_matches.py --out my_file.csv
  python3 scripts/fetch_fifa_matches.py --verbose

OUTPUT
------
  data/reference/fifa_matches_fetched.csv
  columns: match_id, stage, date_utc, home_team_id, home_team, home_score,
           away_team_id, away_team, away_score, venue, source

NOTE ON ENVIRONMENT
-------------------
This script requires outbound internet access. It was authored in a sandbox
whose egress is proxied and therefore could NOT be executed end-to-end there;
run it on your own machine. The parsing logic is defensive: if a source is
unreachable or its response shape differs, the script says so explicitly
rather than emitting a half-built file.

Standard library only. Python 3.7+.
=============================================================================
"""

import argparse
import csv
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
FIFA_COMPETITION_ID = "17"        # FIFA World Cup
FIFA_SEASON_ID = "285023"         # 2026 edition (Canada/Mexico/USA)
FIFA_BASE = "https://api.fifa.com/api/v3"

ESPN_BASE = ("https://site.api.espn.com/apis/site/v2/sports/soccer/"
             "fifa.world/scoreboard")

TOURNAMENT_START = date(2026, 6, 11)
TOURNAMENT_END = date(2026, 7, 19)

# Resolved from this file's location so the script runs from anywhere.
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(DATA_ROOT, "reference", "fifa_matches_fetched.csv")

OUTPUT_COLUMNS = [
    "match_id", "stage", "date_utc",
    "home_team_id", "home_team", "home_score",
    "away_team_id", "away_team", "away_score",
    "venue", "source",
]

USER_AGENT = "Mozilla/5.0 (compatible; HIT140-coursework-fetch/1.0)"
TIMEOUT = 30
RETRIES = 3
RETRY_BACKOFF = 2.0

# --------------------------------------------------------------------------
# WRANGLING RULE 1 - country name harmonisation
# --------------------------------------------------------------------------
# Different providers spell the same nation differently. The target dataset
# (wc2026_team_distance.csv) already uses a fixed set of spellings in its
# group-stage rows. Every fetched name is normalised to THAT convention, so
# the reference table joins cleanly and one country never becomes two values.
NAME_NORMALISATION = {
    "cape verde": "Cabo Verde",
    "cabo verde": "Cabo Verde",
    "dr congo": "Congo DR",
    "congo dr": "Congo DR",
    "democratic republic of congo": "Congo DR",
    "congo democratic republic": "Congo DR",
    "united states": "USA",
    "united states of america": "USA",
    "usa": "USA",
    "us": "USA",
    "ivory coast": "Côte d'Ivoire",
    "cote d'ivoire": "Côte d'Ivoire",
    "côte d'ivoire": "Côte d'Ivoire",
    "south korea": "Korea Republic",
    "korea republic": "Korea Republic",
    "korea, republic of": "Korea Republic",
    "iran": "IR Iran",
    "ir iran": "IR Iran",
    "islamic republic of iran": "IR Iran",
    "turkey": "Türkiye",
    "türkiye": "Türkiye",
    "turkiye": "Türkiye",
    "czech republic": "Czechia",
    "czechia": "Czechia",
    "bosnia-herzegovina": "Bosnia and Herzegovina",
    "bosnia and herzegovina": "Bosnia and Herzegovina",
    "bosnia & herzegovina": "Bosnia and Herzegovina",
    "curacao": "Curaçao",
    "curaçao": "Curaçao",
    "saudi arabia": "Saudi Arabia",
    "new zealand": "New Zealand",
    "south africa": "South Africa",
}


def normalise_team_name(raw):
    """Map a provider's spelling onto the dataset's own convention."""
    if not raw:
        return ""
    cleaned = " ".join(str(raw).strip().split())
    return NAME_NORMALISATION.get(cleaned.lower(), cleaned)


# --------------------------------------------------------------------------
# WRANGLING RULE 2 - stage label harmonisation
# --------------------------------------------------------------------------
# The target dataset labels stages: First Stage, Round of 32, Round of 16,
# Quarter-final, Semi-final, Play-off for third place, Final.
STAGE_NORMALISATION = {
    "first stage": "First Stage",
    "group stage": "First Stage",
    "group": "First Stage",
    "round of 32": "Round of 32",
    "last 32": "Round of 32",
    "round of 16": "Round of 16",
    "last 16": "Round of 16",
    "quarter-final": "Quarter-final",
    "quarter-finals": "Quarter-final",
    "quarterfinal": "Quarter-final",
    "quarter final": "Quarter-final",
    "semi-final": "Semi-final",
    "semi-finals": "Semi-final",
    "semifinal": "Semi-final",
    "semi final": "Semi-final",
    "play-off for third place": "Play-off for third place",
    "third place play-off": "Play-off for third place",
    "third-place play-off": "Play-off for third place",
    "bronze final": "Play-off for third place",
    "3rd place": "Play-off for third place",
    "final": "Final",
}


def normalise_stage(raw):
    if not raw:
        return ""
    cleaned = " ".join(str(raw).strip().split())
    return STAGE_NORMALISATION.get(cleaned.lower(), cleaned)


# FIFA IdStage values observed in FIFA's own match-centre URLs. Used only as
# a hint when the API response does not carry a readable stage name.
FIFA_STAGE_HINTS = {
    "289287": "Round of 32",
    "289288": "Round of 16",
    "289289": "Quarter-final",
    "289290": "Semi-final",
    "289291": "Play-off for third place",
    "289292": "Final",
}


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
def http_get_json(url, verbose=False):
    """GET a URL and parse JSON, with retries. Raises on final failure."""
    last_error = None
    for attempt in range(1, RETRIES + 1):
        try:
            if verbose:
                print(f"    GET (attempt {attempt}) {url}", file=sys.stderr)
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            })
            context = ssl.create_default_context()
            with urllib.request.urlopen(request, timeout=TIMEOUT,
                                        context=context) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:                      # noqa: BLE001
            last_error = exc
            if attempt < RETRIES:
                time.sleep(RETRY_BACKOFF * attempt)
    raise RuntimeError(f"GET failed after {RETRIES} attempts: {url}\n"
                       f"  last error: {type(last_error).__name__}: {last_error}")


# --------------------------------------------------------------------------
# SOURCE 1 - FIFA
# --------------------------------------------------------------------------
def _fifa_text(value):
    """
    FIFA returns localised strings as [{'Locale': 'en-GB', 'Description': ...}].
    Prefer English; fall back to the first entry; accept a plain string.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value:
        for entry in value:
            if isinstance(entry, dict) and str(entry.get("Locale", "")).startswith("en"):
                return entry.get("Description", "")
        first = value[0]
        if isinstance(first, dict):
            return first.get("Description", "")
    if isinstance(value, dict):
        return value.get("Description", "")
    return ""


def fetch_from_fifa(verbose=False):
    """
    Pull the full match calendar for competition 17 / season 285023.
    Returns a list of output-row dicts. Raises if unreachable or unparseable.
    """
    print("  [FIFA] requesting match calendar ...", file=sys.stderr)
    rows = []
    page = 0
    url = (f"{FIFA_BASE}/calendar/matches"
           f"?idCompetition={FIFA_COMPETITION_ID}"
           f"&idSeason={FIFA_SEASON_ID}"
           f"&count=500&language=en")

    while url:
        payload = http_get_json(url, verbose=verbose)
        results = payload.get("Results")
        if results is None:
            raise RuntimeError(
                "FIFA response did not contain a 'Results' array. "
                "The API shape may have changed. Top-level keys seen: "
                f"{sorted(payload.keys())[:12]}")

        for match in results:
            home = match.get("Home") or {}
            away = match.get("Away") or {}

            stage = normalise_stage(_fifa_text(match.get("StageName")))
            if not stage:
                stage = FIFA_STAGE_HINTS.get(str(match.get("IdStage", "")), "")
            if not stage and match.get("IdGroup"):
                stage = "First Stage"

            rows.append({
                "match_id": str(match.get("IdMatch", "")).strip(),
                "stage": stage,
                "date_utc": str(match.get("Date", "")).strip(),
                # IdTeam is the prize: FIFA's own numeric team identifier.
                "home_team_id": str(home.get("IdTeam", "") or "").strip(),
                "home_team": normalise_team_name(_fifa_text(home.get("TeamName"))),
                "home_score": home.get("Score", ""),
                "away_team_id": str(away.get("IdTeam", "") or "").strip(),
                "away_team": normalise_team_name(_fifa_text(away.get("TeamName"))),
                "away_score": away.get("Score", ""),
                "venue": _fifa_text((match.get("Stadium") or {}).get("Name")),
                "source": "fifa",
            })

        page += 1
        token = payload.get("ContinuationToken")
        if token and len(results) >= 500 and page < 10:
            url = (f"{FIFA_BASE}/calendar/matches"
                   f"?idCompetition={FIFA_COMPETITION_ID}"
                   f"&idSeason={FIFA_SEASON_ID}"
                   f"&count=500&language=en&continuationToken={token}")
        else:
            url = None

    if not rows:
        raise RuntimeError("FIFA returned zero matches.")

    with_ids = sum(1 for r in rows if r["home_team_id"] and r["away_team_id"])
    print(f"  [FIFA] {len(rows)} matches; {with_ids} carry team IDs.",
          file=sys.stderr)
    return rows


# --------------------------------------------------------------------------
# SOURCE 2 - ESPN (fallback)
# --------------------------------------------------------------------------
def fetch_from_espn(verbose=False):
    """
    Walk the tournament date range and collect every event ESPN reports.
    ESPN ids differ from FIFA's; Stage 2 handles that.
    """
    print("  [ESPN] walking tournament dates ...", file=sys.stderr)
    rows, seen = [], set()
    current = TOURNAMENT_START
    while current <= TOURNAMENT_END:
        url = f"{ESPN_BASE}?dates={current.strftime('%Y%m%d')}"
        try:
            payload = http_get_json(url, verbose=verbose)
        except RuntimeError as exc:
            print(f"    ! {current}: {exc}", file=sys.stderr)
            current += timedelta(days=1)
            continue

        for event in payload.get("events", []):
            event_id = str(event.get("id", ""))
            if event_id in seen:
                continue
            seen.add(event_id)

            competition = (event.get("competitions") or [{}])[0]
            competitors = competition.get("competitors") or []
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})

            stage = normalise_stage(
                ((event.get("season") or {}).get("type") or {}).get("name")
                or (competition.get("notes") or [{}])[0].get("headline", "")
            )

            rows.append({
                "match_id": event_id,          # NB: ESPN's id, not FIFA's
                "stage": stage,
                "date_utc": str(event.get("date", "")).strip(),
                "home_team_id": str((home.get("team") or {}).get("id", "")),
                "home_team": normalise_team_name(
                    (home.get("team") or {}).get("displayName")),
                "home_score": home.get("score", ""),
                "away_team_id": str((away.get("team") or {}).get("id", "")),
                "away_team": normalise_team_name(
                    (away.get("team") or {}).get("displayName")),
                "away_score": away.get("score", ""),
                "venue": ((competition.get("venue") or {}).get("fullName") or ""),
                "source": "espn",
            })
        current += timedelta(days=1)

    if not rows:
        raise RuntimeError("ESPN returned zero matches across the date range.")
    print(f"  [ESPN] {len(rows)} matches.", file=sys.stderr)
    return rows


# --------------------------------------------------------------------------
# Post-fetch checks
# --------------------------------------------------------------------------
def audit(rows):
    """Report on completeness. Warnings only - the CSV is still written."""
    print("\n  --- fetch audit ---", file=sys.stderr)
    print(f"  rows fetched          : {len(rows)}", file=sys.stderr)
    if len(rows) != 104:
        print(f"  ! expected 104 matches for the 48-team format, got {len(rows)}",
              file=sys.stderr)

    missing_names = [r["match_id"] for r in rows
                     if not r["home_team"] or not r["away_team"]]
    if missing_names:
        print(f"  ! {len(missing_names)} row(s) missing a team name "
              f"(e.g. {missing_names[:5]}) - unresolved bracket placeholders?",
              file=sys.stderr)

    with_ids = sum(1 for r in rows if r["home_team_id"] and r["away_team_id"])
    print(f"  rows with team IDs    : {with_ids}/{len(rows)}", file=sys.stderr)
    if with_ids == len(rows):
        print("  => Stage 2 can attempt a DIRECT team-code join.", file=sys.stderr)
    else:
        print("  => Stage 2 will need the positional (home/away) join.",
              file=sys.stderr)

    stages = {}
    for r in rows:
        stages[r["stage"] or "(blank)"] = stages.get(r["stage"] or "(blank)", 0) + 1
    print(f"  stages                : {stages}", file=sys.stderr)

    duplicates = len(rows) - len({r["match_id"] for r in rows})
    if duplicates:
        print(f"  ! {duplicates} duplicate match_id value(s)", file=sys.stderr)
    print("  -------------------\n", file=sys.stderr)


def write_csv(rows, path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    rows = sorted(rows, key=lambda r: (r["date_utc"], r["match_id"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in OUTPUT_COLUMNS})
    print(f"  wrote {len(rows)} rows -> {path}", file=sys.stderr)


# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Fetch the 2026 FIFA World Cup match list to CSV.")
    parser.add_argument("--source", choices=["auto", "fifa", "espn"],
                        default="auto",
                        help="auto = try FIFA, fall back to ESPN (default)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--verbose", action="store_true",
                        help="print every HTTP request")
    args = parser.parse_args()

    print("=" * 70, file=sys.stderr)
    print(" STAGE 1 - fetching 2026 FIFA World Cup matches", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    rows, errors = None, []

    if args.source in ("auto", "fifa"):
        try:
            rows = fetch_from_fifa(verbose=args.verbose)
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"FIFA: {exc}")
            print(f"  [FIFA] FAILED: {exc}", file=sys.stderr)
            if args.source == "fifa":
                print("\nFIFA was requested explicitly and failed. "
                      "Re-run with --source espn or --source auto.",
                      file=sys.stderr)
                sys.exit(1)
            print("  [FIFA] falling back to ESPN ...", file=sys.stderr)

    if rows is None and args.source in ("auto", "espn"):
        try:
            rows = fetch_from_espn(verbose=args.verbose)
        except Exception as exc:                       # noqa: BLE001
            errors.append(f"ESPN: {exc}")
            print(f"  [ESPN] FAILED: {exc}", file=sys.stderr)

    if rows is None:
        print("\nAll sources failed. Nothing was written.", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        print("\nIf you are behind a proxy or corporate network, try from a "
              "different connection. The reference CSV shipped with this "
              "project (fifa_matches_REFERENCE.csv) can be used instead.",
              file=sys.stderr)
        sys.exit(1)

    audit(rows)
    write_csv(rows, args.out)
    print("Stage 1 complete. Verify the CSV, then run Stage 2:", file=sys.stderr)
    print(f"  python3 scripts/build_mapping_from_fetch.py --fetched {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
