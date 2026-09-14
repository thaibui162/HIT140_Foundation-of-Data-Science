#!/usr/bin/env python3
"""
=============================================================================
 STAGE 1b - build_reference_csv.py
 Build the fixture reference and the team roster that Stage 2 joins against.
=============================================================================

WHY THIS SCRIPT EXISTS
----------------------
The knockout rows of `wc2026_team_distance.csv` name teams by numeric code
(43883, 43899, ...). Those codes are not arbitrary: they are FIFA team ids,
emitted by the stats API that built the dataset in the first place. See
`getkm.py` - when the match list has no name for a team, the extract falls
back to writing the raw id.

That means the codes can be RESOLVED BY LOOKUP rather than inferred. The
lookup table is built here.

WHERE THE NAMES COME FROM
-------------------------
The same match list `getkm.py` downloads:

  raw.githubusercontent.com/Bustami/efi-fifa-data-wc-2026/
      master/data/wc2026_matches.csv

Its 32 knockout rows carry the literal string "NA" for both teams - the
bracket was unresolved when it was published, which is the whole source of
the problem. But its 72 GROUP-STAGE rows name all 48 teams and give each one
its FIFA team id. One pass over those rows produces a complete team_id ->
team_name table covering every code that appears anywhere in the tournament,
knockout rounds included.

WHY THIS IS NOT CIRCULAR
------------------------
The roster is derived from FIFA's own published identifiers. It does not use
the dataset's row ordering, does not use any hand-entered fixture evidence,
and does not consult `wrangle_wc2026.py`. It is therefore genuinely
independent of the positional derivation, which is what makes the Stage 2
cross-check meaningful rather than self-confirming.

OUTPUTS
-------
  reference/fifa_matches_REFERENCE.csv   104 fixtures (Stage 2 --fetched input)
  reference/team_roster.csv               48 team_id -> team_name rows

USAGE
-----
  python3 build_reference_csv.py
  python3 build_reference_csv.py --offline path/to/wc2026_matches.csv

Standard library only. Python 3.7+.
=============================================================================
"""

import argparse
import csv
import io
import os
import sys
import urllib.request

DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MATCHES_URL = (
    "https://raw.githubusercontent.com/"
    "Bustami/efi-fifa-data-wc-2026/"
    "master/data/wc2026_matches.csv"
)

OUT_REFERENCE = os.path.join(DATA_ROOT, "reference", "fifa_matches_REFERENCE.csv")
OUT_ROSTER = os.path.join(DATA_ROOT, "reference", "team_roster.csv")

USER_AGENT = "Mozilla/5.0 (compatible; HIT140-coursework-fetch/1.0)"
TIMEOUT = 30

GROUP_STAGE = "First Stage"
EXPECTED_MATCHES = 104
EXPECTED_TEAMS = 48

# The upstream file spells "no value" this way. Treated as a string, never as
# a null, so the knockout placeholders stay visible instead of vanishing.
MISSING = {"", "NA", "N/A", "nan", "NaN", "None"}


def is_missing(value):
    return str(value).strip() in MISSING


def rel(path):
    return os.path.relpath(path, os.path.dirname(DATA_ROOT))


def fetch_matches(offline):
    """Return the upstream match list as a list of dicts."""
    if offline:
        if not os.path.exists(offline):
            sys.exit(f"FATAL: --offline file not found: {offline}")
        with open(offline, newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh)), f"local file {offline}"

    request = urllib.request.Request(MATCHES_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            text = response.read().decode("utf-8")
    except Exception as exc:                                # noqa: BLE001
        sys.exit(f"FATAL: could not fetch the match list ({exc}).\n"
                 f"Re-run with --offline <path to wc2026_matches.csv>.")
    return list(csv.DictReader(io.StringIO(text))), MATCHES_URL


def build_roster(matches, log):
    """team_id -> (team_name, number of group matches it was named in)."""
    roster = {}
    counts = {}
    for row in matches:
        for id_column, name_column in (("home_team_id", "home_team"),
                                       ("away_team_id", "away_team")):
            team_id = str(row.get(id_column, "")).strip()
            team_name = str(row.get(name_column, "")).strip()
            if is_missing(team_id) or is_missing(team_name):
                continue
            if team_id in roster and roster[team_id] != team_name:
                sys.exit(f"FATAL: team id {team_id} has two names: "
                         f"{roster[team_id]!r} and {team_name!r}")
            roster[team_id] = team_name
            counts[team_id] = counts.get(team_id, 0) + 1
    log(f"  team ids resolved           : {len(roster)}")
    return roster, counts


def main():
    parser = argparse.ArgumentParser(
        description="Build the fixture reference and team roster for Stage 2.")
    parser.add_argument("--offline", default=None,
                        help="use a local copy of wc2026_matches.csv instead "
                             "of fetching it")
    args = parser.parse_args()

    def log(message=""):
        print(message)

    log("=" * 74)
    log(" STAGE 1b - building the fixture reference and team roster")
    log("=" * 74)

    matches, origin = fetch_matches(args.offline)
    log(f" source  : {origin}")
    log(f" rows    : {len(matches)}")
    log("")

    if len(matches) != EXPECTED_MATCHES:
        log(f"  ! expected {EXPECTED_MATCHES} matches, got {len(matches)}")

    roster, counts = build_roster(matches, log)
    if len(roster) != EXPECTED_TEAMS:
        log(f"  ! expected {EXPECTED_TEAMS} teams, got {len(roster)}")

    named = sum(1 for r in matches if not is_missing(r.get("home_team")))
    log(f"  fixtures with named teams   : {named}")
    log(f"  fixtures still unresolved   : {len(matches) - named} "
        f"(knockout placeholders)")
    log("")

    os.makedirs(os.path.dirname(OUT_REFERENCE), exist_ok=True)

    # ---- fixture reference ------------------------------------------------
    # Column names match Stage 1's own output so either file can be handed to
    # Stage 2 via --fetched.
    with open(OUT_REFERENCE, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["match_id", "stage", "date_utc",
                         "home_team_id", "home_team",
                         "away_team_id", "away_team", "source"])
        for row in matches:
            resolved = not is_missing(row.get("home_team"))
            writer.writerow([
                str(row.get("match_id", "")).strip(),
                str(row.get("stage", "")).strip(),
                str(row.get("date", "")).strip(),
                "" if is_missing(row.get("home_team_id")) else str(row["home_team_id"]).strip(),
                "" if is_missing(row.get("home_team")) else str(row["home_team"]).strip(),
                "" if is_missing(row.get("away_team_id")) else str(row["away_team_id"]).strip(),
                "" if is_missing(row.get("away_team")) else str(row["away_team"]).strip(),
                "upstream_match_list" if resolved else "upstream_unresolved",
            ])
    log(f"  wrote {rel(OUT_REFERENCE)}  ({len(matches)} fixtures)")

    # ---- team roster ------------------------------------------------------
    with open(OUT_ROSTER, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["team_id", "team_name", "n_group_matches_named_in",
                         "source"])
        for team_id in sorted(roster, key=lambda t: roster[t]):
            writer.writerow([team_id, roster[team_id], counts[team_id],
                             "upstream_match_list"])
    log(f"  wrote {rel(OUT_ROSTER)}  ({len(roster)} teams)")


if __name__ == "__main__":
    main()
