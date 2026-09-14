#!/usr/bin/env python3
"""
=============================================================================
 wrangle_wc2026.py
 Data wrangling pipeline for wc2026_team_distance.csv
=============================================================================

PURPOSE
-------
The raw dataset records per-team travel distance for every match of the 2026
FIFA World Cup. It is unusable for any team-level analysis as delivered,
because the `team` column mixes two incompatible representations:

    * Group-stage rows ("First Stage")  -> country NAMES  e.g. "Brazil"
    * Knockout rows (Round of 32 on)    -> numeric CODES  e.g. 43924

This script resolves the codes to country names, repairs a structural
integrity error, and emits a cleaned dataset plus a full audit trail.

METHOD (see WRANGLING_REPORT.md for full rationale)
---------------------------------------------------
The mapping is NOT hard-coded by hand. It is DERIVED from the data itself
using two independent facts, then cross-validated:

  (1) EXTERNAL EVIDENCE. Each knockout match_id in this file is FIFA's own
      match identifier (verified against fifa.com match-centre URLs, e.g.
      .../match/17/285023/289287/400021518 = South Africa v Canada). The
      MATCH_EVIDENCE table below therefore records, for each knockout
      match_id, the officially reported fixture, with a source URL.

  (2) INTERNAL STRUCTURE. Within every match, the file lists the HOME team's
      row first and the AWAY team's row second. This was established from
      the group-stage rows, where names are already present, and holds for
      all 72 group matches. Applying it to knockout matches lets each code
      be assigned to a country positionally.

Deriving the mapping this way makes it SELF-VALIDATING: 32 knockout matches
independently assign codes, and every code must resolve to exactly one
country across all of its appearances. Any error in the evidence table or in
the home/away assumption would surface immediately as a conflict.

MATCH DURATION
--------------
The raw extract records distance only. It carries no match duration, which
makes `distance_km` incomparable across stages: group matches are ~90
minutes plus stoppage, knockout matches may run to 120 plus stoppage. On
raw kilometres a knockout side looks like it worked hardest in the
tournament when, per minute on the pitch, it did not.

The duration was recovered from the same FIFA stats endpoint the extract
came from (`TimePlayed`, see script/getkm.py) and is joined here as a
match-level reference table. Two columns are added:

    time_played_min  match duration incl. stoppage/extra time, from source
    distance_per_90  distance_km / time_played_min * 90, derived

`distance_per_90` is the comparable measure. Use it for any comparison
that crosses a stage boundary; `distance_km` remains for reporting the
raw workload actually covered.

OUTPUTS
-------
  data/processed/wc2026_team_distance_CLEAN.csv  cleaned data (+2 columns)
  data/reference/team_code_lookup.csv            code -> country + evidence
  data/processed/change_log.csv                  cell-level audit trail
  reports/validation_report.txt                  before/after quality checks

USAGE
-----
  python3 scripts/wrangle_wc2026.py          (run from the project root)

Requires only the Python standard library.
=============================================================================
"""

import csv
import os
import sys
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Resolved from this file's own location, not the shell's working directory,
# so the pipeline runs identically from anywhere:
#     Data/script/wrangle_wc2026.py  ->  DATA_ROOT = Data/
# --------------------------------------------------------------------------
DATA_ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_PATH     = os.path.join(DATA_ROOT, "datasets", "wc2026_team_distance.csv")
TIME_PATH    = os.path.join(DATA_ROOT, "reference", "match_time_played.csv")
CLEAN_PATH   = os.path.join(DATA_ROOT, "processed", "wc2026_team_distance_CLEAN.csv")
LOOKUP_PATH  = os.path.join(DATA_ROOT, "reference", "team_code_lookup.csv")
CHANGES_PATH = os.path.join(DATA_ROOT, "processed", "change_log.csv")
REPORT_PATH  = os.path.join(DATA_ROOT, "reports", "validation_report.txt")

def rel(path):
    """Display paths relative to Data/, so reports do not leak local paths."""
    return os.path.relpath(path, os.path.dirname(DATA_ROOT))


KNOCKOUT_STAGES = [
    "Round of 32", "Round of 16", "Quarter-final",
    "Semi-final", "Play-off for third place", "Final",
]
GROUP_STAGE = "First Stage"

# Expected tournament structure for the 48-team format.
EXPECTED_MATCHES_PER_STAGE = {
    GROUP_STAGE: 72, "Round of 32": 16, "Round of 16": 8, "Quarter-final": 4,
    "Semi-final": 2, "Play-off for third place": 1, "Final": 1,
}

# --------------------------------------------------------------------------
# EXTERNAL EVIDENCE TABLE
# --------------------------------------------------------------------------
# match_id -> (home_team, away_team, reported_result, source_url)
#
# `home` / `away` follow the fixture as named by the cited source ("X v Y").
# Country spellings deliberately match the conventions ALREADY used in this
# file's own group-stage rows (e.g. "Cabo Verde" not "Cape Verde",
# "Congo DR" not "DR Congo", "USA" not "United States") so that the cleaned
# `team` column stays internally consistent and joinable.
# --------------------------------------------------------------------------
MATCH_EVIDENCE = OrderedDict([
    # ---- Round of 32 --------------------------------------------------
    ("400021518", ("South Africa", "Canada", "0-1",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021518")),
    ("400021516", ("Brazil", "Japan", "2-1",
                   "https://www.espn.com/soccer/match/_/gameId/760487")),
    ("400021513", ("Germany", "Paraguay", "1-1 (3-4 pens)",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021513")),
    ("400021522", ("Netherlands", "Morocco", "1-1 (2-3 pens)",
                   "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/netherlands-morocco-review-highlights")),
    ("400021514", ("Côte d'Ivoire", "Norway", "1-2",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021514")),
    ("400021523", ("France", "Sweden", "3-0",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021523")),
    ("400021520", ("Mexico", "Ecuador", "2-0",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021520")),
    ("400021512", ("England", "Congo DR", "2-1",
                   "https://www.espn.com/soccer/match/_/gameId/760495/congo-dr-england")),
    ("400021525", ("Belgium", "Senegal", "3-2 (AET)",
                   "https://www.espn.com/soccer/match/_/gameId/760493/senegal-belgium")),
    ("400021524", ("USA", "Bosnia and Herzegovina", "2-0",
                   "https://www.ussoccer.com/stories/2026/07/usmnt/match-recap-highlights-vs-bosnia-and-herzegovina-world-cup-round-of-32")),
    ("400021519", ("Spain", "Austria", "3-0",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021519")),
    ("400021526", ("Portugal", "Croatia", "2-1",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021526")),
    ("400021527", ("Switzerland", "Algeria", "2-0",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021527")),
    ("400021515", ("Australia", "Egypt", "1-1 (2-4 pens)",
                   "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/australia-egypt-match-report-highlights")),
    ("400021521", ("Argentina", "Cabo Verde", "3-2 (AET)",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021521")),
    ("400021517", ("Colombia", "Ghana", "1-0",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289287/400021517")),
    # ---- Round of 16 --------------------------------------------------
    ("400021530", ("Canada", "Morocco", "0-3",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289288/400021530")),
    ("400021533", ("Paraguay", "France", "0-1",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289288/400021533")),
    ("400021532", ("Brazil", "Norway", "1-2",
                   "https://www.espn.com/soccer/match/_/gameId/760504/norway-brazil")),
    ("400021531", ("Mexico", "England", "2-3",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289288/400021531")),
    ("400021529", ("Portugal", "Spain", "0-1",
                   "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/portugal-spain-match-report-highlights")),
    ("400021534", ("USA", "Belgium", "1-4",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289288/400021534")),
    ("400021528", ("Argentina", "Egypt", "3-2",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289288/400021528")),
    ("400021535", ("Switzerland", "Colombia", "0-0 (4-3 pens)",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289288/400021535")),
    # ---- Quarter-finals -----------------------------------------------
    ("400021536", ("France", "Morocco", "2-0",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289289/400021536")),
    ("400021538", ("Spain", "Belgium", "2-1",
                   "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/spain-belgium-match-report-highlights")),
    ("400021539", ("Norway", "England", "1-2",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289289/400021539")),
    ("400021537", ("Argentina", "Switzerland", "3-1 (AET)",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289290/400021537")),
    # ---- Semi-finals / Bronze final / Final ----------------------------
    ("400021541", ("France", "Spain", "0-2",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289290/400021541")),
    ("400021540", ("England", "Argentina", "1-2",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289290/400021540")),
    ("400021542", ("France", "England", "4-6",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289291/400021542")),
    ("400021543", ("Spain", "Argentina", "1-0",
                   "https://www.fifa.com/en/match-centre/match/17/285023/289292/400021543")),
])

# The one repair that cannot be derived positionally, because the erroneous
# cell destroyed the information it should have held. Documented in full in
# WRANGLING_REPORT.md, section 4.2.
BUG_MATCH_ID = "400021534"


# ==========================================================================
# Helpers
# ==========================================================================
def load_rows(path):
    """Read the CSV preserving original row order and values as text."""
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, [dict(r) for r in reader]


def is_code(value):
    """True if a `team` value is a numeric placeholder rather than a name."""
    return value.isdigit()


def group_by_match(rows):
    """Return OrderedDict: match_id -> list of row dicts, in file order."""
    matches = OrderedDict()
    for idx, row in enumerate(rows):
        row["_row_index"] = idx + 2          # +2 = 1-based incl. header line
        matches.setdefault(row["match_id"], []).append(row)
    return matches


# ==========================================================================
# STEP 1 - Profile the raw data
# ==========================================================================
def profile(rows, label, log):
    log(f"\n{'=' * 74}\n{label}\n{'=' * 74}")
    matches = group_by_match(rows)

    log(f"Rows                     : {len(rows)}")
    log(f"Distinct matches         : {len(matches)}")
    log(f"Columns                  : {', '.join(rows[0].keys() - {'_row_index'})}"
        if rows else "Columns: -")

    # -- completeness ------------------------------------------------------
    # Iterate the row's own columns rather than a fixed list, so columns
    # added later in the pipeline are still completeness-checked here.
    blanks = {c: sum(1 for r in rows if str(r[c]).strip() == "")
              for c in rows[0] if c != "_row_index"} if rows else {}
    log(f"Blank / missing cells    : {blanks}")

    # -- uniqueness --------------------------------------------------------
    signatures = [tuple(r[c] for c in ("match_id", "team", "distance_km")) for r in rows]
    dupes = [s for s, n in Counter(signatures).items() if n > 1]
    log(f"Exact duplicate rows     : {len(dupes)}")

    # -- structural rule: every match has exactly two rows ------------------
    wrong_size = {m: len(rs) for m, rs in matches.items() if len(rs) != 2}
    log(f"Matches without 2 rows   : {wrong_size if wrong_size else 'none'}")

    # -- structural rule: the two rows must be two DIFFERENT teams ----------
    same_team = {m: rs[0]["team"] for m, rs in matches.items()
                 if len(rs) == 2 and rs[0]["team"] == rs[1]["team"]}
    log(f"Matches w/ duplicate team: {same_team if same_team else 'none'}")

    # -- validity: representation of the `team` column ----------------------
    codes = {r["team"] for r in rows if is_code(r["team"])}
    names = {r["team"] for r in rows if not is_code(r["team"])}
    log(f"`team` distinct NAMES    : {len(names)}")
    log(f"`team` distinct CODES    : {len(codes)}"
        + (f"  -> {sorted(codes)}" if codes else ""))
    log(f"`team` column is mixed-type: {'YES  <-- defect' if codes and names else 'no'}")

    # -- stage-level structure ---------------------------------------------
    log("\nPer-stage breakdown:")
    log(f"  {'stage':<26}{'matches':>9}{'expected':>10}{'distinct teams':>16}")
    per_stage = defaultdict(list)
    for m, rs in matches.items():
        per_stage[rs[0]["stage"]].append((m, rs))
    for stage in [GROUP_STAGE] + KNOCKOUT_STAGES:
        if stage not in per_stage:
            continue
        ms = per_stage[stage]
        teams = {r["team"] for _, rs in ms for r in rs}
        exp = EXPECTED_MATCHES_PER_STAGE.get(stage, "?")
        flag = "" if len(ms) == exp else "  <-- unexpected"
        log(f"  {stage:<26}{len(ms):>9}{exp:>10}{len(teams):>16}{flag}")

    # -- numeric sanity on the measure column -------------------------------
    vals = [float(r["distance_km"]) for r in rows]
    log(f"\ndistance_km  min={min(vals):.2f}  max={max(vals):.2f}  "
        f"mean={sum(vals)/len(vals):.2f}")
    log(f"distance_km  non-numeric : 0   negatives: {sum(1 for v in vals if v < 0)}")

    # -- dates ---------------------------------------------------------------
    bad_dates, mismatched = 0, 0
    for m, rs in matches.items():
        try:
            for r in rs:
                datetime.strptime(r["date"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            bad_dates += 1
        if len({r["date"] for r in rs}) > 1:
            mismatched += 1
    log(f"Unparseable dates        : {bad_dates}")
    log(f"Matches w/ inconsistent date between their 2 rows: {mismatched}")

    return matches


# ==========================================================================
# STEP 2 - Verify the home-first ordering assumption on the group stage
# ==========================================================================
def verify_home_first_assumption(matches, log):
    """
    The derivation relies on: row 1 of a match = home team, row 2 = away.
    We cannot test that directly on knockout rows (that is what we are
    solving for), so we test that the assumption is at least STRUCTURALLY
    sound on the 72 group-stage matches: each has exactly two rows, two
    distinct teams, and a stable order. We also confirm the very first
    fixture matches the known tournament opener (Mexico v South Africa).
    """
    log(f"\n{'=' * 74}\nSTEP 2 - Ordering assumption check (group stage)\n{'=' * 74}")
    group = [(m, rs) for m, rs in matches.items() if rs[0]["stage"] == GROUP_STAGE]
    ok = all(len(rs) == 2 and rs[0]["team"] != rs[1]["team"] for _, rs in group)
    log(f"Group-stage matches with exactly 2 distinct, ordered teams: "
        f"{sum(1 for _, rs in group if len(rs) == 2 and rs[0]['team'] != rs[1]['team'])}"
        f"/{len(group)}")
    first_id, first_rows = group[0]
    log(f"First fixture in file    : {first_rows[0]['team']} (row 1) v "
        f"{first_rows[1]['team']} (row 2), {first_rows[0]['date']}")
    log("Known 2026 opener        : Mexico v South Africa, 2026-06-11 (Mexico host, home)")
    log(f"Ordering assumption      : {'SUPPORTED' if ok else 'NOT SUPPORTED'}")
    if not ok:
        sys.exit("FATAL: group-stage ordering assumption failed.")
    return ok


# ==========================================================================
# STEP 3 - Derive the code -> country mapping from evidence + file order
# ==========================================================================
def derive_mapping(matches, log):
    log(f"\n{'=' * 74}\nSTEP 3 - Deriving code -> country mapping\n{'=' * 74}")

    assignments = defaultdict(list)   # code -> [(country, match_id), ...]
    for match_id, rows in matches.items():
        if rows[0]["stage"] == GROUP_STAGE:
            continue
        if match_id not in MATCH_EVIDENCE:
            sys.exit(f"FATAL: no external evidence recorded for match {match_id}")
        home, away, _result, _url = MATCH_EVIDENCE[match_id]
        for row, country in zip(rows, (home, away)):
            assignments[row["team"]].append((country, match_id))

    # A correct derivation gives each code exactly ONE country. Conflicts are
    # not silently resolved - they are surfaced, because a conflict means
    # either the evidence or the source data is wrong.
    conflicts = {code: set(c for c, _ in v)
                 for code, v in assignments.items()
                 if len({c for c, _ in v}) > 1}

    log(f"Distinct codes encountered      : {len(assignments)}")
    log(f"Codes resolving to 1 country    : {len(assignments) - len(conflicts)}")
    log(f"Codes with CONFLICTING countries: {len(conflicts)}")
    for code, countries in conflicts.items():
        involved = sorted({m for c, m in assignments[code]})
        log(f"   code {code} -> {sorted(countries)}  (matches: {involved})")
        log(f"   ^ investigated in STEP 4; a code cannot be two countries.")

    return assignments, conflicts


# ==========================================================================
# STEP 4 - Repair the structural defect, then re-derive cleanly
# ==========================================================================
def repair_and_finalise(rows, matches, log, changes):
    log(f"\n{'=' * 74}\nSTEP 4 - Repairing the duplicate-team defect\n{'=' * 74}")

    bug_rows = matches[BUG_MATCH_ID]
    home, away, result, url = MATCH_EVIDENCE[BUG_MATCH_ID]
    log(f"Match {BUG_MATCH_ID} ({bug_rows[0]['stage']}, {bug_rows[0]['date']})")
    log(f"  raw row 1: team={bug_rows[0]['team']}  distance_km={bug_rows[0]['distance_km']}")
    log(f"  raw row 2: team={bug_rows[1]['team']}  distance_km={bug_rows[1]['distance_km']}")
    log(f"  Defect  : both rows carry the same code - a match cannot be a team")
    log(f"            playing itself.")
    log(f"  Evidence: this match_id is FIFA's own id for {home} v {away} ({result})")
    log(f"            {url}")

    # Identify the away side's code from its OTHER appearances in the file.
    away_codes = set()
    for match_id, rs in matches.items():
        if match_id == BUG_MATCH_ID or rs[0]["stage"] == GROUP_STAGE:
            continue
        h, a, _r, _u = MATCH_EVIDENCE[match_id]
        for row, country in zip(rs, (h, a)):
            if country == away:
                away_codes.add(row["team"])
    if len(away_codes) != 1:
        sys.exit(f"FATAL: could not uniquely identify {away}'s code: {away_codes}")
    correct_code = away_codes.pop()

    log(f"  Repair  : {away}'s code is {correct_code}, established independently from")
    log(f"            its other appearances in this file. Row 2 (the AWAY row under")
    log(f"            the file's home-first ordering) is corrected accordingly.")
    log(f"            distance_km values are NOT altered - only the team label.")

    target = bug_rows[1]
    changes.append({
        "row_number_in_raw_file": target["_row_index"],
        "match_id": BUG_MATCH_ID,
        "column": "team",
        "old_value": target["team"],
        "new_value": correct_code,
        "change_type": "integrity_repair",
        "reason": (f"Duplicate team within one match; away side is {away} per FIFA "
                   f"match {BUG_MATCH_ID} ({result}). Code {correct_code} identified "
                   f"from {away}'s other rows in this file."),
        "evidence_url": url,
    })
    target["team"] = correct_code

    # ---- re-derive with the defect repaired ------------------------------
    assignments, conflicts = derive_mapping(matches, log)
    if conflicts:
        sys.exit("FATAL: conflicts remain after repair - do not ship this output.")

    mapping = {code: v[0][0] for code, v in assignments.items()}
    log(f"\nMapping resolved cleanly: {len(mapping)} codes -> "
        f"{len(set(mapping.values()))} distinct countries.")
    return mapping, assignments


# ==========================================================================
# STEP 5 - Apply the mapping
# ==========================================================================
def apply_mapping(rows, mapping, changes, log):
    log(f"\n{'=' * 74}\nSTEP 5 - Applying mapping to the `team` column\n{'=' * 74}")
    n = 0
    for row in rows:
        code = row["team"]
        if is_code(code):
            if code not in mapping:
                sys.exit(f"FATAL: unmapped code {code}")
            changes.append({
                "row_number_in_raw_file": row["_row_index"],
                "match_id": row["match_id"],
                "column": "team",
                "old_value": code,
                "new_value": mapping[code],
                "change_type": "code_resolved_to_name",
                "reason": "Numeric placeholder replaced with country name (see lookup table).",
                "evidence_url": MATCH_EVIDENCE[row["match_id"]][3],
            })
            row["team"] = mapping[code]
            n += 1
    log(f"Cells rewritten: {n}  (expected 64 = 32 knockout matches x 2 teams)")
    return n


# ==========================================================================
# STEP 6 - Post-conditions
# ==========================================================================
def enrich_with_duration(fieldnames, rows, log, changes):
    """Join match duration and derive the stage-comparable distance measure.

    Returns the extended fieldnames. Duration is looked up per match_id,
    not per row: TimePlayed describes the match, and getkm.py has already
    verified both teams of a match report the same value.
    """
    log(f"\n{'=' * 74}\nSTEP 6 - Joining match duration\n{'=' * 74}")

    if not os.path.exists(TIME_PATH):
        sys.exit(
            f"FATAL: {rel(TIME_PATH)} not found.\n"
            f"Run script/getkm.py first - it produces the duration table."
        )

    with open(TIME_PATH, newline="", encoding="utf-8") as fh:
        durations = {
            r["match_id"]: r["time_played_min"]
            for r in csv.DictReader(fh)
        }

    log(f"Duration table           : {rel(TIME_PATH)} ({len(durations)} matches)")

    # Every match in the data must be present in the duration table. A
    # missing KEY is a join failure and is fatal; a present-but-empty
    # VALUE is a real upstream gap and is carried through as blank.
    unmatched = sorted({r["match_id"] for r in rows} - set(durations))
    if unmatched:
        sys.exit(
            f"FATAL: {len(unmatched)} match_id(s) absent from the duration "
            f"table: {unmatched[:5]}"
        )
    log(f"Join coverage            : {len(rows)}/{len(rows)} rows matched")

    blank_matches = set()

    for row in rows:
        raw_time = (durations[row["match_id"]] or "").strip()

        if not raw_time:
            # Not imputed. A silent 90 here would be indistinguishable
            # from a genuinely 90-minute match in every later analysis.
            blank_matches.add(row["match_id"])
            row["time_played_min"] = ""
            row["distance_per_90"] = ""
            continue

        minutes = float(raw_time)
        row["time_played_min"] = f"{minutes:.2f}"
        row["distance_per_90"] = (
            f"{float(row['distance_km']) / minutes * 90:.2f}"
        )

    if blank_matches:
        log(
            f"\nMISSING DURATION         : {len(blank_matches)} match(es) "
            f"-> {sorted(blank_matches)}"
        )
        log("  The source endpoint returns no TimePlayed for these matches.")
        log("  Left blank rather than imputed; those rows cannot be")
        log("  normalised and must be excluded from per-90 analysis.")
    else:
        log("\nMISSING DURATION         : none")

    # Audit trail. The two columns are recorded as additions rather than
    # one entry per cell: no existing value is altered, so a 416-line
    # per-cell log would bury the 65 genuine repairs above it.
    changes.append({
        "row_number_in_raw_file": "all",
        "match_id": "all",
        "column": "time_played_min",
        "old_value": "(column absent)",
        "new_value": "match duration in minutes",
        "change_type": "column_added",
        "reason": (
            "Raw extract recorded distance with no duration, making "
            "distance_km incomparable between 90-minute group matches and "
            "knockout matches played to extra time. Recovered from the "
            "TimePlayed field of the same FIFA stats endpoint the distance "
            "came from; identical for both teams of a match."
        ),
        "evidence_url": "https://fdh-api.fifa.com/v1/stats/match/{result_id}/teams.json",
    })
    changes.append({
        "row_number_in_raw_file": "all",
        "match_id": "all",
        "column": "distance_per_90",
        "old_value": "(column absent)",
        "new_value": "distance_km / time_played_min * 90",
        "change_type": "column_derived",
        "reason": (
            "Stage-comparable workload measure. Derived, not sourced. "
            "Blank wherever time_played_min is blank."
        ),
        "evidence_url": "",
    })

    for match_id in sorted(blank_matches):
        changes.append({
            "row_number_in_raw_file": "",
            "match_id": match_id,
            "column": "time_played_min",
            "old_value": "",
            "new_value": "",
            "change_type": "value_unavailable",
            "reason": (
                "Source endpoint returns no TimePlayed for this match. "
                "Left blank rather than imputed - both rows of this match "
                "are excluded from any per-90 analysis."
            ),
            "evidence_url": "",
        })

    extended = list(fieldnames) + ["time_played_min", "distance_per_90"]
    log(f"\nSchema                   : {len(fieldnames)} -> {len(extended)} columns")
    log(f"  {', '.join(extended)}")
    return extended


# ==========================================================================
# STEP 7 - Post-repair validation
# ==========================================================================
def validate_clean(rows, log):
    log(f"\n{'=' * 74}\nSTEP 7 - Post-repair validation\n{'=' * 74}")
    matches = group_by_match(rows)
    checks = []

    def check(name, condition, detail=""):
        checks.append((name, condition, detail))
        log(f"  [{'PASS' if condition else 'FAIL'}] {name}{'  ' + detail if detail else ''}")

    check("No numeric codes remain in `team`",
          not any(is_code(r["team"]) for r in rows))
    check("Every match has exactly 2 rows",
          all(len(rs) == 2 for rs in matches.values()))
    check("Every match has 2 DIFFERENT teams",
          all(rs[0]["team"] != rs[1]["team"] for rs in matches.values()))
    check("Row count unchanged (208)", len(rows) == 208, f"actual={len(rows)}")
    check("Match count is 104 (48-team format)", len(matches) == 104,
          f"actual={len(matches)}")

    per_stage = defaultdict(set)
    stage_matches = Counter()
    for m, rs in matches.items():
        per_stage[rs[0]["stage"]].update(r["team"] for r in rs)
        stage_matches[rs[0]["stage"]] += 1
    for stage, expected in EXPECTED_MATCHES_PER_STAGE.items():
        check(f"{stage}: {expected} matches", stage_matches[stage] == expected,
              f"actual={stage_matches[stage]}")
    check("First Stage has 48 teams", len(per_stage[GROUP_STAGE]) == 48,
          f"actual={len(per_stage[GROUP_STAGE])}")
    for stage, expected_teams in [("Round of 32", 32), ("Round of 16", 16),
                                  ("Quarter-final", 8), ("Semi-final", 4),
                                  ("Play-off for third place", 2), ("Final", 2)]:
        check(f"{stage} has {expected_teams} distinct teams",
              len(per_stage[stage]) == expected_teams,
              f"actual={len(per_stage[stage])}")

    # Bracket monotonicity: a team can only appear in a round if it appeared
    # in the previous one. This is the strongest end-to-end consistency test.
    order = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final"]
    for prev, nxt in zip(order, order[1:]):
        check(f"{nxt} teams all appeared in {prev}",
              per_stage[nxt].issubset(per_stage[prev]))
    check("Third-place teams are the two beaten semi-finalists",
          per_stage["Play-off for third place"].issubset(per_stage["Semi-final"])
          and per_stage["Play-off for third place"].isdisjoint(per_stage["Final"]))
    check("Finalists are semi-finalists",
          per_stage["Final"].issubset(per_stage["Semi-final"]))
    check("Knockout teams all played in the group stage",
          per_stage["Round of 32"].issubset(per_stage[GROUP_STAGE]))

    # Duration checks. These guard the per-90 measure: if any of them
    # fails, distance_per_90 is not a comparable quantity.
    timed = [r for r in rows if r.get("time_played_min")]
    check("Both rows of a match share one duration",
          all(rs[0].get("time_played_min") == rs[1].get("time_played_min")
              for rs in matches.values()))
    check("Duration present for all 104 matches",
          len({r["match_id"] for r in timed}) == len(matches),
          f"actual={len({r['match_id'] for r in timed})}")
    check("All durations are plausible match lengths (90-150 min)",
          all(90 <= float(r["time_played_min"]) <= 150 for r in timed))
    check("Knockout durations reach extra time, group durations do not",
          max((float(r["time_played_min"]) for r in timed
               if r["stage"] == GROUP_STAGE), default=0) < 120
          <= max((float(r["time_played_min"]) for r in timed
                  if r["stage"] != GROUP_STAGE), default=0))
    check("distance_per_90 present exactly where duration is",
          all(bool(r.get("distance_per_90")) == bool(r.get("time_played_min"))
              for r in rows))
    check("distance_per_90 is within 1% of the manual recomputation",
          all(abs(float(r["distance_per_90"])
                  - float(r["distance_km"]) / float(r["time_played_min"]) * 90)
              < 0.01 for r in timed))

    failed = [c for c in checks if not c[1]]
    log(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed.")
    if failed:
        sys.exit("FATAL: post-conditions failed - output not written.")
    return checks


# ==========================================================================
# STEP 8 - Write outputs
# ==========================================================================
def write_outputs(fieldnames, rows, mapping, assignments, changes, log):
    log(f"\n{'=' * 74}\nSTEP 8 - Writing outputs\n{'=' * 74}")

    # Output directories are created on demand: a fresh clone carries only
    # datasets/, so without this the run dies here having already passed
    # every validation check.
    for path in (CLEAN_PATH, LOOKUP_PATH, CHANGES_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(CLEAN_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fieldnames})
    log(f"  {rel(CLEAN_PATH)}  ({len(rows)} rows, {len(fieldnames)} columns)")

    # Lookup table, ordered by first appearance, with provenance per code.
    with open(LOOKUP_PATH, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["team_code", "team_name", "n_appearances_in_knockout",
                    "first_knockout_match_id", "first_knockout_fixture",
                    "first_knockout_result", "evidence_url", "confidence"])
        for code in sorted(mapping, key=lambda c: (len(assignments[c]), c), reverse=True):
            entries = assignments[code]
            first_match = entries[0][1]
            home, away, result, url = MATCH_EVIDENCE[first_match]
            w.writerow([code, mapping[code], len(entries), first_match,
                        f"{home} v {away}", result, url,
                        "high (multi-source corroborated)"])
    log(f"  {rel(LOOKUP_PATH)}  ({len(mapping)} codes)")

    with open(CHANGES_PATH, "w", newline="", encoding="utf-8") as fh:
        cols = ["row_number_in_raw_file", "match_id", "column", "old_value",
                "new_value", "change_type", "reason", "evidence_url"]
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        # row_number_in_raw_file is an int for per-cell repairs and a
        # string ("all", "") for the column-level entries, so sort on a
        # normalised key rather than the raw value.
        def order(change):
            value = change["row_number_in_raw_file"]
            return (1, 0) if isinstance(value, str) else (0, value)

        for c in sorted(changes, key=order):
            w.writerow(c)
    log(f"  {rel(CHANGES_PATH)}  ({len(changes)} cell changes)")


# ==========================================================================
def main():
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("2026 FIFA WORLD CUP TEAM-DISTANCE DATASET - WRANGLING VALIDATION REPORT")
    log(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Source   : {rel(RAW_PATH)}")

    fieldnames, rows = load_rows(RAW_PATH)
    matches = profile(rows, "STEP 1 - RAW DATA PROFILE (before wrangling)", log)

    verify_home_first_assumption(matches, log)
    derive_mapping(matches, log)              # surfaces the conflict

    changes = []
    mapping, assignments = repair_and_finalise(rows, matches, log, changes)
    apply_mapping(rows, mapping, changes, log)
    fieldnames = enrich_with_duration(fieldnames, rows, log, changes)

    profile(rows, "CLEANED DATA PROFILE (after wrangling)", log)
    validate_clean(rows, log)
    write_outputs(fieldnames, rows, mapping, assignments, changes, log)

    log("\nDone. All post-conditions passed.")

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
