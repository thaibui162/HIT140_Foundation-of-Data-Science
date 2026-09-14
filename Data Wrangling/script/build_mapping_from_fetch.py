#!/usr/bin/env python3
"""
=============================================================================
 STAGE 2 - build_mapping_from_fetch.py
 Resolve every numeric team code in the raw dataset to a country name, then
 cross-check the result against the original pipeline's derivation.
=============================================================================

INPUTS
------
  --roster    team_id -> team_name table  (from build_reference_csv.py)
  --fetched   fixture table               (from build_reference_csv.py or
                                           fetch_fifa_matches.py)
  --raw       the original dataset

WHAT IT DOES
------------
  1. Repairs the one known integrity defect (see INTEGRITY REPAIR below).

  2. Resolves each numeric code by the strongest route available:

       ROUTE 1 - ROSTER (preferred)
         The codes in the dataset are FIFA team ids. If the roster names
         every code, each one is resolved by plain dictionary lookup.
         No inference of any kind - not from row order, not from fixtures.

       ROUTE 2 - DIRECT (per-match team ids)
         The fetched fixture rows carry home_team_id / away_team_id for the
         knockout matches. Also a plain lookup, but match by match.

       ROUTE 3 - POSITIONAL (fallback)
         No usable ids anywhere. Fall back to the dataset's home-first row
         ordering: row 1 of a match is the home team, row 2 the away team.
         This is the method `wrangle_wc2026.py` uses.

     The script picks the route automatically and reports which it used.

  3. Requires every code to resolve to exactly ONE country across all of its
     appearances. Conflicts abort the run - they mean the inputs disagree.

  4. Cross-checks against `wrangle_wc2026.py`'s mapping and reports agreement
     code by code.

INTEGRITY REPAIR
----------------
One Round-of-16 match lists the same team code on both of its rows, which is
impossible. The true value is recovered from the bracket itself, with no
external evidence: a team that appears in the quarter-finals must have played
in the Round of 16 to get there, so a quarter-finalist missing from the
Round of 16 can only be the corrupted side.

This works because the corrupted cell in this dataset happens to be the side
that WON that match, so it reappears in the next round. It is not a general
repair: had the losing side been corrupted, it would appear nowhere later and
the bracket could not identify it. The code therefore requires the bracket to
yield exactly one candidate and refuses to guess otherwise - see
repair_duplicate_rows(). A refusal leaves the duplicate in place, which the
post-conditions then catch.

INDEPENDENCE OF THE CROSS-CHECK
-------------------------------
The cross-check is only meaningful if the two mappings were derived by
genuinely different means. Under ROUTE 1 they are: the roster comes from
FIFA's published team ids, while `wrangle_wc2026.py` works from row ordering
plus hand-collected fixture evidence.

Under ROUTE 3 they are NOT independent - both would rest on the same
home-first assumption, so agreement would confirm very little. The verdict
line says so explicitly rather than claiming more than the run has shown.

OUTPUTS
-------
  reference/team_code_lookup_FROM_FETCH.csv
  reports/mapping_crosscheck.txt

USAGE
-----
  python3 build_reference_csv.py          # build the inputs first
  python3 build_mapping_from_fetch.py

Standard library only. Python 3.7+.
=============================================================================
"""

import argparse
import csv
import os
import sys
from collections import OrderedDict, defaultdict

# --- paths ----------------------------------------------------------------
# Resolved from this file's location, not the working directory.
DATA_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_RAW = os.path.join(DATA_ROOT, "datasets", "wc2026_team_distance.csv")
DEFAULT_ROSTER = os.path.join(DATA_ROOT, "reference", "team_roster.csv")
DEFAULT_FETCHED = os.path.join(DATA_ROOT, "reference", "fifa_matches_REFERENCE.csv")

OUT_LOOKUP = os.path.join(DATA_ROOT, "reference", "team_code_lookup_FROM_FETCH.csv")
OUT_REPORT = os.path.join(DATA_ROOT, "reports", "mapping_crosscheck.txt")

# --- constants ------------------------------------------------------------
# Inlined (not imported) so this stage runs standalone: it must work on a
# machine that has only the reference files and the raw dataset.
GROUP_STAGE = "First Stage"

# Rounds in bracket order. Used by the integrity repair: every team in a
# round must also appear in the round before it.
KNOCKOUT_ORDER = ["Round of 32", "Round of 16", "Quarter-final", "Semi-final"]

# The cross-check against the original derivation is optional. If the main
# pipeline is not alongside this file, the mapping is still built - only the
# comparison section goes empty.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from wrangle_wc2026 import MATCH_EVIDENCE  # noqa: E402
except ImportError:
    MATCH_EVIDENCE = {}


def rel(path):
    return os.path.relpath(path, os.path.dirname(DATA_ROOT))


def is_code(value):
    return str(value).strip().isdigit()


def load_raw(path):
    if not os.path.exists(path):
        sys.exit(f"FATAL: raw dataset not found: {path}")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    matches = OrderedDict()
    for index, row in enumerate(rows):
        row["_line"] = index + 2
        matches.setdefault(row["match_id"], []).append(row)
    return rows, matches


def load_roster(path, log):
    """team_id -> team_name. Absent file is not fatal; it costs Route 1."""
    if not os.path.exists(path):
        log(f"  roster                      : not found ({rel(path)})")
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"team_id", "team_name"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        sys.exit(f"FATAL: roster is missing column(s): {sorted(missing)}")
    roster = {str(r["team_id"]).strip(): str(r["team_name"]).strip()
              for r in rows if str(r["team_id"]).strip()}
    log(f"  roster entries loaded       : {len(roster)}")
    return roster


def load_fetched(path, log):
    if not os.path.exists(path):
        log(f"  fetched fixtures            : not found ({rel(path)})")
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    required = {"match_id", "home_team", "away_team"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        sys.exit(f"FATAL: fetched CSV is missing column(s): {sorted(missing)}")
    fetched = {str(r["match_id"]).strip(): r for r in rows}
    if len(fetched) != len(rows):
        log(f"  ! fetched CSV has duplicate match_ids "
            f"({len(rows)} rows -> {len(fetched)} unique)")
    log(f"  fetched fixtures loaded     : {len(fetched)}")
    return fetched


def knockout_codes(raw_matches):
    codes = set()
    for pair in raw_matches.values():
        if pair[0]["stage"] == GROUP_STAGE:
            continue
        for row in pair:
            if is_code(row["team"]):
                codes.add(str(row["team"]).strip())
    return codes


# ==========================================================================
# INTEGRITY REPAIR
# ==========================================================================

def repair_duplicate_rows(raw_matches, log):
    """
    Fix any match whose two rows carry the same team code.

    The replacement is derived from bracket continuity alone: a team that
    played in round N+1 must have played in round N, so a team appearing in
    the next round but not this one can only be the missing side.

    This identifies the corrupted cell only when that side went on to win.
    If the bracket does not name exactly one candidate, no repair is made and
    the duplicate is left for the post-conditions to reject.
    """
    broken = [(mid, pair) for mid, pair in raw_matches.items()
              if pair[0]["stage"] != GROUP_STAGE and len(pair) == 2
              and str(pair[0]["team"]).strip() == str(pair[1]["team"]).strip()]

    if not broken:
        log("  integrity repair            : nothing to repair")
        return []

    by_stage = defaultdict(set)
    for pair in raw_matches.values():
        for row in pair:
            by_stage[row["stage"]].add(str(row["team"]).strip())

    repairs = []
    for match_id, pair in broken:
        stage = pair[0]["stage"]
        if stage not in KNOCKOUT_ORDER:
            log(f"  ! {match_id} is duplicated in {stage}; no later round to "
                f"derive from.")
            continue

        later = KNOCKOUT_ORDER[KNOCKOUT_ORDER.index(stage) + 1] \
            if KNOCKOUT_ORDER.index(stage) + 1 < len(KNOCKOUT_ORDER) else None
        if later is None:
            log(f"  ! {match_id}: no following round available.")
            continue

        # Teams that reached the next round but are absent from this one.
        orphans = by_stage[later] - by_stage[stage]
        if len(orphans) != 1:
            log(f"  ! {match_id}: bracket gives {len(orphans)} candidate(s) "
                f"{sorted(orphans)}; cannot repair unambiguously.")
            continue

        true_code = orphans.pop()
        old = str(pair[1]["team"]).strip()
        pair[1]["team"] = true_code
        by_stage[stage].add(true_code)
        repairs.append((match_id, old, true_code))
        log(f"  integrity repair            : match {match_id} row 2 "
            f"{old} -> {true_code}")
        log(f"     basis: {true_code} appears in {later} but was absent from "
            f"{stage}, which the bracket forbids.")

    return repairs


# ==========================================================================
# ROUTE SELECTION
# ==========================================================================

def choose_route(raw_matches, fetched, roster, log):
    """Pick the strongest resolution route the inputs actually support."""
    codes = knockout_codes(raw_matches)
    log(f"  dataset knockout codes      : {len(codes)}")

    covered = codes & set(roster)
    log(f"  codes named by the roster   : {len(covered)}")
    if codes and covered >= codes:
        log("  ROUTE 1 (ROSTER lookup) - every code has a published name.")
        return "roster"

    fetched_ids = set()
    for row in fetched.values():
        for key in ("home_team_id", "away_team_id"):
            value = str(row.get(key, "") or "").strip()
            if value:
                fetched_ids.add(value)
    overlap = codes & fetched_ids
    log(f"  codes with per-match ids    : {len(overlap)}")

    if codes and overlap >= codes:
        log("  ROUTE 2 (DIRECT team-ID join) - fetched ids cover every code.")
        return "direct"

    if covered or overlap:
        log(f"  ! partial coverage (roster {len(covered)}, ids {len(overlap)}, "
            f"of {len(codes)}). Not safe to mix routes.")
    log("  ROUTE 3 (POSITIONAL home/away join) selected.")
    return "positional"


# ==========================================================================
# MAPPING
# ==========================================================================

def build_mapping(raw_matches, fetched, roster, route, log):
    """Return per-code assignment evidence and any conflicts."""
    assignments = defaultdict(list)
    unmatched = []
    unresolved = set()

    for match_id, pair in raw_matches.items():
        if pair[0]["stage"] == GROUP_STAGE:
            continue

        if route == "roster":
            for row in pair:
                code = str(row["team"]).strip()
                name = roster.get(code)
                if name is None:
                    unresolved.add(code)
                    continue
                assignments[code].append(
                    (name, match_id, "roster lookup on FIFA team id"))
            continue

        fetched_row = fetched.get(match_id)
        if fetched_row is None:
            unmatched.append(match_id)
            continue

        home = fetched_row["home_team"].strip()
        away = fetched_row["away_team"].strip()
        if not home or not away:
            unmatched.append(match_id)
            continue

        if route == "direct":
            id_to_name = {
                str(fetched_row.get("home_team_id", "")).strip(): home,
                str(fetched_row.get("away_team_id", "")).strip(): away,
            }
            for row in pair:
                code = str(row["team"]).strip()
                if code in id_to_name:
                    assignments[code].append(
                        (id_to_name[code], match_id, "direct team-ID lookup"))
                else:
                    # Route selection is meant to rule this out. If it ever
                    # fires, the inputs disagree and silence would hide it.
                    unresolved.add(code)
        else:
            for index, (row, country) in enumerate(zip(pair, (home, away))):
                code = str(row["team"]).strip()
                position = "row 1 = HOME" if index == 0 else "row 2 = AWAY"
                assignments[code].append((country, match_id, position))

    if unmatched:
        log(f"  ! {len(unmatched)} knockout match(es) had no usable fetched "
            f"row: {unmatched[:8]}")
    if unresolved:
        log(f"  ! {len(unresolved)} code(s) could not be named: "
            f"{sorted(unresolved)}")

    conflicts = {code: sorted({c for c, _, _ in entries})
                 for code, entries in assignments.items()
                 if len({c for c, _, _ in entries}) > 1}

    return assignments, conflicts, unmatched, unresolved


def main():
    # Rebound from --out-lookup / --out-report below, so that callers (the
    # route tests) can redirect the outputs instead of overwriting the real
    # mapping. Declared here because argparse reads the defaults from them.
    global OUT_LOOKUP, OUT_REPORT

    parser = argparse.ArgumentParser(
        description="Resolve team codes by lookup, then cross-check.")
    parser.add_argument("--raw", default=DEFAULT_RAW)
    parser.add_argument("--roster", default=DEFAULT_ROSTER)
    parser.add_argument("--fetched", default=DEFAULT_FETCHED)
    parser.add_argument("--allow-bug", action="store_true",
                        help="do not auto-repair the duplicate-team row")
    parser.add_argument("--force-route", choices=["roster", "direct", "positional"],
                        help="override automatic route selection")
    parser.add_argument("--out-lookup", default=OUT_LOOKUP,
                        help="where to write the code -> country mapping")
    parser.add_argument("--out-report", default=OUT_REPORT,
                        help="where to write the run log")
    args = parser.parse_args()

    OUT_LOOKUP, OUT_REPORT = args.out_lookup, args.out_report

    lines = []

    def log(message=""):
        print(message)
        lines.append(message)

    log("=" * 74)
    log(" STAGE 2 - resolving team codes")
    log("=" * 74)
    log(f" raw     : {rel(args.raw)}")
    log(f" roster  : {rel(args.roster)}")
    log(f" fetched : {rel(args.fetched)}")
    log("")

    _raw_rows, raw_matches = load_raw(args.raw)
    roster = load_roster(args.roster, log)
    fetched = load_fetched(args.fetched, log)
    log("")

    log("  --- integrity repair ---")
    if args.allow_bug:
        log("  skipped (--allow-bug); expect a conflict if a row is duplicated.")
    else:
        repair_duplicate_rows(raw_matches, log)
    log("")

    log("  --- route selection ---")
    route = choose_route(raw_matches, fetched, roster, log)
    if args.force_route and args.force_route != route:
        log(f"  ! --force-route overrides selection: {route} -> {args.force_route}")
        route = args.force_route
    log("")

    assignments, conflicts, _unmatched, _unresolved = build_mapping(
        raw_matches, fetched, roster, route, log)

    log("  --- resolution ---")
    log(f"  codes encountered           : {len(assignments)}")
    log(f"  codes resolving to 1 country: {len(assignments) - len(conflicts)}")
    log(f"  conflicting codes           : {len(conflicts)}")
    for code, countries in conflicts.items():
        log(f"     {code} -> {countries}")
    if conflicts:
        log("\nFATAL: inputs disagree. Nothing written.")
        _write_report(lines)
        sys.exit(1)

    mapping = {code: entries[0][0] for code, entries in assignments.items()}
    log(f"  distinct countries          : {len(set(mapping.values()))}")
    log("")

    # ---- post-conditions -------------------------------------------------
    # The conflict check above cannot catch everything. A code duplicated
    # within one match resolves consistently (it is the same code twice), so
    # it passes silently; and a route that resolves nothing produces an empty
    # mapping rather than an error. Both are caught here instead.
    log("  --- post-conditions ---")
    failures = []

    expected = knockout_codes(raw_matches)
    unnamed = expected - set(mapping)
    log(f"  codes required / resolved   : {len(expected)} / "
        f"{len(expected) - len(unnamed)}")
    if unnamed:
        failures.append(f"{len(unnamed)} code(s) never resolved: "
                        f"{sorted(unnamed)[:8]}")

    bad_matches = []
    for match_id, pair in raw_matches.items():
        if pair[0]["stage"] == GROUP_STAGE:
            continue
        names = [mapping.get(str(r["team"]).strip()) for r in pair]
        if len(names) == 2 and names[0] is not None and names[0] == names[1]:
            bad_matches.append((match_id, names[0]))
    log(f"  knockout matches with 2 distinct teams: "
        f"{sum(1 for m, p in raw_matches.items() if p[0]['stage'] != GROUP_STAGE) - len(bad_matches)}"
        f"/{sum(1 for m, p in raw_matches.items() if p[0]['stage'] != GROUP_STAGE)}")
    for match_id, name in bad_matches:
        failures.append(f"match {match_id} resolves to {name!r} on both rows")

    if failures:
        for message in failures:
            log(f"     FAIL {message}")
        log("\nFATAL: post-conditions failed. Nothing written.")
        _write_report(lines)
        sys.exit(1)
    log("  all post-conditions passed.")
    log("")

    # ---- cross-check against the original pipeline -----------------------
    log("  --- cross-check vs wrangle_wc2026.py ---")
    original = {}
    for match_id, pair in raw_matches.items():
        if pair[0]["stage"] == GROUP_STAGE or match_id not in MATCH_EVIDENCE:
            continue
        home, away, _r, _u = MATCH_EVIDENCE[match_id]
        for row, country in zip(pair, (home, away)):
            original[str(row["team"]).strip()] = country

    agree = [c for c in mapping if original.get(c) == mapping[c]]
    differ = [c for c in mapping if c in original and original[c] != mapping[c]]
    only_new = [c for c in mapping if c not in original]

    log(f"  codes in both mappings      : {len(set(mapping) & set(original))}")
    log(f"  AGREE                       : {len(agree)}")
    log(f"  DISAGREE                    : {len(differ)}")
    for code in differ:
        log(f"     {code}: this run says {mapping[code]!r}, "
            f"wrangle_wc2026.py says {original[code]!r}")
    if only_new:
        log(f"  present only in this route  : {only_new}")

    if not original:
        verdict = ("NOT RUN - wrangle_wc2026.py is not available here, so the "
                   "mapping above stands on the lookup alone")
    elif differ:
        verdict = "DIFFERENCES FOUND - investigate before using"
    elif route == "positional":
        verdict = ("AGREES, but NOT independently - this run and "
                   "wrangle_wc2026.py both assume home-first row order")
    else:
        verdict = ("IDENTICAL - a lookup against FIFA team ids reproduces the "
                   "mapping wrangle_wc2026.py derived from row order and "
                   "fixture evidence")
    log(f"  VERDICT                     : {verdict}")
    log("")

    # ---- write outputs ---------------------------------------------------
    os.makedirs(os.path.dirname(OUT_LOOKUP), exist_ok=True)
    with open(OUT_LOOKUP, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["team_code", "team_name", "resolution_route",
                         "n_assignments", "matches_used",
                         "agrees_with_wrangle_pipeline"])
        for code in sorted(mapping, key=lambda c: (-len(assignments[c]), c)):
            entries = assignments[code]
            if code not in original:
                verdict_cell = "n/a"          # the other method never saw it
            elif original[code] == mapping[code]:
                verdict_cell = "yes"
            else:
                verdict_cell = "no"           # a real disagreement, not unknown
            writer.writerow([
                code, mapping[code], route, len(entries),
                ";".join(m for _, m, _ in entries),
                verdict_cell,
            ])
    log(f"  wrote {rel(OUT_LOOKUP)}")
    _write_report(lines)
    log(f"  wrote {rel(OUT_REPORT)}")


def _write_report(lines):
    os.makedirs(os.path.dirname(OUT_REPORT), exist_ok=True)
    with open(OUT_REPORT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
