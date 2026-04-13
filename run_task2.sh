#!/bin/bash
# ============================================================================
# Task 2 (TSP) runner — edit this file to invoke YOUR solution.
#
# The autograder calls this script with these positional arguments:
#   $1 = start stop name     (e.g. "Wrocław Główny")
#   $2 = stops to visit      (semicolon-separated, e.g. "Jelenia Góra;Legnica;Brzeg")
#   $3 = criterion           ("t" for time, "p" for transfers)
#   $4 = start time          (e.g. "06:00:00")
#
# The date is available as the environment variable GTFS_DATE (e.g. "2026-03-04").
#
# Your solution MUST print route segments to stdout and criterion value to stderr.
#
# Examples (uncomment and adapt ONE line):
#   python3 solution.py "$1" "$2" "$3" "$4"
#   python3 solution.py --tsp "$1" "$2" "$3" "$4" "$GTFS_DATE"
# ============================================================================

python3 solution.py "$1" "$2" "$3" "$4" --task2
