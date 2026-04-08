[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/1mf_jpi1)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=23475135&assignment_repo_type=AssignmentRepo)
# Assignment #1 — Graph Traversals

**Course:** Artificial Intelligence and Knowledge Engineering (AI&KE), 2025/2026, SI4024L

**Deadline:** 24.03.2026, 17:05 (cutoff — repository becomes read-only)

---

## Overview

In this assignment you implement pathfinding algorithms on a real railway timetable (Koleje Dolnośląskie, GTFS format):

- **Task 1** — Find the shortest path from stop A to stop B using Dijkstra and A\*, optimizing either travel time (`t`) or number of transfers (`p`).
- **Task 2** — Solve a TSP variant using Tabu Search: visit all stops in a list L and return to the start.

See the full assignment PDF for details, scoring, and theoretical background.

---

## Repository structure

```
.
├── google_transit/          # GTFS dataset (DO NOT MODIFY)
│   ├── stops.txt
│   ├── stop_times.txt
│   ├── calendar.txt
│   ├── calendar_dates.txt
│   ├── trips.txt
│   ├── routes.txt
│   └── ...
├── run_task1.sh             # Runner script for Task 1 (edit this)
├── run_task2.sh             # Runner script for Task 2 (edit this)
├── tests/
│   └── autograder.py        # Automated tests (DO NOT MODIFY)
├── .github/                 # CI/autograding config (DO NOT MODIFY)
└── README.md
```

---

## How to submit your solution

### 1. Clone this repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Add your solution

Place your solution files (e.g. `solution.py`, `main.py`, or compiled binaries) in the root of this repository. You can use any programming language.

### 3. Configure the runner scripts

Edit **`run_task1.sh`** and **`run_task2.sh`** — change the last line to invoke your solution.

The autograder calls these scripts with positional arguments:

**`run_task1.sh`** receives:
| Argument | Description | Example |
|----------|-------------|---------|
| `$1` | Starting stop | `"Wrocław Główny"` |
| `$2` | Ending stop | `"Jelenia Góra"` |
| `$3` | Criterion (`t` or `p`) | `t` |
| `$4` | Start time | `06:00:00` |

**`run_task2.sh`** receives:
| Argument | Description | Example |
|----------|-------------|---------|
| `$1` | Starting stop | `"Wrocław Główny"` |
| `$2` | Stops to visit (`;`-separated) | `"Jelenia Góra;Legnica;Brzeg"` |
| `$3` | Criterion (`t` or `p`) | `t` |
| `$4` | Start time | `06:00:00` |

The date is available as the **`$GTFS_DATE`** environment variable (e.g. `2026-03-04`).

Examples — pick one and adapt:

```bash
# Python (default, already set):
python3 solution.py "$1" "$2" "$3" "$4"

# Python with date:
python3 solution.py "$1" "$2" "$3" "$4" "$GTFS_DATE"

# Python with named args:
python3 main.py --start "$1" --end "$2" --criterion "$3" --time "$4" --date "$GTFS_DATE"

# Java:
java -jar solution.jar "$1" "$2" "$3" "$4"

# C++:
./solution "$1" "$2" "$3" "$4"
```

### 4. Output format

As specified in the assignment PDF:

- **stdout** — successive lines, each describing a route segment: starting stop, ending stop, line name, departure time, arrival time.
- **stderr** — the value of the minimized criterion and computation time.

### 5. Push your solution

```bash
git add .
git commit -m "Add my solution"
git push
```

Every push triggers the **autograder** via GitHub Actions. Check results in the **Actions** tab of your repository.

---

## Autograder tests

The autograder runs 11 correctness tests (44 points). These tests only verify that your solution produces correct outputs — they do **not** determine your final grade.

> **The final grade is determined by the teacher** based on the autograder results, the quality of your report, and your understanding of the theoretical background (algorithm design, heuristic admissibility, experimental comparisons). Passing all autograder tests is necessary but not sufficient for a full score. See the grading rubric for the complete breakdown.

| Test | What it checks | Points |
|------|---------------|--------|
| S1_1 | Direct connection — graph construction (D6, arr 08:26) | 4 |
| S1_2 | Multi-platform stations / `parent_station` (D1, arr 08:41) | 4 |
| S1_3 | Dwell time at stops (D66, arr 14:31, 61 min not 50) | 4 |
| S1_4 | Service filtering by date / `calendar.txt` (Wed ≠ Sat) | 4 |
| S1_5 | Transfer criterion — direct route, 0 transfers | 4 |
| S1_5_MULTI | Transfer criterion — multi-transfer route (Szklarska→Brzeg) | 4 |
| S2_1 | TSP: 2 stops on one line (~69 min round trip) | 4 |
| S2_2 | TSP: 3 stops in different directions | 4 |
| S2_3 | TSP: cost asymmetry / time-dependent costs | 4 |
| S2_4 | TSP: Tabu list mechanism (3 stops, ~345 min) | 4 |
| S2_5 | TSP: transfer criterion (≥1 transfer) | 4 |

### Running tests locally

```bash
# Run a single test:
GTFS_DATE=2026-03-04 python3 tests/autograder.py S1_1

# Run all tests:
for t in S1_1 S1_2 S1_3 S1_4 S1_5 S1_5_MULTI S2_1 S2_2 S2_3 S2_4 S2_5; do
    echo "--- $t ---"
    GTFS_DATE=2026-03-04 python3 tests/autograder.py $t
done
```

---

## Important notes

- **Do not modify** files in `tests/`, `.github/`, or `google_transit/`. Changes to these protected files will be flagged.
- The autograder parses your **stdout**, so make sure each route segment is printed on a separate line with visible departure and arrival times (e.g. `06:10`, `08:26`).
- If your solution filters services by date, read `$GTFS_DATE` (set by the autograder). In Python: `os.environ.get('GTFS_DATE', '2026-03-04')`.
- The GTFS dataset (Koleje Dolnośląskie) is valid from **2026-03-03** to **2026-12-12**. The default test date is **2026-03-04 (Wednesday)**.

---

## Quick Git reference

| What you want to do | Command |
|---------------------|---------|
| Clone your repo | `git clone <url>` |
| Check what changed | `git status` |
| Stage files | `git add solution.py run_task1.sh` |
| Commit | `git commit -m "description"` |
| Push to GitHub | `git push` |
| Pull latest changes | `git pull` |

New to Git? See the [Git Handbook](https://docs.github.com/en/get-started/using-git/about-git) or [this short video](https://www.youtube.com/watch?v=w3jLJU7DT5E).
