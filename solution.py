"""
main.py — Entry point for Task 1 and Task 2.

Usage
-----
Task 1 (shortest path A → B):
    python main.py <start> <end> <criterion> <time> [--algo ALGO] [--date DATE]

    ALGO choices:
        dijkstra        Dijkstra's algorithm           (Task 1a)
        astar           A* search                      (Task 1b / 1c)
        bidirectional   Bidirectional A* (travel time) (Task 1d)

Task 2 (TSP tour A → L → A):
    python main.py <start> <stops> <criterion> <time> --task2 [--tabu-size N]
                   [--variable-t] [--aspiration] [--sample FRAC] [--max-iter N]
                   [--date DATE]

Arguments
---------
    start       Starting stop name  (e.g. "Wrocław Główny")
    end/stops   Ending stop (Task 1) or semicolon-separated list (Task 2)
    criterion   't' = minimise travel time  |  'p' = minimise transfers
    time        Departure time  HH:MM  or  HH:MM:SS

Options
-------
    --algo          Algorithm for Task 1  (default: astar)
    --task2         Switch to Task 2 mode
    --tabu-size N   Fixed Tabu list size  (default: 0 = unbounded)
    --variable-t    Use variable T size based on |L|  (Task 2b)
    --aspiration    Enable aspiration criterion  (Task 2c)
    --sample FRAC   Neighbourhood sample fraction 0..1  (Task 2d, default: 1.0)
    --max-iter N    Max Tabu Search iterations  (default: 200)
    --date DATE     Query date  YYYY-MM-DD  (default: 2026-04-07, a Tuesday)

Examples
--------
    python main.py "Wrocław Główny" "Jelenia Góra" t 08:00
    python main.py "Wrocław Główny" "Jelenia Góra" t 08:00 --algo dijkstra
    python main.py "Wrocław Główny" "Jelenia Góra" p 08:00 --algo astar
    python main.py "Wrocław Główny" "Jelenia Góra" t 08:00 --algo bidirectional
    python main.py "Wrocław Główny" "Jelenia Góra;Legnica" t 08:00 --task2
    python main.py "Wrocław Główny" "Jelenia Góra;Legnica" t 08:00 --task2 --variable-t --aspiration
    python main.py "Wrocław Główny" "Jelenia Góra;Legnica" t 08:00 --task2 --sample 0.5 --max-iter 100
"""

import argparse
import os
import sys
from datetime import date

from gtfs_loader import GTFSLoader, parse_gtfs_time
from graph import TransitGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GTFS_DIR = "google_transit"
DEFAULT_DATE = date(2026, 4, 7)   # Tuesday — most services run


def parse_time_arg(time_str: str) -> int:
    """Accept HH:MM or HH:MM:SS and return seconds since midnight."""
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        h, m = parts
        return int(h) * 3600 + int(m) * 60
    elif len(parts) == 3:
        return parse_gtfs_time(time_str)
    else:
        print(f"ERROR: Invalid time format '{time_str}'. Use HH:MM or HH:MM:SS.", file=sys.stderr)
        sys.exit(1)


def parse_date_arg(date_str: str) -> date:
    from datetime import datetime
    return datetime.strptime(date_str.strip(), "%Y-%m-%d").date()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Transit pathfinding — Koleje Dolnośląskie GTFS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("start",     help="Starting stop name")
    p.add_argument("end_stops", metavar="end/stops",
                   help="Ending stop (Task 1) or semicolon-separated list (Task 2)")
    p.add_argument("criterion", choices=["t", "p"],
                   help="'t' = travel time  |  'p' = transfers")
    p.add_argument("time",      help="Departure time  HH:MM  or  HH:MM:SS")

    # Task 1 options
    p.add_argument("--algo", choices=["dijkstra", "astar", "bidirectional"],
                   default="astar",
                   help="Algorithm for Task 1 (default: astar)")
    p.add_argument("--compare", action="store_true",
                   help="Run all Task-1 algorithms and print a comparison table")
    p.add_argument("--tabu-compare", action="store_true",
                   help="Run Task-2 with several T sizes and print comparison table")

    # Task 2 mode
    p.add_argument("--task2", action="store_true",
                   help="Enable Task 2 (TSP tour) mode")
    p.add_argument("--tabu-size", type=int, default=0, metavar="N",
                   help="Tabu list size; 0 = unbounded (default: 0)")
    p.add_argument("--variable-t", action="store_true",
                   help="Variable T size based on |L| (Task 2b)")
    p.add_argument("--aspiration", action="store_true",
                   help="Enable aspiration criterion (Task 2c)")
    p.add_argument("--sample", type=float, default=1.0, metavar="FRAC",
                   help="Neighbourhood sample fraction 0..1 (Task 2d, default: 1.0)")
    p.add_argument("--max-iter", type=int, default=200, metavar="N",
                   help="Max Tabu Search iterations (default: 200)")

    # Common options
    p.add_argument("--date", default=None, metavar="YYYY-MM-DD",
                   help=f"Query date (default: {DEFAULT_DATE})")

    return p


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def run_compare(graph, loader, start_name: str, end_name: str,
                criterion: str, start_time: int):
    """
    Run all Task-1 algorithms on the same query and print a comparison table
    to stderr.  Criterion 't' runs Dijkstra / A* / Bidirectional A*;
    criterion 'p' runs Dijkstra / A*.
    """
    import time as _time
    from dijkstra import dijkstra, dijkstra_transfers, format_path
    from astar import astar_time, astar_transfers, astar_bidirectional

    starts = loader.stop_name_to_ids(start_name)
    ends   = set(loader.stop_name_to_ids(end_name))
    if not starts or not ends:
        print("ERROR: Stop not found.", file=sys.stderr)
        return

    rows = []

    if criterion == "t":
        algos = [
            ("Dijkstra",        lambda: dijkstra(graph, loader, starts, ends, start_time)),
            ("A*",              lambda: astar_time(graph, loader, starts, ends, start_time)),
            ("Bidirectional A*",lambda: astar_bidirectional(graph, loader, starts, ends, start_time)),
        ]
    else:
        algos = [
            ("Dijkstra (p)",    lambda: dijkstra_transfers(graph, loader, starts, ends, start_time)),
            ("A* (p)",          lambda: astar_transfers(graph, loader, starts, ends, start_time)),
        ]

    for name, fn in algos:
        t0 = _time.perf_counter()
        path, cost, nodes = fn()
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        cost_str = f"{cost/60:.1f} min" if criterion == "t" else f"{int(cost)} transfers"
        rows.append((name, cost_str, nodes, f"{elapsed_ms:.1f} ms"))

    # Print table to stderr
    print("\n── Algorithm comparison ─────────────────────────────────────────────", file=sys.stderr)
    header = f"{'Algorithm':<22} {'Cost':<14} {'Nodes':>8} {'Time':>10}"
    print(header, file=sys.stderr)
    print("─" * len(header), file=sys.stderr)
    baseline_nodes = rows[0][2]
    for name, cost_str, nodes, time_str in rows:
        savings = f"  ({(1-nodes/baseline_nodes)*100:+.0f}%)" if nodes != baseline_nodes else ""
        print(f"{name:<22} {cost_str:<14} {nodes:>8}{savings:<10} {time_str:>10}", file=sys.stderr)
    print("─" * len(header), file=sys.stderr)


def run_tabu_compare(graph, loader, start_name: str, visit_names: list,
                     criterion: str, start_time: int, max_iter: int = 100):
    """
    Run Tabu Search with several T sizes and print a comparison table (Task 2b).
    Also compares full neighbourhood vs 50% sampling (Task 2d) and
    with/without aspiration (Task 2c).
    """
    import time as _time, math
    from tabu_search import TabuSearch

    n = len(visit_names) + 1   # includes start stop
    all_stops = [start_name] + visit_names

    configs = [
        # (label, tabu_size, aspiration, sample_fraction)
        ("Greedy + unbounded T",     0,           False, 1.0),
        (f"Greedy + T=√n={int(n**0.5)}", int(n**0.5), False, 1.0),
        (f"Greedy + T=n={n}",        n,           False, 1.0),
        (f"Greedy + T=2n={2*n}",     2*n,         False, 1.0),
        ("Greedy + asp",             0,           True,  1.0),
        ("Greedy + 50% sample",      0,           False, 0.5),
    ]

    print("\n── Tabu Search comparison ───────────────────────────────────────────", file=sys.stderr)
    header = f"{'Configuration':<28} {'Cost':>12} {'Iters':>6} {'Asp':>5} {'Time':>10}"
    print(header, file=sys.stderr)
    print("─" * len(header), file=sys.stderr)

    for label, tsize, asp, frac in configs:
        ts = TabuSearch(
            stops_list=all_stops, graph=graph, loader=loader,
            criterion=criterion, start_time=start_time,
            tabu_size=tsize, use_aspiration=asp,
            sample_fraction=frac, max_iter=max_iter, seed=42,
        )
        t0 = _time.perf_counter()
        _, best_cost, _ = ts.run()
        elapsed_ms = (_time.perf_counter() - t0) * 1000
        cost_str = f"{best_cost/60:.1f} min" if criterion == "t" else f"{int(best_cost)} tr"
        print(
            f"{label:<28} {cost_str:>12} {ts.iterations_done:>6} "
            f"{ts.aspiration_activations:>5} {elapsed_ms:>9.0f} ms",
            file=sys.stderr,
        )
    print("─" * len(header), file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args   = parser.parse_args()

    # Resolve query date: --date flag > GTFS_DATE env var > default
    if args.date:
        query_date = parse_date_arg(args.date)
    elif os.environ.get('GTFS_DATE'):
        query_date = parse_date_arg(os.environ['GTFS_DATE'])
    else:
        query_date = DEFAULT_DATE

    # Parse start time
    start_time = parse_time_arg(args.time)

    # Load GTFS data
    print(f"Loading GTFS data for {query_date} ...", file=sys.stderr)
    loader = GTFSLoader(GTFS_DIR)
    loader.compute_active_services(query_date)
    graph  = TransitGraph(loader)
    print("GTFS data loaded.", file=sys.stderr)

    end_name = args.end_stops

    # -----------------------------------------------------------------------
    # --compare: algorithm benchmark table (Task 1d requirement)
    # -----------------------------------------------------------------------
    if args.compare:
        run_compare(graph, loader, args.start, end_name, args.criterion, start_time)
        return

    # -----------------------------------------------------------------------
    # --tabu-compare: Tabu parameter benchmark table (Tasks 2b/2c/2d)
    # -----------------------------------------------------------------------
    if args.tabu_compare:
        visit_names = [s.strip() for s in end_name.split(";") if s.strip()]
        if not visit_names:
            print("ERROR: --tabu-compare requires a semicolon-separated stop list.", file=sys.stderr)
            sys.exit(1)
        run_tabu_compare(graph, loader, args.start, visit_names,
                         args.criterion, start_time, args.max_iter)
        return

    # -----------------------------------------------------------------------
    # Task 2 — TSP tour
    # -----------------------------------------------------------------------
    if args.task2:
        from tabu_search import run_tabu

        visit_names = [s.strip() for s in end_name.split(";") if s.strip()]
        if not visit_names:
            print("ERROR: --task2 requires at least one stop in the list.", file=sys.stderr)
            sys.exit(1)

        run_tabu(
            graph=graph,
            loader=loader,
            start_name=args.start,
            visit_names=visit_names,
            criterion=args.criterion,
            start_time=start_time,
            tabu_size=args.tabu_size,
            variable_t=args.variable_t,
            aspiration=args.aspiration,
            sample_fraction=args.sample,
            max_iter=args.max_iter,
        )
        return

    # -----------------------------------------------------------------------
    # Task 1 — single shortest path
    # -----------------------------------------------------------------------
    if args.algo == "dijkstra":
        from dijkstra import run_dijkstra
        run_dijkstra(graph, loader, args.start, end_name, args.criterion, start_time)

    elif args.algo == "astar":
        from astar import run_astar
        run_astar(graph, loader, args.start, end_name, args.criterion, start_time,
                  bidirectional=False)

    elif args.algo == "bidirectional":
        if args.criterion != "t":
            print("ERROR: Bidirectional A* only supports criterion 't'.", file=sys.stderr)
            sys.exit(1)
        from astar import run_astar
        run_astar(graph, loader, args.start, end_name, args.criterion, start_time,
                  bidirectional=True)

    else:
        print(f"ERROR: Unknown algorithm '{args.algo}'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
