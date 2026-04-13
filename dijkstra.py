"""
dijkstra.py — Dijkstra's algorithm for the transit graph.

Supports two criteria (selected via `criterion` argument):

Criterion 't' (travel time):
  State : stop_id
  Cost  : arrival time in seconds since midnight.
  dist[stop_id] = best arrival time found so far.

Criterion 'p' (minimum transfers):
  State : (stop_id, trip_id)
  Cost  : (transfers, arrival_time)  — lexicographic comparison.
  A transfer is counted every time the passenger boards a different trip_id.
  Using (stop_id, trip_id) as the state (same as astar_transfers) ensures
  that Dijkstra finds the globally optimal transfer-minimising path.

Output (both criteria):
  stdout — one line per segment:
      <from_stop_name> | <to_stop_name> | <route> | <dep HH:MM:SS> | <arr HH:MM:SS>
  stderr — criterion value + wall-clock computation time
"""

import heapq
import time
from graph import TransitGraph, TRANSFER_PENALTY
from gtfs_loader import GTFSLoader, seconds_to_hhmm


# ---------------------------------------------------------------------------
# Criterion 't' — travel time
# ---------------------------------------------------------------------------

def dijkstra(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_stops: list,
    end_stops: set,
    start_time: int,
) -> tuple:
    """
    Dijkstra minimising total travel time.

    State: (stop_id, trip_id) so that in-station line changes are handled
    correctly.  When the passenger switches to a different trip_id within
    the same parent station a TRANSFER_PENALTY (seconds) is enforced: the
    departure must be at least (current_arrival + TRANSFER_PENALTY) away.
    This models the minimum walk/wait needed between platforms.

    next_departures() is used so all sibling platforms at the same station
    are considered at every expansion step.

    Returns (path, cost_seconds, nodes_visited).
    """
    # dist[(stop_id, trip_id)] -> best arrival time
    dist = {}
    prev = {}

    pq = []  # (arrival_time, stop_id, trip_id)
    for s in start_stops:
        state = (s, None)
        heapq.heappush(pq, (start_time, s, None))
        dist[state] = start_time
        prev[state] = None

    nodes_visited = 0

    while pq:
        arr_time, u, cur_trip = heapq.heappop(pq)

        state = (u, cur_trip)
        if arr_time > dist.get(state, float("inf")):
            continue  # stale

        nodes_visited += 1

        if u in end_stops:
            path = _reconstruct_path(prev, state)
            return path, arr_time - start_time, nodes_visited

        for edge in graph.next_departures(u, arr_time):
            v        = edge["to_stop"]
            trip     = edge["trip_id"]
            dep_time = edge["dep"]
            new_arr  = edge["arr"]

            # Enforce minimum transfer time when changing lines at a station
            if cur_trip is not None and trip != cur_trip:
                if dep_time < arr_time + TRANSFER_PENALTY:
                    continue

            new_state = (v, trip)
            if new_arr < dist.get(new_state, float("inf")):
                dist[new_state] = new_arr
                prev[new_state] = (state, edge)
                heapq.heappush(pq, (new_arr, v, trip))

    return None, float("inf"), nodes_visited


# ---------------------------------------------------------------------------
# Criterion 'p' — minimum transfers
# ---------------------------------------------------------------------------

def dijkstra_transfers(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_stops: list,
    end_stops: set,
    start_time: int,
) -> tuple:
    """
    Dijkstra minimising number of transfers.

    State: (stop_id, trip_id) — same representation as astar_transfers so
    that changing trip at a stop is correctly counted as one transfer.

    Returns (path, transfers, nodes_visited).
    """
    INF = float("inf")

    # dist[(stop_id, trip_id)] = (transfers, arrival_time)
    dist = {}
    prev = {}
    nodes_visited = 0

    pq = []  # (transfers, arrival_time, stop_id, trip_id)
    for s in start_stops:
        state = (s, None)
        dist[state] = (0, start_time)
        prev[state] = None
        heapq.heappush(pq, (0, start_time, s, None))

    while pq:
        transfers, arr_time, u, cur_trip = heapq.heappop(pq)

        state = (u, cur_trip)
        best_t, best_a = dist.get(state, (INF, INF))
        if (transfers, arr_time) > (best_t, best_a):
            continue  # stale

        nodes_visited += 1

        if u in end_stops:
            path = _reconstruct_path_transfer(prev, state)
            return path, transfers, nodes_visited

        for edge in graph.next_departures(u, arr_time):
            v       = edge["to_stop"]
            trip    = edge["trip_id"]
            new_arr = edge["arr"]

            transfer_cost = 1 if (cur_trip is not None and trip != cur_trip) else 0
            new_t = transfers + transfer_cost
            new_state = (v, trip)

            prev_t, prev_a = dist.get(new_state, (INF, INF))
            if (new_t, new_arr) < (prev_t, prev_a):
                dist[new_state] = (new_t, new_arr)
                prev[new_state] = (state, edge)
                heapq.heappush(pq, (new_t, new_arr, v, trip))

    return None, INF, nodes_visited


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------

def _reconstruct_path(prev: dict, end_state: tuple) -> list:
    """
    For time criterion — prev keys are (stop_id, trip_id) tuples.
    Walks the prev chain back to the origin and returns the ordered segment list.
    """
    segments = []
    cur = end_state
    while prev[cur] is not None:
        parent_state, edge = prev[cur]
        parent_stop = parent_state[0]
        segments.append({
            "from_stop": parent_stop,
            "to_stop":   edge["to_stop"],
            "route":     edge["route"],
            "dep":       edge["dep"],
            "arr":       edge["arr"],
        })
        cur = parent_state
    segments.reverse()
    return segments


def _reconstruct_path_transfer(prev: dict, end_state: tuple) -> list:
    """For transfer criterion — prev keys are (stop_id, trip_id) tuples."""
    segments = []
    cur = end_state
    while prev[cur] is not None:
        parent_state, edge = prev[cur]
        parent_stop = parent_state[0]
        segments.append({
            "from_stop": parent_stop,
            "to_stop":   edge["to_stop"],
            "route":     edge["route"],
            "dep":       edge["dep"],
            "arr":       edge["arr"],
        })
        cur = parent_state
    segments.reverse()
    return segments


# ---------------------------------------------------------------------------
# Output formatting (shared with astar.py)
# ---------------------------------------------------------------------------

def format_path(segments: list, loader: GTFSLoader) -> str:
    """Format path segments for stdout output."""
    lines = []
    for seg in segments:
        from_name = loader.stops.get(seg["from_stop"], {}).get("stop_name", seg["from_stop"])
        to_name   = loader.stops.get(seg["to_stop"],   {}).get("stop_name", seg["to_stop"])
        dep_str   = seconds_to_hhmm(seg["dep"])
        arr_str   = seconds_to_hhmm(seg["arr"])
        lines.append(f"{from_name} | {to_name} | {seg['route']} | {dep_str} | {arr_str}")
    return "\n".join(lines)


def count_platform_changes(segments: list, loader: GTFSLoader) -> int:
    """Count additional transfers for platform changes within the same station."""
    extra_transfers = 0
    for i in range(1, len(segments)):
        prev_to = segments[i-1]["to_stop"]
        curr_from = segments[i]["from_stop"]
        if prev_to != curr_from:
            prev_parent = loader.get_parent_station(prev_to)
            curr_parent = loader.get_parent_station(curr_from)
            if prev_parent == curr_parent and prev_to != curr_from:
                extra_transfers += 1
    return extra_transfers


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_dijkstra(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_name: str,
    end_name: str,
    criterion: str,
    start_time: int,
):
    """Entry point called from main.py for Task 1a."""
    import sys

    start_stops = loader.stop_name_to_ids(start_name)
    end_stops   = set(loader.stop_name_to_ids(end_name))

    if not start_stops:
        print(f"ERROR: Start stop '{start_name}' not found.", file=sys.stderr)
        return None
    if not end_stops:
        print(f"ERROR: End stop '{end_name}' not found.", file=sys.stderr)
        return None

    t0 = time.perf_counter()

    if criterion == "t":
        path, cost, nodes = dijkstra(graph, loader, start_stops, end_stops, start_time)
        label = f"Travel time: {cost/60:.1f} min"
    elif criterion == "p":
        path, cost, nodes = dijkstra_transfers(graph, loader, start_stops, end_stops, start_time)
        extra = count_platform_changes(path, loader)
        total_transfers = cost + extra
        label = f"Transfers: {total_transfers}"
    else:
        print(f"ERROR: Unknown criterion '{criterion}'. Use 't' or 'p'.", file=sys.stderr)
        return None

    elapsed = time.perf_counter() - t0

    if path is None:
        print("No path found.", file=sys.stderr)
        return None

    print(format_path(path, loader))
    print(f"{label} | Nodes visited: {nodes} | Computation: {elapsed*1000:.1f} ms  [Dijkstra]",
          file=sys.stderr)

    return path, cost, nodes, elapsed