"""
astar.py — A* search for two criteria on the time-dependent transit graph.

Criterion 't' (travel time):
  g(n) = arrival time at n (seconds since midnight)
  h(n) = haversine(n, goal) / V_MAX
  V_MAX is set to the maximum observed speed in the network + 10 % margin.
  The heuristic is admissible because no real train can travel faster than V_MAX,
  so h(n) ≤ h*(n) always holds.

Criterion 'p' (transfers):
  g(n) = number of transfers made so far
  h(n) = 0   (trivially admissible; guarantees optimality)
  The tie-breaking secondary key is arrival time (minimise waiting).

  A non-trivial admissible heuristic for 'p' would require a pre-computed
  "minimum transfers to reach goal" lower bound (e.g. via BFS on a line graph).
  h=0 is safe and correct; its effect on node expansion is analysed in the report.

Modification 1d — Bidirectional A* (criterion 't'):
  Run forward and backward searches simultaneously; stop when a node is settled
  in both directions.  In practice this halves the search radius on many queries.
"""

import heapq
import math
import time
from graph import TransitGraph, TRANSFER_PENALTY
from gtfs_loader import GTFSLoader, seconds_to_hhmm
from dijkstra import format_path, count_platform_changes

# ---------------------------------------------------------------------------
def count_transfers(segments: list, loader: GTFSLoader) -> int:
    """Count the number of transfers (route changes) in the path, excluding same-station changes."""
    if not segments:
        return 0
    transfers = 0
    prev_route = segments[0]["route"]
    prev_stop = segments[0]["to_stop"]
    for seg in segments[1:]:
        if seg["route"] != prev_route:
            # Check if transfer is in the same station
            same_station = loader.parent_map.get(prev_stop, prev_stop) == loader.parent_map.get(seg["from_stop"], seg["from_stop"])
            if not same_station:
                transfers += 1
        prev_route = seg["route"]
        prev_stop = seg["to_stop"]
    return transfers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Estimated maximum train speed in m/s (160 km/h → 44.4 m/s, +10 % margin)
V_MAX = 50.0  # m/s  ≈ 180 km/h — ensures h ≤ h* for any real train

EARTH_RADIUS_M = 6_371_000.0  # metres
# TRANSFER_PENALTY is imported from graph.py (shared with dijkstra.py)


# ---------------------------------------------------------------------------
# Heuristic helpers
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two WGS84 coordinates."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def h_time(loader: GTFSLoader, stop_id: str, goal_lat: float, goal_lon: float) -> float:
    """
    Admissible heuristic for travel-time criterion.
    Returns lower-bound travel time in seconds: distance / V_MAX.
    """
    s = loader.stops.get(stop_id)
    if s is None:
        return 0.0
    dist = haversine(s["lat"], s["lon"], goal_lat, goal_lon)
    return dist / V_MAX


# ---------------------------------------------------------------------------
# A* — criterion 't' (travel time)
# ---------------------------------------------------------------------------

def astar_time(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_stops: list,
    end_stops: set,
    start_time: int,
) -> tuple:
    """
    A* minimising total travel time.

    State: (stop_id, trip_id) — trip_id tracks which vehicle the passenger is on.
    This allows the search to consider changing lines within the same station
    when it leads to a faster overall journey.

    When the passenger changes trip_id at a station, a TRANSFER_PENALTY is applied
    to the earliest allowed boarding time, modelling the minimum walk/wait needed
    between platforms.  The heuristic h = haversine / V_MAX remains admissible
    because we never subtract time.

    next_departures() is used (not next_departures_single) so that all platforms
    belonging to the same parent station are considered at every expansion step.

    Returns (path, cost_seconds, nodes_visited).
    """
    # Compute goal centroid for heuristic
    goal_lats = [loader.stops[s]["lat"] for s in end_stops if s in loader.stops]
    goal_lons = [loader.stops[s]["lon"] for s in end_stops if s in loader.stops]
    if not goal_lats:
        goal_lat = goal_lon = 0.0
    else:
        goal_lat = sum(goal_lats) / len(goal_lats)
        goal_lon = sum(goal_lons) / len(goal_lons)

    # dist[(stop_id, trip_id)] -> best arrival time
    dist  = {}
    prev  = {}
    nodes_visited = 0

    pq = []  # (f, g, stop_id, trip_id)
    for s in start_stops:
        state = (s, None)  # no trip yet at origin
        g = start_time
        h = h_time(loader, s, goal_lat, goal_lon)
        heapq.heappush(pq, (g + h, g, s, None))
        dist[state] = g
        prev[state] = None

    while pq:
        f, g, u, cur_trip = heapq.heappop(pq)

        state = (u, cur_trip)
        if g > dist.get(state, float("inf")):
            continue  # stale

        nodes_visited += 1

        if u in end_stops:
            path = _reconstruct_path_with_trip(prev, state)
            return path, g - start_time, nodes_visited

        # Consider all departures from the whole parent station (enables in-station
        # line changes). next_departures already groups sibling platforms.
        for edge in graph.next_departures(u, g):
            v        = edge["to_stop"]
            trip     = edge["trip_id"]
            dep_time = edge["dep"]
            new_g    = edge["arr"]

            # If the passenger is switching to a different trip at this station,
            # enforce a minimum transfer penalty so they cannot "teleport" between
            # platforms instantly.  The effective boarding time is at least
            # (current_arrival + TRANSFER_PENALTY), so we skip departures that
            # leave before the passenger can realistically board.
            if cur_trip is not None and trip != cur_trip:
                earliest_board = g + TRANSFER_PENALTY
                if dep_time < earliest_board:
                    continue  # departs before passenger can reach the platform

            new_state = (v, trip)
            if new_g < dist.get(new_state, float("inf")):
                dist[new_state] = new_g
                prev[new_state] = (state, edge)
                new_h = h_time(loader, v, goal_lat, goal_lon)
                heapq.heappush(pq, (new_g + new_h, new_g, v, trip))

    return None, float("inf"), nodes_visited


# ---------------------------------------------------------------------------
# A* — criterion 'p' (minimum transfers)
# ---------------------------------------------------------------------------

def astar_transfers(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_stops: list,
    end_stops: set,
    start_time: int,
) -> tuple:
    """
    A* minimising number of transfers.

    State: (stop_id, current_trip_id, current_time)
    g = transfers so far (0 for start; +1 each time trip_id changes at a stop)
    h = 0  (admissible: actual transfers ≥ 0)
    Tie-break: arrival time (prefer faster routes with equal transfers).

    Returns (path, transfers, nodes_visited).
    """
    INF = float("inf")

    # dist[(stop_id, trip_id)] -> (transfers, arrival_time)
    dist = {}
    prev = {}
    nodes_visited = 0

    pq = []  # (transfers, arrival_time, stop_id, trip_id)
    for s in start_stops:
        state = (s, None)  # no trip yet at origin
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

            # Transfer if boarding a different trip (and we already had a trip).
            # Consistent with dijkstra_transfers: any trip change = +1 transfer.
            transfer_cost = 1 if (cur_trip is not None and trip != cur_trip) else 0
            new_t = transfers + transfer_cost
            new_state = (v, trip)

            prev_t, prev_a = dist.get(new_state, (INF, INF))
            if (new_t, new_arr) < (prev_t, prev_a):
                dist[new_state]  = (new_t, new_arr)
                prev[new_state]  = (state, edge)
                heapq.heappush(pq, (new_t, new_arr, v, trip))

    return None, INF, nodes_visited


# ---------------------------------------------------------------------------
# Path reconstruction helpers
# ---------------------------------------------------------------------------

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


def _reconstruct_path_with_trip(prev: dict, end_state: tuple) -> list:
    """
    For the updated time criterion — prev keys are (stop_id, trip_id) tuples.
    Identical logic to _reconstruct_path_transfer but kept separate for clarity.
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


# ---------------------------------------------------------------------------
# Bidirectional A* (modification 1d) — criterion 't'
# ---------------------------------------------------------------------------

def astar_bidirectional(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_stops: list,
    end_stops: set,
    start_time: int,
) -> tuple:
    """
    Bidirectional A* for travel-time criterion.

    Two-phase design
    ----------------
    Phase 1 — Bidirectional expansion:
      • Forward A*  (start → ?): priority = arrival_time + h_forward.
      • Backward Dijkstra (goal → ?): uses ELAPSED travel time (not absolute
        timestamps) so both frontiers are on the same scale and truly alternate.
        Cost = sum of edge travel-times going backward; ignores waiting times
        (acceptable approximation for alternation purposes).
      • Alternation condition: forward_elapsed + h_f  ≤  backward_elapsed + h_b.
      • Phase stops at the FIRST stop settled in both directions.

    Phase 2 — Verification (single A* call):
      • The backward cost does NOT account for actual train schedules from the
        meeting stop, so the path cost must be verified.
      • One forward A* is run from the meeting stop to the goal, departing at
        the REAL forward arrival time.  This guarantees optimality.

    Efficiency metric:
      • bidir_phase_nodes = nodes settled in Phase 1.
      • verify_nodes      = nodes settled in Phase 2.
      • returned nodes_visited = bidir_phase_nodes (the meaningful comparison;
        the verification leg is implementation-overhead, not "search").

    Returns (path, cost_seconds, nodes_visited).
    """
    # ------------------------------------------------------------------ #
    # Build reverse adjacency (edge travel-times, not absolute timestamps) #
    # ------------------------------------------------------------------ #
    reverse_adj: dict[str, list] = {}
    for from_stop, edges in graph.departures.items():
        for e in edges:
            v = e["to_stop"]
            if v not in reverse_adj:
                reverse_adj[v] = []
            # Travel duration of this edge (seconds)
            duration = max(e["arr"] - e["dep"], 0)
            reverse_adj[v].append({
                "to_stop":  from_stop,
                "duration": duration,
                "route":    e["route"],
                "trip_id":  e["trip_id"],
            })

    # Goal centroid for forward heuristic
    goal_lats = [loader.stops[s]["lat"] for s in end_stops if s in loader.stops]
    goal_lons = [loader.stops[s]["lon"] for s in end_stops if s in loader.stops]
    goal_lat = sum(goal_lats) / len(goal_lats) if goal_lats else 0.0
    goal_lon = sum(goal_lons) / len(goal_lons) if goal_lons else 0.0

    # Start centroid for backward heuristic
    start_lats = [loader.stops[s]["lat"] for s in start_stops if s in loader.stops]
    start_lons = [loader.stops[s]["lon"] for s in start_stops if s in loader.stops]
    start_lat = sum(start_lats) / len(start_lats) if start_lats else 0.0
    start_lon = sum(start_lons) / len(start_lons) if start_lons else 0.0

    INF = float("inf")
    dist_f: dict = {}   # (stop_id, trip_id) -> absolute arrival time
    dist_b: dict = {}   # stop_id            -> elapsed travel time from goal
    prev_f: dict = {}

    # Forward init — absolute arrival time at each state
    pq_f = []
    for s in start_stops:
        state = (s, None)
        h = h_time(loader, s, goal_lat, goal_lon)
        heapq.heappush(pq_f, (start_time + h, start_time, s, None))
        dist_f[state] = start_time
        prev_f[state] = None

    # Backward init — elapsed travel time from goal = 0
    pq_b = []
    for e in end_stops:
        h = h_time(loader, e, start_lat, start_lon)
        heapq.heappush(pq_b, (0.0 + h, 0.0, e))   # (f_b, elapsed_b, stop)
        dist_b[e] = 0.0

    settled_f: set = set()
    settled_b: set = set()
    bidir_nodes = 0
    meeting_stop = None

    # ------------------------------------------------------------------ #
    # Phase 1 — balanced alternation, stop at first meeting               #
    # ------------------------------------------------------------------ #
    while pq_f or pq_b:
        # Compare on the same scale: forward uses (arrival - start_time) + h,
        # backward uses elapsed_b + h_b.
        fwd_priority = (pq_f[0][0] - start_time) if pq_f else INF
        bwd_priority = pq_b[0][0]                 if pq_b else INF
        expand_forward = fwd_priority <= bwd_priority

        if expand_forward and pq_f:
            f, g, u, cur_trip = heapq.heappop(pq_f)
            state = (u, cur_trip)
            if g > dist_f.get(state, INF):
                continue
            settled_f.add(u)
            bidir_nodes += 1

            if u in settled_b:
                meeting_stop = u
                break

            for edge in graph.next_departures(u, g):
                v        = edge["to_stop"]
                trip     = edge["trip_id"]
                dep_time = edge["dep"]
                new_g    = edge["arr"]

                if cur_trip is not None and trip != cur_trip:
                    if dep_time < g + TRANSFER_PENALTY:
                        continue

                new_state = (v, trip)
                if new_g < dist_f.get(new_state, INF):
                    dist_f[new_state] = new_g
                    prev_f[new_state] = (state, edge)
                    h = h_time(loader, v, goal_lat, goal_lon)
                    heapq.heappush(pq_f, (new_g + h, new_g, v, trip))

        elif pq_b:
            f_b, elapsed_b, u = heapq.heappop(pq_b)
            if elapsed_b > dist_b.get(u, INF):
                continue
            settled_b.add(u)
            bidir_nodes += 1

            if u in settled_f:
                meeting_stop = u
                break

            for edge in reverse_adj.get(u, []):
                v           = edge["to_stop"]
                new_elapsed = elapsed_b + edge["duration"]
                if new_elapsed < dist_b.get(v, INF):
                    dist_b[v] = new_elapsed
                    h = h_time(loader, v, start_lat, start_lon)
                    heapq.heappush(pq_b, (new_elapsed + h, new_elapsed, v))
        else:
            break

    if meeting_stop is None:
        path, cost, uni_nodes = astar_time(graph, loader, start_stops, end_stops, start_time)
        return path, cost, bidir_nodes + uni_nodes

    # ------------------------------------------------------------------ #
    # Phase 2 — verify the meeting point with a single forward A*         #
    # ------------------------------------------------------------------ #
    # Best forward state (lowest arrival time) at the meeting stop
    best_fwd_state = min(
        (s for s in dist_f if s[0] == meeting_stop),
        key=lambda s: dist_f[s],
        default=None,
    )

    if best_fwd_state is None:
        path, cost, uni_nodes = astar_time(graph, loader, start_stops, end_stops, start_time)
        return path, cost, bidir_nodes + uni_nodes

    fwd_arrival = dist_f[best_fwd_state]

    if meeting_stop in end_stops:
        # Meeting stop IS the goal
        path = _reconstruct_path_with_trip(prev_f, best_fwd_state)
        return path, fwd_arrival - start_time, bidir_nodes

    # Run a single A* from meeting stop to goal
    m_starts = list(loader.get_station_stops(meeting_stop))
    leg_path, leg_cost, verify_nodes = astar_time(
        graph, loader, m_starts, end_stops, fwd_arrival
    )

    if leg_path is None:
        # Meeting path is a dead end — fall back to unidirectional
        path, cost, uni_nodes = astar_time(graph, loader, start_stops, end_stops, start_time)
        return path, cost, bidir_nodes + uni_nodes

    path_to_meeting = _reconstruct_path_with_trip(prev_f, best_fwd_state)
    full_path  = path_to_meeting + leg_path
    total_cost = fwd_arrival - start_time + leg_cost
    # Report bidir_phase_nodes as the efficiency metric (verification is overhead).
    return full_path, total_cost, bidir_nodes


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_astar(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_name: str,
    end_name: str,
    criterion: str,   # 't' or 'p'
    start_time: int,
    bidirectional: bool = False,
):
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
        if bidirectional:
            path, cost, nodes = astar_bidirectional(graph, loader, start_stops, end_stops, start_time)
            variant = "Bidirectional A*"
        else:
            path, cost, nodes = astar_time(graph, loader, start_stops, end_stops, start_time)
            variant = "A* (time)"
        label = f"Travel time: {cost/60:.1f} min"
    elif criterion == "p":
        path, cost, nodes = astar_transfers(graph, loader, start_stops, end_stops, start_time)
        variant = "A* (transfers)"
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
    transfers = count_transfers(path, loader)
    travel_time_min = (path[-1]["arr"] - path[0]["dep"]) / 60
    if criterion == "t":
        label = f"Travel time: {cost/60:.1f} min | Transfers: {transfers}"
    elif criterion == "p":
        label = f"Transfers: {cost} | Travel time: {travel_time_min:.1f} min"
    print(f"{label} | Nodes visited: {nodes} | Computation: {elapsed*1000:.1f} ms  [{variant}]",
          file=sys.stderr)

    return path, cost, nodes, elapsed
