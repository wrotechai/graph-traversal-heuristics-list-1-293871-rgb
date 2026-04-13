"""
tabu_search.py — Tabu Search for the transit TSP.

Given a starting stop A and a list L = [A2, ..., An] of stops to visit,
find the shortest closed tour A → (permutation of L) → A.

The edge cost between two stops is computed by running A* with the ACTUAL
arrival time at each intermediate stop — making the cost fully time-dependent.
This correctly handles Test S2.3 (asymmetric costs) because:
  cost(A→B→C) ≠ cost(A→C→B)  in general (different train connections available).

Four variants are implemented:
  2a. Basic Tabu Search with unbounded T (T grows forever).
  2b. Variable T size:  T ∈ [√n, n], parameterised on |L|.
  2c. Aspiration criterion: allow a tabu move if it produces a new global best.
  2d. Neighbourhood sampling: instead of evaluating all C(n,2) swap pairs,
      randomly sample a fixed fraction of them.

Neighbourhood: 2-opt (reverse a sub-sequence of the tour).
  For n stops, the full neighbourhood has C(n,2) = n*(n-1)/2 pairs.
"""

import math
import random
import time

from graph import TransitGraph
from gtfs_loader import GTFSLoader, seconds_to_hhmm
from astar import astar_time, astar_transfers


# ---------------------------------------------------------------------------
# Time-dependent tour cost
# ---------------------------------------------------------------------------

def compute_segment_cost(
    from_name: str,
    to_name: str,
    depart_time: int,
    graph: TransitGraph,
    loader: GTFSLoader,
    criterion: str,
) -> tuple:
    """
    Run A* from from_name to to_name starting at depart_time.
    Returns (cost, arrival_time).
    cost = travel seconds (criterion t) or transfer count (criterion p).
    arrival_time = absolute seconds-since-midnight when we reach to_name.
    """
    start_ids = loader.stop_name_to_ids(from_name)
    end_ids   = set(loader.stop_name_to_ids(to_name))
    if not start_ids or not end_ids:
        return float("inf"), depart_time

    if criterion == "t":
        path, cost, _ = astar_time(graph, loader, start_ids, end_ids, depart_time)
    else:
        path, cost, _ = astar_transfers(graph, loader, start_ids, end_ids, depart_time)
        if path is not None:
            from dijkstra import count_platform_changes
            extra = count_platform_changes(path, loader)
            cost += extra

    if path is None or cost == float("inf"):
        return float("inf"), depart_time

    arrival = path[-1]["arr"] if path else depart_time
    return cost, arrival


def tour_cost_timedep(
    tour: list,
    stops_list: list,
    start_time: int,
    graph: TransitGraph,
    loader: GTFSLoader,
    criterion: str,
) -> tuple:
    """
    Compute total cost of a tour by chaining A* calls with actual arrival times.

    tour  : list of indices into stops_list (index 0 = start/end stop A)
    Returns (total_cost, list_of_arrival_times).

    This is the correct implementation for Test S2.3:
    each segment starts from the arrival time of the previous segment.
    """
    total_cost = 0
    current_time = start_time
    arrivals = []

    # Closed tour: follow the tour order, then return to stop 0
    full_tour = tour + [0]

    for k in range(len(full_tour) - 1):
        from_name = stops_list[full_tour[k]]
        to_name   = stops_list[full_tour[k + 1]]
        seg_cost, arrival = compute_segment_cost(
            from_name, to_name, current_time, graph, loader, criterion
        )
        total_cost += seg_cost
        current_time = arrival
        arrivals.append(arrival)

    return total_cost, arrivals


# ---------------------------------------------------------------------------
# Neighbourhood: 2-opt
# ---------------------------------------------------------------------------

def two_opt_swap(tour: list, i: int, j: int) -> list:
    """Return new tour with the segment [i+1 .. j] reversed (2-opt move)."""
    return tour[:i + 1] + tour[i + 1:j + 1][::-1] + tour[j + 1:]


def full_neighbourhood(tour: list) -> list:
    """Return all (i, j) index pairs for 2-opt (i < j, j-i >= 2)."""
    n = len(tour)
    pairs = []
    for i in range(n - 1):
        for j in range(i + 2, n):
            pairs.append((i, j))
    return pairs


def sampled_neighbourhood(tour: list, sample_size: int, rng: random.Random) -> list:
    """Return a random sample of (i, j) 2-opt pairs."""
    all_pairs = full_neighbourhood(tour)
    if sample_size >= len(all_pairs):
        return all_pairs
    return rng.sample(all_pairs, sample_size)


# ---------------------------------------------------------------------------
# Tabu Search
# ---------------------------------------------------------------------------

class TabuSearch:
    """
    Tabu Search for TSP on transit graph (time-dependent costs).

    Parameters
    ----------
    stops_list      : list of stop names; index 0 is start/end stop A.
    graph           : TransitGraph instance.
    loader          : GTFSLoader instance.
    criterion       : 't' (travel time) or 'p' (transfers).
    start_time      : departure time in seconds since midnight.
    tabu_size       : FIFO tabu list size. 0 = unbounded.
    use_aspiration  : enable aspiration criterion (Task 2c).
    sample_fraction : fraction of neighbourhood to evaluate (Task 2d).
    max_iter        : max iterations.
    seed            : RNG seed.
    """

    def __init__(
        self,
        stops_list: list,
        graph: TransitGraph,
        loader: GTFSLoader,
        criterion: str,
        start_time: int,
        tabu_size: int = 0,
        use_aspiration: bool = False,
        sample_fraction: float = 1.0,
        max_iter: int = 200,
        seed: int = 42,
    ):
        self.stops           = stops_list
        self.graph           = graph
        self.loader          = loader
        self.criterion       = criterion
        self.start_time      = start_time
        self.n               = len(stops_list)
        self.tabu_size       = tabu_size
        self.use_aspiration  = use_aspiration
        self.sample_fraction = sample_fraction
        self.max_iter        = max_iter
        self.rng             = random.Random(seed)

        # Statistics
        self.aspiration_activations = 0
        self.iterations_done        = 0

    # ------------------------------------------------------------------
    def _cost(self, tour: list) -> float:
        """Compute time-dependent tour cost (full A* chain)."""
        cost, _ = tour_cost_timedep(
            tour, self.stops, self.start_time,
            self.graph, self.loader, self.criterion
        )
        return cost

    # ------------------------------------------------------------------
    def _initial_solution(self) -> list:
        """Initial tour: greedy nearest-neighbour."""
        # --- Greedy nearest-neighbour ---
        unvisited = list(range(1, self.n))
        tour = [0]
        current_time = self.start_time

        while unvisited:
            last_name = self.stops[tour[-1]]
            best_idx  = None
            best_cost = float("inf")
            best_arr  = current_time

            for idx in unvisited:
                seg_cost, arr = compute_segment_cost(
                    last_name, self.stops[idx],
                    current_time, self.graph, self.loader, self.criterion
                )
                if seg_cost < best_cost:
                    best_cost = seg_cost
                    best_idx  = idx
                    best_arr  = arr

            tour.append(best_idx)
            unvisited.remove(best_idx)
            current_time = best_arr

        return tour

    # ------------------------------------------------------------------
    def run(self) -> tuple:
        """
        Execute Tabu Search.

        Returns
        -------
        best_tour  : list of stop indices (does NOT repeat index 0 at end)
        best_cost  : float
        history    : list of (iteration, best_cost_so_far)
        """
        import sys
        print("Computing initial solution...", file=sys.stderr)
        s = self._initial_solution()
        best_cost = self._cost(s)
        s_star    = s[:]
        history   = [(0, best_cost)]
        print(f"  Initial tour cost: {best_cost/60:.1f} min  tour={s}", file=sys.stderr)

        tabu_list   = []   # FIFO list of (i, j) moves
        all_pairs   = full_neighbourhood(s)
        sample_size = max(1, int(len(all_pairs) * self.sample_fraction))

        for iteration in range(1, self.max_iter + 1):
            self.iterations_done = iteration

            # Generate neighbour candidates
            if self.sample_fraction < 1.0:
                pairs = sampled_neighbourhood(s, sample_size, self.rng)
            else:
                pairs = full_neighbourhood(s)

            best_neighbour      = None
            best_neighbour_cost = float("inf")
            best_move           = None

            for (i, j) in pairs:
                candidate = two_opt_swap(s, i, j)
                c_cost    = self._cost(candidate)
                move      = (i, j)
                move_rev  = (j, i)

                is_tabu = (move in tabu_list) or (move_rev in tabu_list)

                # Aspiration: allow tabu move if it beats global best
                if is_tabu and self.use_aspiration and c_cost < best_cost:
                    self.aspiration_activations += 1
                    is_tabu = False

                if not is_tabu and c_cost < best_neighbour_cost:
                    best_neighbour      = candidate
                    best_neighbour_cost = c_cost
                    best_move           = move

            if best_neighbour is None:
                break  # all moves tabu and aspiration not triggered

            s = best_neighbour

            # Update FIFO tabu list
            if best_move:
                tabu_list.append(best_move)
                if self.tabu_size > 0 and len(tabu_list) > self.tabu_size:
                    tabu_list.pop(0)

            # Update global best
            if best_neighbour_cost < best_cost:
                best_cost = best_neighbour_cost
                s_star    = s[:]

            history.append((iteration, best_cost))

        return s_star, best_cost, history


# ---------------------------------------------------------------------------
# Tour reconstruction to path segments
# ---------------------------------------------------------------------------

def reconstruct_full_tour(
    tour: list,
    stops_list: list,
    graph: TransitGraph,
    loader: GTFSLoader,
    criterion: str,
    start_time: int,
) -> list:
    """
    Re-run A* on consecutive stops in the tour to get detailed path segments.
    Uses real arrival times for each segment (time-dependent).
    Returns a flat list of segment dicts for printing.
    """
    from astar import astar_time, astar_transfers

    all_segments = []
    current_time = start_time
    full_tour_names = [stops_list[i] for i in tour] + [stops_list[0]]

    for k in range(len(full_tour_names) - 1):
        from_name = full_tour_names[k]
        to_name   = full_tour_names[k + 1]
        start_ids = loader.stop_name_to_ids(from_name)
        end_ids   = set(loader.stop_name_to_ids(to_name))

        if criterion == "t":
            path, _, _ = astar_time(graph, loader, start_ids, end_ids, current_time)
        else:
            path, _, _ = astar_transfers(graph, loader, start_ids, end_ids, current_time)

        if path:
            all_segments.extend(path)
            current_time = path[-1]["arr"]
        else:
            all_segments.append({
                "from_stop": from_name,
                "to_stop":   to_name,
                "route":     "N/A",
                "dep":       current_time,
                "arr":       current_time,
            })

    return all_segments


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tabu(
    graph: TransitGraph,
    loader: GTFSLoader,
    start_name: str,
    visit_names: list,
    criterion: str,
    start_time: int,
    tabu_size: int = 0,
    variable_t: bool = False,
    aspiration: bool = False,
    sample_fraction: float = 1.0,
    max_iter: int = 200,
):
    import sys
    from dijkstra import format_path

    all_stops = [start_name] + visit_names
    n = len(all_stops)

    # Variable T size (2b): clamp to [sqrt(n), n]
    if variable_t:
        effective_tabu_size = max(int(math.sqrt(n)), min(n, tabu_size if tabu_size > 0 else n))
    else:
        effective_tabu_size = tabu_size  # 0 = unbounded

    ts = TabuSearch(
        stops_list=all_stops,
        graph=graph,
        loader=loader,
        criterion=criterion,
        start_time=start_time,
        tabu_size=effective_tabu_size,
        use_aspiration=aspiration,
        sample_fraction=sample_fraction,
        max_iter=max_iter,
    )

    t0 = time.perf_counter()
    best_tour, best_cost, history = ts.run()
    elapsed = time.perf_counter() - t0

    segments = reconstruct_full_tour(best_tour, all_stops, graph, loader, criterion, start_time)
    print(format_path(segments, loader))

    label = f"Travel time: {best_cost/60:.1f} min" if criterion == "t" else f"Transfers: {int(best_cost)}"
    print(
        f"{label} | Iterations: {ts.iterations_done} | "
        f"Tabu size: {effective_tabu_size if effective_tabu_size > 0 else 'unbounded'} | "
        f"Aspiration activations: {ts.aspiration_activations} | "
        f"Sample fraction: {sample_fraction:.0%} | "
        f"Computation: {elapsed*1000:.0f} ms",
        file=sys.stderr,
    )

    return best_tour, best_cost, history
