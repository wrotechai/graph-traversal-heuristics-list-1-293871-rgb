"""
graph.py — Build a time-dependent directed graph from GTFS data.

Nodes: stop_ids (platforms).
Edges:
  - Trip edges: consecutive stops within one trip (same vehicle run).
  - Transfer edges: implicit — handled in search algorithms by collecting
    all departures from the same parent station.

The graph is stored as:
  departures[stop_id] = sorted list of departure events:
      {
        "dep":      int,   # departure time in seconds since midnight
        "arr":      int,   # arrival time at next stop (seconds)
        "to_stop":  str,   # destination stop_id
        "trip_id":  str,
        "route":    str,   # human-readable route name (e.g. "D1")
        "pickup":   str,   # "0" = can board, "1" = no boarding
      }

This representation allows O(log n) lookup of the next available departure
from any stop at any time (binary search on dep).
"""

from collections import defaultdict
from gtfs_loader import GTFSLoader
import bisect

# Minimum transfer time (seconds) when a passenger changes trip/line within
# the same parent station.  Both Dijkstra and A* enforce this so that a
# line-change is only considered when the connecting departure leaves at least
# this many seconds after the passenger's arrival on the platform.
TRANSFER_PENALTY = 120  # 2 minutes


class TransitGraph:
    """Time-dependent transit graph built from a GTFSLoader instance."""

    def __init__(self, loader: GTFSLoader):
        self.loader = loader
        # departures[stop_id] -> list of edge dicts, sorted by dep time
        self.departures: dict[str, list] = defaultdict(list)
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        active_trips = self.loader.get_active_trips()
        for tid in active_trips:
            stops = self.loader.stop_times[tid]
            route = self.loader.get_route_name(tid)
            for i in range(len(stops) - 1):
                cur  = stops[i]
                nxt  = stops[i + 1]
                # If pickup_type == 1 at current stop, passengers cannot board here
                if cur["pickup_type"] == "1":
                    continue
                edge = {
                    "dep":     cur["departure"],
                    "arr":     nxt["arrival"],
                    "to_stop": nxt["stop_id"],
                    "trip_id": tid,
                    "route":   route,
                    "pickup":  cur["pickup_type"],
                }
                self.departures[cur["stop_id"]].append(edge)

        # Sort each stop's departures by departure time for binary search
        for stop_id in self.departures:
            self.departures[stop_id].sort(key=lambda e: e["dep"])

    # ------------------------------------------------------------------
    def next_departures(self, stop_id: str, from_time: int) -> list:
        """
        Return all departure edges from stop_id with dep >= from_time.
        Also considers all platform siblings at the same parent station.
        """
        results = []
        # Include all sibling platforms (handles parent_station grouping)
        for sid in self.loader.get_station_stops(stop_id):
            edges = self.departures.get(sid, [])
            if not edges:
                continue
            # Binary search for first edge with dep >= from_time
            keys = [e["dep"] for e in edges]
            idx = bisect.bisect_left(keys, from_time)
            results.extend(edges[idx:])
        return results

    # ------------------------------------------------------------------
    def next_departures_single(self, stop_id: str, from_time: int) -> list:
        """
        Return departure edges only from this specific stop_id with dep >= from_time.
        Does not consider sibling platforms.
        """
        edges = self.departures.get(stop_id, [])
        if not edges:
            return []
        # Binary search for first edge with dep >= from_time
        keys = [e["dep"] for e in edges]
        idx = bisect.bisect_left(keys, from_time)
        return edges[idx:]

    # ------------------------------------------------------------------
    def get_all_stops(self) -> set:
        """Return all stop_ids that appear in the graph (have departures)."""
        return set(self.departures.keys())
