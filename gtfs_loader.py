"""
gtfs_loader.py — Load and preprocess GTFS data for Koleje Dolnośląskie.

Handles:
- calendar.txt  (weekly service patterns + date range)
- calendar_dates.txt (holiday exceptions: add/remove service)
- stops.txt  (platforms + parent_station mapping)
- trips.txt  (route → service_id mapping)
- stop_times.txt (timetable; post-midnight times like 25:10:00)
- routes.txt (route_short_name / route_long_name)
"""

import csv
import os
from datetime import date, datetime, timedelta
from collections import defaultdict


# ---------------------------------------------------------------------------
# Time utilities
# ---------------------------------------------------------------------------

def parse_gtfs_time(time_str: str) -> int:
    """Convert 'HH:MM:SS' (possibly HH >= 24) to seconds since midnight."""
    h, m, s = time_str.strip().split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def seconds_to_hhmm(seconds: int) -> str:
    """Convert seconds-since-midnight to human-readable HH:MM:SS string."""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_date(date_str: str) -> date:
    """Parse YYYYMMDD string to date object."""
    return datetime.strptime(date_str.strip(), "%Y%m%d").date()


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

class GTFSLoader:
    """Load all GTFS files and provide query interfaces."""

    def __init__(self, gtfs_dir: str):
        self.gtfs_dir = gtfs_dir
        self.stops = {}          # stop_id -> dict
        self.parent_map = {}     # stop_id -> parent_station stop_id (or itself)
        self.children_map = defaultdict(list)  # parent_id -> [child stop_ids]
        self.routes = {}         # route_id -> route_short_name
        self.trips = {}          # trip_id -> {route_id, service_id}
        self.stop_times = defaultdict(list)    # trip_id -> [stop_time dicts], sorted by stop_sequence
        self.active_services = set()  # service_ids active on query date
        self._load_all()

    # ------------------------------------------------------------------
    def _load_all(self):
        self._load_stops()
        self._load_routes()
        self._load_trips()
        self._load_stop_times()

    # ------------------------------------------------------------------
    def _path(self, filename: str) -> str:
        return os.path.join(self.gtfs_dir, filename)

    # ------------------------------------------------------------------
    def _load_stops(self):
        with open(self._path("stops.txt"), encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["stop_id"].strip()
                self.stops[sid] = {
                    "stop_id":   sid,
                    "stop_name": row["stop_name"].strip(),
                    "lat":       float(row["stop_lat"]),
                    "lon":       float(row["stop_lon"]),
                    "location_type": row.get("location_type", "0").strip(),
                    "parent_station": row.get("parent_station", "").strip(),
                }
        # Build parent / children maps
        for sid, info in self.stops.items():
            parent = info["parent_station"]
            if parent:
                self.parent_map[sid] = parent
                self.children_map[parent].append(sid)
            else:
                self.parent_map[sid] = sid  # itself is its own "station"

    # ------------------------------------------------------------------
    def _load_routes(self):
        with open(self._path("routes.txt"), encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row["route_id"].strip()
                short = row.get("route_short_name", "").strip()
                long  = row.get("route_long_name",  "").strip()
                self.routes[rid] = short if short else long

    # ------------------------------------------------------------------
    def _load_trips(self):
        with open(self._path("trips.txt"), encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row["trip_id"].strip()
                self.trips[tid] = {
                    "route_id":   row["route_id"].strip(),
                    "service_id": row["service_id"].strip(),
                }

    # ------------------------------------------------------------------
    def _load_stop_times(self):
        with open(self._path("stop_times.txt"), encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = row["trip_id"].strip()
                # pickup_type == 1 means no boarding allowed
                pickup = row.get("pickup_type", "0").strip()
                entry = {
                    "stop_id":       row["stop_id"].strip(),
                    "arrival":       parse_gtfs_time(row["arrival_time"]),
                    "departure":     parse_gtfs_time(row["departure_time"]),
                    "stop_sequence": int(row["stop_sequence"]),
                    "pickup_type":   pickup,
                }
                self.stop_times[tid].append(entry)
        # Sort each trip by stop_sequence once
        for tid in self.stop_times:
            self.stop_times[tid].sort(key=lambda x: x["stop_sequence"])

    # ------------------------------------------------------------------
    def compute_active_services(self, query_date: date):
        """
        Determine which service_ids are active on query_date.
        Uses calendar.txt (weekly schedule) + calendar_dates.txt (exceptions).
        """
        active = set()
        day_name = query_date.strftime("%A").lower()  # e.g. "monday"

        # 1. Weekly schedule
        with open(self._path("calendar.txt"), encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                start = parse_date(row["start_date"])
                end   = parse_date(row["end_date"])
                if start <= query_date <= end:
                    if row[day_name].strip() == "1":
                        active.add(row["service_id"].strip())

        # 2. Exceptions from calendar_dates.txt
        date_str = query_date.strftime("%Y%m%d")
        with open(self._path("calendar_dates.txt"), encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["date"].strip() == date_str:
                    sid  = row["service_id"].strip()
                    etype = row["exception_type"].strip()
                    if etype == "1":
                        active.add(sid)       # service added on this date
                    elif etype == "2":
                        active.discard(sid)   # service removed on this date

        self.active_services = active

    # ------------------------------------------------------------------
    def get_active_trips(self) -> list:
        """Return list of trip_ids whose service is active."""
        return [tid for tid, info in self.trips.items()
                if info["service_id"] in self.active_services]

    # ------------------------------------------------------------------
    def stop_name_to_ids(self, name: str) -> list:
        """
        Find all platform stop_ids for a given station name.

        Strategy (in order of priority):
          1. Exact match on stop_name (case-insensitive, stripped).
          2. If no exact match found, substring match as fallback.

        Do NOT match by exact stop_id here (e.g. user enters a station name).
        This is important for names like "Wrocław Główny" where stop_id is a numeric
        code and will never equal the human-readable station name.

        For each matched stop_id:
          - If it is a parent station (location_type=1), return all its children.
          - If it is a platform (location_type=0 with a parent), return all siblings.
          - If it has no parent, return it directly.

        This correctly handles "Wrocław Główny" NOT matching "Wrocław Główny Nadodrze".
        """
        name_lower = name.strip().lower()
        # Avoid accidentally treating the query as a stop_id; only match stop_name.
        if name in self.stops:
            # Continue to stop_name based lookup to honor the requirement.
            pass

        # --- Pass 1: exact match ---
        exact_ids = [
            sid for sid, info in self.stops.items()
            if info["stop_name"].lower() == name_lower
        ]

        # --- Pass 2: substring match (only if no exact match) ---
        if not exact_ids:
            exact_ids = [
                sid for sid, info in self.stops.items()
                if name_lower in info["stop_name"].lower()
            ]

        if not exact_ids:
            return []

        # Expand to all platform children of the matched stations
        platform_ids = set()
        for sid in exact_ids:
            info = self.stops[sid]
            loc_type = info["location_type"]

            if loc_type == "1":
                # It's a parent station → return all children platforms
                children = self.children_map.get(sid, [])
                if children:
                    platform_ids.update(children)
                else:
                    platform_ids.add(sid)
            else:
                # It's a platform → include all siblings (same parent station)
                parent = self.parent_map.get(sid, sid)
                siblings = self.children_map.get(parent, [])
                if siblings:
                    platform_ids.update(siblings)
                else:
                    platform_ids.add(sid)

        return list(platform_ids)

    # ------------------------------------------------------------------
    def get_station_stops(self, stop_id: str) -> list:
        """Return all platform stop_ids belonging to the same parent station."""
        parent = self.parent_map.get(stop_id, stop_id)
        children = self.children_map.get(parent, [])
        # Also include the parent itself if it's a valid stop
        result = list(children)
        if parent in self.stops and parent not in result:
            result.append(parent)
        if not result:
            result = [stop_id]
        return result

    # ------------------------------------------------------------------
    def get_parent_station(self, stop_id: str) -> str:
        """Return the parent station stop_id for a given stop_id."""
        return self.parent_map.get(stop_id, stop_id)

    # ------------------------------------------------------------------
    def get_route_name(self, trip_id: str) -> str:
        """Return the route short (or long) name for a trip."""
        route_id = self.trips[trip_id]["route_id"]
        return self.routes.get(route_id, route_id)
