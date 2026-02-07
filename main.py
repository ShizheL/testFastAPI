import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from pycontrails import Flight
from pycontrails.datalib.ecmwf import ERA5
from pycontrails.models.cocip import Cocip
from pycontrails.models.ps_model import PSFlight

load_dotenv()

CDS_API_KEY = os.getenv("ECMWF_API_KEY")
CDS_URL = "https://cds.climate.copernicus.eu/api"

cds_api_rc = f"""url: {CDS_URL}
key: {CDS_API_KEY}
"""

with open(os.path.expanduser('~/.cdsapirc'), 'w') as f:
    f.write(cds_api_rc)

class MinPriorityQueue:
    def __init__(self):
        self._heap = []                # list of [value, coord]
        self._index_map = {}           # coord -> index in heap
        self._value_map = {}           # coord -> current value

    def _swap(self, i, j):
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        self._index_map[self._heap[i][1]] = i
        self._index_map[self._heap[j][1]] = j

    def _sift_up(self, i):
        while i > 0:
            parent = (i - 1) // 2
            if self._heap[i][0] < self._heap[parent][0]:
                self._swap(i, parent)
                i = parent
            else:
                break

    def _sift_down(self, i):
        n = len(self._heap)
        while True:
            smallest = i
            left = 2 * i + 1
            right = 2 * i + 2
            if left < n and self._heap[left][0] < self._heap[smallest][0]:
                smallest = left
            if right < n and self._heap[right][0] < self._heap[smallest][0]:
                smallest = right
            if smallest != i:
                self._swap(i, smallest)
                i = smallest
            else:
                break

    def enqueue(self, coord, value):
        """Insert a coordinate with an associated value."""
        if coord in self._index_map:
            raise ValueError(f"{coord} already in queue. Use decrease_value() instead.")
        entry = [value, coord]
        self._heap.append(entry)
        idx = len(self._heap) - 1
        self._index_map[coord] = idx
        self._value_map[coord] = value
        self._sift_up(idx)

    def dequeue_min(self):
        """Remove and return (coord, value) with the smallest value."""
        if not self._heap:
            raise IndexError("dequeue from empty queue")
        self._swap(0, len(self._heap) - 1)
        value, coord = self._heap.pop()
        del self._index_map[coord]
        del self._value_map[coord]
        if self._heap:
            self._sift_down(0)
        return coord, value

    def decrease_value(self, coord, new_value):
        """Decrease the value associated with a coordinate.
        
        Raises ValueError if new_value is not strictly less than current value.
        """
        if coord not in self._index_map:
            raise KeyError(f"{coord} not found in queue")
        old_value = self._value_map[coord]
        if new_value >= old_value:
            raise ValueError(f"New value {new_value} must be less than current value {old_value}")
        idx = self._index_map[coord]
        self._heap[idx][0] = new_value
        self._value_map[coord] = new_value
        self._sift_up(idx)

    def peek_min(self):
        """Return (coord, value) with the smallest value without removing it."""
        if not self._heap:
            raise IndexError("peek from empty queue")
        return self._heap[0][1], self._heap[0][0]

    def get_value(self, coord):
        """Return the current value associated with a coordinate."""
        return self._value_map[coord]

    def __contains__(self, coord):
        return coord in self._index_map

    def __len__(self):
        return len(self._heap)

    def __bool__(self):
        return bool(self._heap)

    def __repr__(self):
        items = [(entry[1], entry[0]) for entry in self._heap]
        return f"MinPriorityQueue({items})"

def build_adjacency_list(edges):
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    return adj


def dijkstra(edges, start, end, cost_fn):
    adj = build_adjacency_list(edges)

    if start not in adj:
        return float('inf'), []

    pq = MinPriorityQueue()
    dist = {start: 0}
    prev = {start: None}

    pq.enqueue(start, 0)

    while pq:
        current, current_dist = pq.dequeue_min()

        # Early exit
        if current == end:
            break

        # Skip stale entries
        if current_dist > dist.get(current, float('inf')):
            continue

        for neighbor, _edge_weight in adj.get(current, []):
            # Use custom cost function instead of (or in addition to) edge weight
            edge_cost = cost_fn(current, neighbor)
            new_dist = current_dist + edge_cost

            if new_dist < dist.get(neighbor, float('inf')):
                dist[neighbor] = new_dist
                prev[neighbor] = current

                if neighbor in pq:
                    pq.decrease_value(neighbor, new_dist)
                else:
                    pq.enqueue(neighbor, new_dist)

    # Reconstruct path
    if end not in dist:
        return float('inf'), []

    path = []
    node = end
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()

    return dist[end], path

def compute_ef(
        start_time, duration_hours,
        longs, lats,
        altitude_ft, aircraft_type
):
    end_time = start_time + timedelta(hours=duration_hours)

    flight_data = pd.DataFrame({
        "longitude": np.array(longs),
        "latitude": np.array(lats),
        "altitude_ft": np.array([altitude_ft for _ in range(len(longs))]),
        "time": np.array([start_time for _ in range(len(longs))])
    })

    flight = Flight(
        data=flight_data,
        aircraft_type=aircraft_type,
        flight_id="test_flight"
    )

    era5 = ERA5(
        time=(start_time, end_time),
        variables=Cocip.met_variables,
        pressure_levels = [300, 250, 225, 200],
    )
    met = era5.open_metdataset()

    era5_rad = ERA5(
        time=(start_time, end_time),
        variables=Cocip.rad_variables,
    )
    rad = era5_rad.open_metdataset()

    ps_model = PSFlight()
    cocip = Cocip(
        met=met,
        rad=rad,
        aircraft_performance=ps_model
    )

    output = cocip.eval(flight)

    result_df = output.dataframe
    return result_df['ef'].tolist()

app = FastAPI()

class FlightData(BaseModel):
    grid_density: int
    start_long: float
    start_lat: float
    end_long: float
    end_lat: float
    start_time: datetime
    duration_hours: float
    fuel_cost_per_km: float
    lambda_value: float

@app.post("/optimum_ef_route")
def main(dat: FlightData):
    # 1. Build a grid of waypoints between start and end
    n = dat.grid_density
    lons = np.linspace(dat.start_long, dat.end_long, n)
    lats = np.linspace(dat.start_lat, dat.end_lat, n)

    # Create a 2D grid of (lon, lat) nodes
    # We allow lateral deviation: for each longitude step, we have multiple latitude options
    lat_spread = abs(dat.end_lat - dat.start_lat) * 0.5  # allow deviation
    grid = {}  # (i, j) -> (lon, lat)
    coords_list = []

    for i in range(n):
        center_lat = lats[i]
        lat_options = np.linspace(center_lat - lat_spread, center_lat + lat_spread, n)
        for j in range(n):
            grid[(i, j)] = (lons[i], lat_options[j])
            coords_list.append((i, j))

    # 2. Build edges: connect each node in column i to every node in column i+1
    edges = []
    for i in range(n - 1):
        for j1 in range(n):
            for j2 in range(n):
                edges.append(((i, j1), (i + 1, j2), 1.0))  # weight unused, cost_fn handles it

    # 3. Precompute EF for every edge using pycontrails
    #    For each edge, compute contrail EF along that segment
    altitude_ft = 35000
    aircraft_type = "A320"

    ef_cache = {}
    for (a, b, _) in edges:
        lon_a, lat_a = grid[a]
        lon_b, lat_b = grid[b]
        try:
            ef_values = compute_ef(
                start_time=dat.start_time,
                duration_hours=dat.duration_hours,
                longs=[lon_a, lon_b],
                lats=[lat_a, lat_b],
                altitude_ft=altitude_ft,
                aircraft_type=aircraft_type,
            )
            ef_cache[(a, b)] = sum(ef_values) if ef_values else 0.0
        except Exception:
            ef_cache[(a, b)] = 0.0

    # 4. Define cost function: fuel cost (proportional to distance) + lambda * EF
    def cost_fn(coord_from, coord_to):
        lon_a, lat_a = grid[coord_from]
        lon_b, lat_b = grid[coord_to]
        # Approximate distance in km (haversine-lite)
        dlat = np.radians(lat_b - lat_a)
        dlon = np.radians(lon_b - lon_a)
        a_h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat_a)) * np.cos(np.radians(lat_b)) * np.sin(dlon / 2) ** 2
        dist_km = 6371 * 2 * np.arctan2(np.sqrt(a_h), np.sqrt(1 - a_h))

        fuel_cost = dat.fuel_cost_per_km * dist_km
        ef_cost = dat.lambda_value * ef_cache.get((coord_from, coord_to), 0.0)
        return fuel_cost + ef_cost

    # 5. Find start and end nodes (closest grid nodes)
    start_node = (0, n // 2)
    end_node = (n - 1, n // 2)

    # 6. Run Dijkstra
    total_cost, path = dijkstra(edges, start_node, end_node, cost_fn)

    # 7. Convert path back to lon/lat
    waypoints = [{"longitude": grid[node][0], "latitude": grid[node][1]} for node in path]

    return {
        "total_cost": total_cost,
        "waypoints": waypoints,
        "num_nodes": len(path),

    }

@app.post("/optimum_ef_route_onchain")
def main_onchain(dat: FlightData):
    """
    Same as /optimum_ef_route but returns integer-scaled values
    for Solidity/FDC compatibility.

    Scaling: total_cost is multiplied by 1e18 and truncated to int.
    """
    # === Identical logic to /optimum_ef_route ===
    n = dat.grid_density
    lons = np.linspace(dat.start_long, dat.end_long, n)
    lats = np.linspace(dat.start_lat, dat.end_lat, n)

    lat_spread = abs(dat.end_lat - dat.start_lat) * 0.5
    grid = {}
    coords_list = []

    for i in range(n):
        center_lat = lats[i]
        lat_options = np.linspace(center_lat - lat_spread, center_lat + lat_spread, n)
        for j in range(n):
            grid[(i, j)] = (lons[i], lat_options[j])
            coords_list.append((i, j))

    edges = []
    for i in range(n - 1):
        for j1 in range(n):
            for j2 in range(n):
                edges.append(((i, j1), (i + 1, j2), 1.0))

    altitude_ft = 35000
    aircraft_type = "A320"

    ef_cache = {}
    for (a, b, _) in edges:
        lon_a, lat_a = grid[a]
        lon_b, lat_b = grid[b]
        try:
            ef_values = compute_ef(
                start_time=dat.start_time,
                duration_hours=dat.duration_hours,
                longs=[lon_a, lon_b],
                lats=[lat_a, lat_b],
                altitude_ft=altitude_ft,
                aircraft_type=aircraft_type,
            )
            ef_cache[(a, b)] = sum(ef_values) if ef_values else 0.0
        except Exception:
            ef_cache[(a, b)] = 0.0

    def cost_fn(coord_from, coord_to):
        lon_a, lat_a = grid[coord_from]
        lon_b, lat_b = grid[coord_to]
        dlat = np.radians(lat_b - lat_a)
        dlon = np.radians(lon_b - lon_a)
        a_h = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat_a)) * np.cos(np.radians(lat_b)) * np.sin(dlon / 2) ** 2
        dist_km = 6371 * 2 * np.arctan2(np.sqrt(a_h), np.sqrt(1 - a_h))
        fuel_cost = dat.fuel_cost_per_km * dist_km
        ef_cost = dat.lambda_value * ef_cache.get((coord_from, coord_to), 0.0)
        return fuel_cost + ef_cost

    start_node = (0, n // 2)
    end_node = (n - 1, n // 2)

    total_cost, path = dijkstra(edges, start_node, end_node, cost_fn)

    # === Different from /optimum_ef_route: return scaled integers ===
    COST_SCALE = 10**18

    return {
        "total_cost_scaled": int(total_cost * COST_SCALE),
        "num_nodes": len(path),
        "num_waypoints": len(path),
    }
