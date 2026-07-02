"""Impact, rollover, dead-reckoning, and offline routing physics."""
from __future__ import annotations
import numpy as np
import heapq
from ..common.physics_constants import G

def impact_severity(delta_v, dt):
    return abs(delta_v / max(dt, 1e-3)) / G

def static_stability_factor(track_width, cg_height):
    return track_width / (2.0 * cg_height)

def rollover_risk(lateral_accel_g, roll_angle_deg, ssf):
    angle_term = np.tan(np.radians(abs(roll_angle_deg)))
    return (abs(lateral_accel_g) + angle_term) / max(ssf, 0.1)

def dead_reckon_distance(last_speed_mps: float, time_elapsed_s: float) -> float:
    return max(0.0, last_speed_mps * time_elapsed_s)

class OfflineMapRouter:
    """Simulates an offline OSM (OpenStreetMap) graph. Uses Dijkstra to find paths.
    Note: If two paths have the exact same distance, Dijkstra naturally picks the 
    first one encountered, satisfying the 'choose any' requirement."""
    def __init__(self):
        # Graph: {node: {neighbor: {"distance": km, "traffic": 0-1, "road_cond": 0-1}}}
        self.graph = {
            "crash_node": {
                "hosp_1": {"distance": 8.0, "traffic": 0.8, "road_cond": 0.9},
                "hosp_2": {"distance": 12.0, "traffic": 0.2, "road_cond": 0.95},
                "hosp_3": {"distance": 8.0, "traffic": 0.4, "road_cond": 0.4} # Same dist as Hosp1
            }
        }
        self.hospital_info = {
            "hosp_1": {"capacity": 0.9}, # 90% full
            "hosp_2": {"capacity": 0.3}, # 30% full
            "hosp_3": {"capacity": 0.5}  # 50% full
        }

    def gps_to_graph_node(self, lat: float, lon: float) -> str:
        """Snaps raw GPS coordinates to the nearest node in the offline map."""
        # In a real OSM graph, this uses KD-Trees for nearest neighbor search.
        # Here we simulate it.
        return "crash_node"

    def dijkstra(self, start: str) -> dict:
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0.0
        pq = [(0.0, start)]
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            if current_dist > distances[current_node]: continue
            for neighbor, attrs in self.graph[current_node].items():
                dist = current_dist + attrs["distance"]
                if dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = dist
                    heapq.heappush(pq, (dist, neighbor))
        return distances

    def get_route_options(self, start_node: str, weather_score: float) -> list[dict]:
        """Gathers all route metrics to be scored by the AI."""
        options = []
        distances = self.dijkstra(start_node)
        for hosp, route_attrs in self.graph[start_node].items():
            hosp_attrs = self.hospital_info[hosp]
            options.append({
                "hospital_id": hosp,
                "distance_km": distances[hosp],
                "traffic_level": route_attrs["traffic"],
                "road_condition": route_attrs["road_cond"],
                "weather_score": weather_score, # Global weather state
                "hospital_capacity": hosp_attrs["capacity"]
            })
        return options