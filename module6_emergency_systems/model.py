"""Model config for Module 6 — Incident Classifier + POI XGBoost Scorer."""
from __future__ import annotations
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor

FEATURE_COLS = [
    "long_accel_g", "lat_accel_g", "vert_accel_g", "roll_deg", "pitch_deg",
    "speed", "rpm", "steering_var", "fuel_pct", "time_since_input_s", "engine_temp_c",
    "tire_pressure_kpa", "alarm_armed", "soc"
]
TARGET_COL = "incident"

# POI Scoring Config
POI_SCORER_FEATURES = [
    "distance_km", "traffic_level", "fuel_left_pct", "hospital_capacity", "road_condition", "weather_score"
]
POI_SCORER_TARGET = "estimated_response_mins"

INCIDENT_NAMES = {
    0: "normal", 1: "crash", 2: "rollover", 3: "engine_failure",
    4: "driver_unconscious", 5: "fuel_exhaustion", 6: "stolen", 7: "tire_failure"
}
SEVERITY = {0: "none", 1: "critical", 2: "critical", 3: "high", 4: "critical", 5: "moderate", 6: "high", 7: "moderate"}

def build_model():
    return RandomForestClassifier(n_estimators=250, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1)

def build_poi_scorer():
    """XGBoost-style regressor to estimate response time for each hospital."""
    return GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)

def build_emergency_message(incident: int, telemetry: dict, gps: dict, best_hospital: dict, dead_reckon_dist: float = 0.0):
    name = INCIDENT_NAMES[incident]
    payload = {
        "alert": name != "normal",
        "incident_type": name,
        "severity": SEVERITY[incident],
        "gps_last_known": {"lat": gps["lat"], "lon": gps["lon"]},
        "dead_reckon_offset_m": round(dead_reckon_dist, 1),
        "dispatch_target": {
            "hospital_id": best_hospital["hospital_id"],
            "estimated_response_mins": round(best_hospital["predicted_response_mins"], 1)
        },
        "last_telemetry": {
            "fuel_pct": round(telemetry.get("fuel_pct", 0), 1),
            "soc": round(telemetry.get("soc", 0), 2),
            "tire_pressure_kpa": round(telemetry.get("tire_pressure_kpa", 0), 1)
        },
        "message": _human_message(name)
    }
    return payload

def _human_message(name: str) -> str:
    table = {
        "crash": "Collision detected. Possible injuries. Dispatch responders.",
        "rollover": "Vehicle rollover. High injury risk. Dispatch responders.",
        "engine_failure": "Engine failure. Vehicle stranded in traffic.",
        "driver_unconscious": "Driver unresponsive. Welfare check required.",
        "fuel_exhaustion": "Fuel/Battery exhausted. Vehicle stopping.",
        "stolen": "Unauthorized movement detected! Tracking via dead-reckoning.",
        "tire_failure": "Catastrophic tire failure detected. High accident risk.",
        "normal": "No incident."
    }
    return table[name]

def to_json(payload) -> str:
    return json.dumps(payload, indent=2)