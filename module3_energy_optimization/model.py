"""Model configs for Module 3 — Powertrain Classifier + Fuel Regressor."""
from __future__ import annotations
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestRegressor

FEATURE_COLS = [
    "soc", "irradiance", "temp_c", "speed", "accel", "incline_rad",
    "traffic", "weather_clear", "solar_w", "power_demand_w", "terrain_type",
    "powertrain_type", "is_night"
]
TARGET_COL = "mode"
MODE_NAMES = {0: "EV", 1: "Hybrid", 2: "Engine"}

FUEL_FEATURES = ["speed", "accel", "terrain_type", "raw_fuel_sensor"]
FUEL_TARGET = "true_fuel_level"

def build_mode_clf():
    return HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=None,
        l2_regularization=1.0, random_state=42,
    )

def build_fuel_regressor():
    return RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42)