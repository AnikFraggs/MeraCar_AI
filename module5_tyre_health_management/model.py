"""Model config for Module 5 — LSTM forecaster + Leak Classifier."""
from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier
from ..common.torch_helper import LSTMRegressor

FEATURE_COLS = [
    "pressure_kpa", "temp_c", "t_rotor_c", "speed", "ambient_c", 
    "terrain_type", "distance_km", "tread_mm"
]
TARGET_COLS = ["next_pressure_kpa", "next_tread_mm"]

# Features for the puncture/leak classifier (Tabular)
LEAK_FEATURES = ["pressure_delta_norm", "time_elapsed_hrs", "ambient_c", "speed"]
LEAK_TARGETS = "leak_class" # 0=Healthy, 1=Slow Leak, 2=Puncture

WINDOW = 20

def build_forecaster():
    return LSTMRegressor(n_features=len(FEATURE_COLS), n_outputs=len(TARGET_COLS), hidden=48, layers=2)

def build_leak_classifier():
    return RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)