"""Model definition + config for Module 1."""
from __future__ import annotations
from common.torch_helper import LSTMRegressor

FEATURE_COLS = [
    # Front (Exact math distance)
    "exact_front_gap", "rel_speed", "ego_speed", "ego_accel",
    "driver_throttle", "driver_brake", "incline_deg", "mu",
    # Rear Camera
    "rear_min_distance", "rear_rel_speed",
    # Side & Vibration
    "left_side_distance", "right_side_distance", "engine_vibration_rms"
]
TARGET_COLS = [
    "target_brake_force", "target_throttle", "target_following_distance",
    "target_lateral_shift" # NEW: Steer away from side collision (negative=left, positive=right)
]
WINDOW = 20          
HORIZON = 3          
HIDDEN = 64
LAYERS = 2

def build_model():
    return LSTMRegressor(
        n_features=len(FEATURE_COLS),
        n_outputs=len(TARGET_COLS) * HORIZON,
        hidden=HIDDEN,
        layers=LAYERS,
    )
