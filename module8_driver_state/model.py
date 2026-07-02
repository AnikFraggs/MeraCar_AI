"""Model config for Module 8 — Driver Classifier & Profile Generator."""
from __future__ import annotations
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier

FEATURE_COLS = [
    "accel_mean", "accel_std", "accel_min", "accel_max", "jerk_std", "harsh_rate",
    "steering_entropy", "reaction_latency", "eye_closure", "off_road_gaze",
    "driver_age", "eye_visibility_score", "license_verified"
]
TARGET_COL = "state"
STATE_NAMES = {0: "normal", 1: "fatigued", 2: "harsh_braking", 3: "aggressive_accel", 4: "distracted"}

INTERVENTION = {
    0: "none",
    1: "Audible fatigue alert + suggest rest stop; +30% following distance.",
    2: "Smooth braking assist; log harsh event; gentle haptic warning.",
    3: "Limit throttle response; advisory chime.",
    4: "Attention alert (visual+audio); pre-charge brakes for faster response.",
}

@dataclass
class DriverProfile:
    """Payload sent to Module 1 (Adaptive Driving) and Module 6 (Emergency)."""
    state: int
    state_name: str
    safety_modifier: float
    max_speed_mps: float
    throttle_limit: float  # 1.0 = full, 0.5 = 50% limit
    medical_emergency: bool

def build_model():
    return RandomForestClassifier(n_estimators=250, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1)