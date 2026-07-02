"""Component degradation models for predictive maintenance."""
from __future__ import annotations
import numpy as np

COMPONENTS = ["brakes", "battery", "engine", "suspension", "steering"]

def degrade(health, *, stress, base_rate, dt):
    return float(np.clip(health - base_rate * (1.0 + stress) * dt, 0.0, 1.0))

def sensor_signature(component, health, rng):
    h = health
    if component == "brakes":
        return {
            "pad_thickness_mm": 12.0 * h + rng.normal(0, 0.2),
            "temp_c": 150 + (1 - h) * 250 + rng.normal(0, 10),
            "vibration": (1 - h) * 1.5 + rng.normal(0, 0.05),
        }
    if component == "battery":
        return {
            "capacity_pct": 100 * (0.6 + 0.4 * h) + rng.normal(0, 1),
            "internal_r_mohm": 30 + (1 - h) * 120 + rng.normal(0, 3),
            "temp_c": 25 + (1 - h) * 20 + rng.normal(0, 2),
        }
    if component == "engine":
        return {
            "oil_pressure_kpa": 350 * h + 80 + rng.normal(0, 8),
            "vibration": (1 - h) * 2.0 + rng.normal(0, 0.05),
            "temp_c": 90 + (1 - h) * 45 + rng.normal(0, 4),
        }
    if component == "suspension":
        return {
            "damping_eff": h + rng.normal(0, 0.02),
            "rebound_time_s": 0.3 + (1 - h) * 0.8 + rng.normal(0, 0.02),
            "noise": (1 - h) * 1.2 + rng.normal(0, 0.05),
        }
    if component == "steering":
        return {
            "play_deg": (1 - h) * 8 + rng.normal(0, 0.2),
            "assist_current_a": 5 + (1 - h) * 10 + rng.normal(0, 0.5),
            "vibration": (1 - h) * 0.8 + rng.normal(0, 0.03),
        }
    raise ValueError(component)