"""Synthetic data for Module 8 — Driver Assistance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from . import physics as ph

logger = setup_logger("Module8_DataGen")

def make_window(rng, state, n=60, dt=0.1):
    t = np.arange(n) * dt
    accel = rng.normal(0, 0.5, n)
    steering = np.cumsum(rng.normal(0, 0.5, n)) * 0.1
    reaction_latency = rng.uniform(0.7, 1.1)
    eye_closure = rng.uniform(0.0, 0.1)
    off_road_gaze = rng.uniform(0.0, 0.1)
    
    # Demographics
    driver_age = int(rng.integers(16, 85))
    eye_visibility_score = float(np.clip(rng.normal(0.9, 0.1), 0.5, 1.0)) # Glasses/contacts clarity
    license_verified = int(rng.choice([0, 1], p=[0.05, 0.95]))

    if state == 1:
        reaction_latency = rng.uniform(1.6, 2.6)
        steering += np.cumsum(rng.normal(0, 1.2, n)) * 0.1
        eye_closure = rng.uniform(0.3, 0.7)
    elif state == 2:
        for _ in range(rng.integers(1, 4)):
            i = rng.integers(0, n)
            accel[i:i + 3] -= rng.uniform(6, 12)
    elif state == 3:
        for _ in range(rng.integers(1, 4)):
            i = rng.integers(0, n)
            accel[i:i + 3] += rng.uniform(5, 10)
        steering += np.cumsum(rng.normal(0, 0.9, n)) * 0.1
    elif state == 4:
        off_road_gaze = rng.uniform(0.4, 0.9)
        reaction_latency = rng.uniform(1.3, 2.0)
        steering += rng.normal(0, 0.05, n)

    j = ph.jerk(accel, dt)
    return {
        "accel_mean": float(np.mean(accel)), "accel_std": float(np.std(accel)),
        "accel_min": float(np.min(accel)), "accel_max": float(np.max(accel)),
        "jerk_std": float(np.std(j)), "harsh_rate": ph.harsh_event_rate(j),
        "steering_entropy": ph.steering_entropy(steering), "reaction_latency": reaction_latency,
        "eye_closure": eye_closure, "off_road_gaze": off_road_gaze,
        "driver_age": driver_age, "eye_visibility_score": eye_visibility_score,
        "license_verified": license_verified, "state": state,
    }

def generate(n_per_class: int = 12000, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    weights = {0: 3, 1: 1, 2: 1, 3: 1, 4: 1}
    for state, w in weights.items():
        for _ in range(n_per_class * w):
            rows.append(make_window(rng, state))
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)

def main():
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    df = generate()
    out = data_dir / "driver.csv"
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()