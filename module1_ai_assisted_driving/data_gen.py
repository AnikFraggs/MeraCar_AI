"""Synthetic data generator for Module 1 — 360-Degree Adaptive Driving."""
from __future__ import annotations
import numpy as np
import pandas as pd
from common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from common.physics_constants import VehicleParams, MU_DRY, MU_WET, MU_SNOW
from . import physics as ph

logger = setup_logger("Module1_DataGen")

def simulate_episode(rng, dt=0.1, steps=600):
    veh = VehicleParams()
    p = ph.IDMParams(v0=rng.uniform(20, 33))
    incline_deg = rng.uniform(-6, 6)
    incline = np.radians(incline_deg)
    mu = rng.choice([MU_DRY, MU_WET, MU_SNOW], p=[0.6, 0.3, 0.1])

    ego_v = rng.uniform(8, 28)
    lead_v = ego_v + rng.uniform(-5, 5)
    
    # Exact mathematical distance = bumper to bumper gap
    gap = rng.uniform(15, 60)
    # Rear state
    rear_v = ego_v - rng.uniform(-5, 5) 
    rear_gap = rng.uniform(10, 40)
    # Lateral state
    left_dist = rng.uniform(1.0, 4.0)
    right_dist = rng.uniform(1.0, 4.0)

    records = []
    lateral_benchmark = ph.lateral_safety_benchmark(veh)

    for _ in range(steps):
        lead_a = rng.normal(0, 0.6)
        if rng.random() < 0.02: lead_a = rng.uniform(-4.5, -2.0)
        lead_v = max(0.0, lead_v + lead_a * dt)
        rear_a = rng.normal(0, 0.5)
        rear_v = max(0.0, rear_v + rear_a * dt)
        rear_rel_speed = ego_v - rear_v

        dv = ego_v - lead_v
        accel_cmd = ph.idm_acceleration(ego_v, gap, dv, p)
        accel_cmd = np.clip(accel_cmd, -ph.max_decel_on_surface(mu, incline), p.a)

        throttle, brake = ph.split_command(accel_cmd, veh.mass, ego_v, incline, veh)
        
        # 360 Safety: Limit brake if rear car is too close
        brake = ph.limit_brake_for_rear_safety(brake, rear_gap, rear_rel_speed)

        follow_d = ph.safe_following_distance(ego_v, time_gap=p.T)

        # Side dynamics simulation
        # Simulate aerodynamic buffeting if side object is too close
        vibration = veh.vibration_baseline + rng.normal(0, 0.05)
        if left_dist < lateral_benchmark + 0.5:
            vibration += rng.uniform(0.5, 1.5) # Vibration spikes
        if right_dist < lateral_benchmark + 0.5:
            vibration += rng.uniform(0.5, 1.5)

        # Target Lateral Shift: Steer away from the closest unsafe side
        target_lateral_shift = 0.0
        if left_dist < lateral_benchmark:
            target_lateral_shift = 0.5 # Shift right
        elif right_dist < lateral_benchmark:
            target_lateral_shift = -0.5 # Shift left

        records.append({
            "exact_front_gap": gap,
            "rel_speed": -dv,
            "ego_speed": ego_v,
            "ego_accel": accel_cmd,
            "driver_throttle": float(np.clip(throttle + rng.normal(0, 0.08), 0, 1)),
            "driver_brake": float(np.clip(brake / veh.max_brake_force + rng.normal(0, 0.05), 0, 1)),
            "incline_deg": incline_deg,
            "mu": mu,
            "rear_min_distance": rear_gap,
            "rear_rel_speed": rear_rel_speed,
            "left_side_distance": left_dist,
            "right_side_distance": right_dist,
            "engine_vibration_rms": vibration,
            "target_brake_force": brake,
            "target_throttle": throttle,
            "target_following_distance": follow_d,
            "target_lateral_shift": target_lateral_shift,
        })

        # Advance states
        ego_v = max(0.0, ego_v + accel_cmd * dt)
        gap = max(0.5, gap + (lead_v - ego_v) * dt)
        rear_gap = max(0.5, rear_gap + (ego_v - rear_v) * dt)
        
        # Randomly drift side objects to simulate traffic
        left_dist = max(0.5, left_dist + rng.uniform(-0.2, 0.2))
        right_dist = max(0.5, right_dist + rng.uniform(-0.2, 0.2))

    return records

def generate(n_episodes: int = 250, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_episodes):
        rows.extend(simulate_episode(rng))
    return pd.DataFrame(rows)

def main():
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    df = generate()
    out = data_dir / "driving_360.csv"
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()
