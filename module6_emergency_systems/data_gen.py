"""Synthetic data for Module 6 — Emergency AI & POI Routing."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from ..common.physics_constants import DEFAULT_VEHICLE
from . import model as M,physics as ph

logger = setup_logger("Module6_DataGen")

def sample(rng, label):
    rec = {
        "long_accel_g": rng.normal(0, 0.15), "lat_accel_g": rng.normal(0, 0.2),
        "vert_accel_g": rng.normal(1.0, 0.05), "roll_deg": rng.normal(0, 3),
        "pitch_deg": rng.normal(0, 3), "speed": rng.uniform(0, 33),
        "rpm": rng.uniform(800, 4000), "steering_var": rng.uniform(0.02, 0.3),
        "fuel_pct": rng.uniform(15, 100), "time_since_input_s": rng.uniform(0, 5),
        "engine_temp_c": rng.uniform(80, 105), "tire_pressure_kpa": rng.uniform(200, 250),
        "alarm_armed": 0, "soc": rng.uniform(0.2, 1.0)
    }
    if label == 1: rec["long_accel_g"] = rng.uniform(-12, -5)
    elif label == 2: rec["lat_accel_g"] = rng.uniform(0.8, 1.6)
    elif label == 3: rec["rpm"], rec["engine_temp_c"] = rng.uniform(0, 400), rng.uniform(115, 140)
    elif label == 4: rec["time_since_input_s"] = rng.uniform(15, 60)
    elif label == 5: rec["fuel_pct"], rec["soc"] = rng.uniform(0, 2), rng.uniform(0, 0.05)
    elif label == 6: rec["alarm_armed"], rec["speed"], rec["steering_var"] = 1, rng.uniform(10, 30), rng.uniform(0.3, 0.8)
    elif label == 7: rec["tire_pressure_kpa"], rec["speed"] = rng.uniform(50, 100), rng.uniform(20, 33)
    return rec

def generate_incidents(n_per_class: int = 80000, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    weights = {0: 4, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1}
    for label, w in weights.items():
        for _ in range(n_per_class * w):
            r = sample(rng, label)
            r["incident"] = label
            rows.append(r)
    return pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)

def generate_poi_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generates route data for the XGBoost POI Scorer."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        dist = rng.uniform(2, 20)
        traffic = rng.uniform(0, 1)
        fuel = rng.uniform(0, 1) # 0 to 1 scale
        capacity = rng.uniform(0, 1) # 0 to 1 scale (1=full)
        road_cond = rng.uniform(0, 1) # 1=perfect, 0=destroyed
        weather = rng.uniform(0, 1) # 1=blizzard, 0=clear
        
        # Ground truth physics model for response time (mins)
        avg_speed = 60.0 * (1.0 - traffic*0.5) * road_cond * (1.0 - weather*0.3)
        travel_time = (dist / max(avg_speed, 10.0)) * 60.0
        wait_time_due_to_capacity = capacity * 15.0 # If hospital full, 15 min wait
        fuel_penalty = 0.0 if fuel > 0.1 else 10.0 # If out of fuel, hard to tow
        
        response_time = travel_time + wait_time_due_to_capacity + fuel_penalty
        
        rows.append({
            "distance_km": dist, "traffic_level": traffic, "fuel_left_pct": fuel,
            "hospital_capacity": capacity, "road_condition": road_cond, "weather_score": weather,
            "estimated_response_mins": response_time
        })
    return pd.DataFrame(rows)

def main():
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    
    inc_df = generate_incidents()
    inc_df.to_csv(data_dir / "emergency.csv", index=False)
    
    poi_df = generate_poi_data()
    poi_df.to_csv(data_dir / "poi_routes.csv", index=False)
    
    logger.info(f"Wrote {len(inc_df):,} incident rows and {len(poi_df):,} POI rows -> {data_dir}")

if __name__ == "__main__":
    main()