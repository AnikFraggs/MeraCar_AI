"""Synthetic data for Module 2 — Multi-Modal Chassis Controller."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from .physics import QuarterCar, optimal_setting, auto_gear_logic

logger = setup_logger("Module2_DataGen")

def generate(n_samples: int = 80000, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_samples):
        # FIX: Cast numpy int64 to standard Python int for dictionary keys
        terrain_type = int(rng.integers(0, 4)) 
        speed = rng.uniform(0, 35)
        load_factor = rng.uniform(0.8, 1.4)
        engine_heat = rng.uniform(20, 110) 
        throttle_dir = rng.choice([-1.0, 1.0], p=[0.1, 0.9])
        
        # RF Features
        outside_temp = rng.uniform(5, 45)
        cabin_temp = rng.uniform(15, 35)
        humidity = rng.uniform(10, 90)
        rain = int(rng.choice([0, 1], p=[0.8, 0.2]))
        sunlight = rng.uniform(0, 1200) 
        aqi = rng.uniform(20, 200)
        passenger_pref = int(rng.choice([0, 1])) 
        
        qc = QuarterCar(m_s=400.0 * load_factor)
        # Pass standard int and absolute speed
        res = optimal_setting(qc, terrain_type, abs(speed), engine_heat, seed=i)
        target_gear = auto_gear_logic(abs(speed), load_factor, throttle_dir)
        
        sunroof_state = 0
        if rain == 0 and aqi < 150 and cabin_temp > 22 and sunlight < 1000:
            if passenger_pref == 1: sunroof_state = 1

        rows.append({
            "terrain_type": terrain_type, "speed": speed, "load_factor": load_factor,
            "engine_heat": engine_heat, "throttle_direction": throttle_dir,
            "outside_temp": outside_temp, "cabin_temp": cabin_temp, "humidity": humidity,
            "rain": rain, "sunlight": sunlight, "aqi": aqi, "passenger_pref": passenger_pref,
            "k_eff": res["k_eff"], "c_eff": res["c_eff"], "heater_power": res["heater_power"],
            "target_gear": target_gear, "sunroof_state": sunroof_state
        })
        if (i + 1) % 100 == 0: logger.info(f"Optimised {i + 1}/{n_samples} conditions")
    return pd.DataFrame(rows)

def main():
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    df = generate()
    out = data_dir / "multimodal_chassis.csv"
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()
