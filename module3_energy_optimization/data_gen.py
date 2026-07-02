"""Synthetic data for Module 3 — Hybrid Energy & Fuel Optimization."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from ..common.physics_constants import POWERTRAIN_EV_ONLY, POWERTRAIN_ICE_ONLY, POWERTRAIN_HYBRID
from . import physics as ph

logger = setup_logger("Module3_DataGen")

def generate(n_samples: int = 80000, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_samples):
        terrain_type = int(rng.integers(0, 4))
        powertrain_type = int(rng.choice([POWERTRAIN_EV_ONLY, POWERTRAIN_ICE_ONLY, POWERTRAIN_HYBRID], p=[0.3, 0.3, 0.4]))
        is_night = int(rng.choice([0, 1], p=[0.7, 0.3]))
        
        soc = rng.uniform(0.05, 1.0)
        weather_clear = rng.random() < 0.6
        irradiance = rng.uniform(600, 1000) if (weather_clear and not is_night) else rng.uniform(0, 50)
        temp_c = rng.uniform(-5, 45)
        speed = rng.uniform(0, 33)
        accel = rng.normal(0, 1.0)
        incline_rad = np.radians(rng.uniform(-8, 8))
        traffic = rng.uniform(0, 1)

        solar_w = ph.solar_power(irradiance, temp_c=temp_c)
        power_demand_w = max(0.0, ph.propulsion_power(speed, accel, incline_rad))
        mode = ph.expert_mode(soc, solar_w, power_demand_w, traffic, np.tan(incline_rad), weather_clear, powertrain_type)
        
        true_fuel = rng.uniform(10, 80)
        burn_rate = ph.true_fuel_consumption(speed, accel, terrain_type)
        true_fuel_level = max(0.0, true_fuel - burn_rate)
        raw_fuel_sensor = true_fuel_level + rng.normal(0, 3.0) + (incline_rad * 10)

        rows.append({
            "soc": soc, "irradiance": irradiance, "temp_c": temp_c, "speed": speed,
            "accel": accel, "incline_rad": incline_rad, "traffic": traffic,
            "weather_clear": int(weather_clear), "solar_w": solar_w,
            "power_demand_w": power_demand_w, "terrain_type": terrain_type,
            "powertrain_type": powertrain_type, "is_night": is_night,
            "mode": mode, "raw_fuel_sensor": raw_fuel_sensor, "true_fuel_level": true_fuel_level,
        })
    return pd.DataFrame(rows)

def main():
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    df = generate()
    out = data_dir / "energy_fuel.csv"
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()