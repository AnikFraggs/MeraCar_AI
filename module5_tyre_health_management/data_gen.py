"""Synthetic data for Module 5 — Tire, Brake, & Leak Monitoring."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from ..common.physics_constants import TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_ROCKY, TERRAIN_EXTREME
from . import physics as ph

logger = setup_logger("Module5_DataGen")
FAULT_NAMES = {0: "healthy", 1: "slow_leak", 2: "puncture"}

def simulate_tire(rng, steps=300, dt_hr=1 / 60):
    fault = int(rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))
    terrain_type = int(rng.choice([TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_ROCKY, TERRAIN_EXTREME]))
    
    p_nominal = 240.0
    pressure = rng.uniform(220, 250)
    initial_pressure_norm = ph.normalize_pressure(pressure, rng.uniform(10, 30))
    tread = rng.uniform(6.0, 8.0)
    ambient = rng.uniform(-5, 40)
    temp = ambient + rng.uniform(0, 5)
    t_rotor = ambient + rng.uniform(10, 30)
    
    leak_rate = 0.0
    if fault == 1: leak_rate = rng.uniform(1.0, 4.0)
    puncture_at = rng.integers(50, steps - 10) if fault == 2 else -1

    records = []
    for k in range(steps):
        speed = max(0.0, rng.normal(20, 8))
        brake_intensity = float(np.clip(rng.normal(0.2, 0.2), 0, 1))
        harshness = float(np.clip(rng.normal(0.3, 0.2), 0, 1))
        
        temp += (0.02 * speed - 0.1 * (temp - ambient)) * 1.0 + rng.normal(0, 0.2)
        t_rotor = ph.calculate_rotor_temp(speed, brake_intensity, ambient, t_rotor)

        # Gas law pressure change
        pressure = ph.pressure_from_temperature(pressure, temp - 0.5, temp)

        if fault == 1: pressure = ph.leak_pressure_drop(pressure, leak_rate, dt_hr)
        if fault == 2 and k == puncture_at:
            pressure *= rng.uniform(0.4, 0.6)
            leak_rate = rng.uniform(15, 40)
        if fault == 2 and k > puncture_at >= 0:
            pressure = ph.leak_pressure_drop(pressure, leak_rate, dt_hr)

        dist = speed * dt_hr
        tread = max(0.0, tread - ph.wear_increment(dist, pressure, harshness, terrain_type))
        
        # Normalized pressure drop for leak detection
        current_p_norm = ph.normalize_pressure(pressure, temp)
        p_delta_norm = initial_pressure_norm - current_p_norm

        next_p = pressure
        if fault in (1, 2): next_p = ph.leak_pressure_drop(pressure, leak_rate, dt_hr)
        next_tread = max(0.0, tread - ph.wear_increment(speed * dt_hr, pressure, harshness, terrain_type))

        records.append({
            "pressure_kpa": pressure + rng.normal(0, 0.5),
            "temp_c": temp, "t_rotor_c": t_rotor, "speed": speed, "ambient_c": ambient,
            "terrain_type": terrain_type, "distance_km": dist, "tread_mm": tread,
            "next_pressure_kpa": next_p, "next_tread_mm": next_tread,
            "leak_class": fault, "pressure_delta_norm": p_delta_norm, "time_elapsed_hrs": k * dt_hr
        })
    return records

def generate(n_tires: int = 300, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_tires): rows.extend(simulate_tire(rng))
    return pd.DataFrame(rows)

def main():
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    df = generate()
    out = data_dir / "tire_brake.csv"
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()