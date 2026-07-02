"""Synthetic data for Module 7 — Predictive Maintenance."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from . import physics as ph

logger = setup_logger("Module7_DataGen")
FAILURE_THRESHOLD = 0.2

def simulate_unit(rng, component, dt=1.0, max_hours=2000):
    health = rng.uniform(0.95, 1.0)
    base_rate = rng.uniform(0.0003, 0.0012)
    records = []
    history = []
    hours = 0.0
    while health > FAILURE_THRESHOLD and hours < max_hours:
        stress = float(np.clip(rng.normal(0.5, 0.3), 0, 2))
        sig = ph.sensor_signature(component, health, rng)
        history.append((hours, sig, health))
        health = ph.degrade(health, stress=stress, base_rate=base_rate, dt=dt)
        hours += dt

    fail_time = hours
    for (t, sig, h) in history:
        row = dict(sig)
        row["health"] = h
        row["rul_hours"] = fail_time - t
        records.append(row)
    return records

def generate_component(component, n_units=150, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed + hash(component) % 1000)
    rows = []
    for _ in range(n_units): rows.extend(simulate_unit(rng, component))
    return pd.DataFrame(rows)

def main():
    set_seed(42)
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    for comp in ph.COMPONENTS:
        df = generate_component(comp)
        out = data_dir / f"maintenance_{comp}.csv"
        df.to_csv(out, index=False)
        logger.info(f"{comp:10s}: wrote {len(df):,} rows -> {out.name}")

if __name__ == "__main__":
    main()