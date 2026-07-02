"""Synthetic data for Module 4 — Liquid Cooling & Brake Thermal Management."""
from __future__ import annotations
import numpy as np
import pandas as pd
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from .physics import LiquidCoolingThermal, step_thermal

logger = setup_logger("Module4_DataGen")

def expert_control(t_cabin, t_engine, t_ambient, speed, soc, setpoint, rotor_temps):
    """Greedy low-energy + safety controller."""
    if soc < 0.05:
        return dict(pump_duty=0.0, fan_duty=0.0, vent_open=0.0, compressor_duty=0.0, disc_brake_advisory=0.0)

    cabin_error = t_cabin - setpoint
    engine_error = t_engine - 95.0
    max_rotor = max(rotor_temps)

    # 1. Liquid Cooling Loop
    pump = float(np.clip(engine_error / 20.0, 0.0, 1.0))
    fan = float(np.clip(engine_error / 15.0, 0.0, 1.0))
    
    # If rotors are extremely hot, blast the radiator fan to cool the front wheels
    if max_rotor > 400.0:
        fan = 1.0
        
    # 2. Aero RAM Air
    if speed > 15.0 and t_ambient < t_cabin:
        vent = float(np.clip(cabin_error / 5.0, 0.0, 1.0))
        if max_rotor > 400.0: vent = 1.0 # Force vents open to cool brakes
        fan *= 0.5 
    else:
        vent = 0.0

    # 3. AC Compressor
    comp = 0.0
    if cabin_error > 2.0 and t_ambient > 25.0:
        comp = float(np.clip(cabin_error / 6.0, 0.0, 1.0))

    # 4. Brake Safety Advisory (Critical Integration)
    advisory = 1.0 if max_rotor > 450.0 else 0.0

    return dict(pump_duty=pump, fan_duty=fan, vent_open=vent, compressor_duty=comp, disc_brake_advisory=advisory)

def simulate_session(rng, dt=2.0, steps=400):
    model = LiquidCoolingThermal()
    t_ambient0 = rng.uniform(-5, 42)
    irradiance0 = rng.uniform(0, 1000)
    occupants = int(rng.integers(1, 5))
    setpoint = rng.uniform(20, 24)
    t_cabin = t_ambient0 + rng.uniform(0, 18)     
    t_engine = rng.uniform(70, 105)
    speed = rng.uniform(0, 35)
    soc = rng.uniform(0.0, 1.0) 

    # Module 5 initial states
    t_tire_fl = rng.uniform(30, 60)
    t_rotor_fl = rng.uniform(30, 80)
    # Copy for fr, rl, rr with slight variance
    t_tires = [t_tire_fl + rng.normal(0, 2) for _ in range(4)]
    t_rotors = [t_rotor_fl + rng.normal(0, 5) for _ in range(4)]
    brake_intensity = rng.uniform(0, 0.8) # Simulate mountain descent

    records = []
    for k in range(steps):
        t_ambient = t_ambient0 + 3.0 * np.sin(k / 120.0) + rng.normal(0, 0.1)
        irradiance = max(0.0, irradiance0 * (0.7 + 0.3 * np.sin(k / 90.0)) + rng.normal(0, 20))

        rotor_temps = [t_rotors[i] for i in range(4)]
        ctrl = expert_control(t_cabin, t_engine, t_ambient, speed, soc, setpoint, rotor_temps)
        next_t_cabin, next_t_engine = step_thermal(
            model, t_cabin, t_engine, t_ambient, speed, soc, occupants, irradiance, dt=dt, 
            pump_duty=ctrl["pump_duty"], fan_duty=ctrl["fan_duty"], 
            vent_open=ctrl["vent_open"], compressor_duty=ctrl["compressor_duty"]
        )

        # Simulate brake heat generation & dissipation
        for i in range(4):
            # Braking generates massive rotor heat
            t_rotors[i] += brake_intensity * 15.0 * dt
            # RAM air and fan cool the rotors
            cooling = (speed * 0.5) + (ctrl["fan_duty"] * 10.0)
            t_rotors[i] -= cooling * dt
            # Heat soaks into tires
            t_tires[i] += (t_rotors[i] - t_tires[i]) * 0.05 * dt
            t_rotors[i] = max(30.0, t_rotors[i])
            t_tires[i] = max(30.0, t_tires[i])

        records.append({
            "t_cabin": t_cabin, "t_engine": t_engine, "t_ambient": t_ambient,
            "irradiance": irradiance, "occupants": occupants, "setpoint": setpoint,
            "speed": speed, "soc": soc,
            "t_tire_fl": t_tires[0], "t_tire_fr": t_tires[1], "t_tire_rl": t_tires[2], "t_tire_rr": t_tires[3],
            "t_rotor_fl": t_rotors[0], "t_rotor_fr": t_rotors[1], "t_rotor_rl": t_rotors[2], "t_rotor_rr": t_rotors[3],
            "pump_duty": ctrl["pump_duty"], "fan_duty": ctrl["fan_duty"],
            "vent_open": ctrl["vent_open"], "compressor_duty": ctrl["compressor_duty"],
            "disc_brake_advisory": ctrl["disc_brake_advisory"],
        })
        t_cabin, t_engine = next_t_cabin, next_t_engine
    return records

def generate(n_sessions: int = 200, seed: int = 42) -> pd.DataFrame:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_sessions):
        rows.extend(simulate_session(rng))
    return pd.DataFrame(rows)

def main():
    base = module_dir(__file__)
    data_dir, _ = ensure_dirs(base)
    df = generate()
    out = data_dir / "liquid_thermal_brakes.csv"
    df.to_csv(out, index=False)
    logger.info(f"Wrote {len(df):,} rows -> {out}")

if __name__ == "__main__":
    main()