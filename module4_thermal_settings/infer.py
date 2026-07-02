"""Demo: Liquid cooling, RAM air, and Brake Fade thermal control."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from ..common.utils import module_dir, setup_logger
from . import model as M

logger = setup_logger("Module4_Infer")

def load(model_dir):
    return joblib.load(model_dir / "thermal_tree.joblib")

def predict(tree, state: dict):
    row = pd.DataFrame([state])[M.FEATURE_COLS]
    preds = tree.predict(row)[0]
    return dict(zip(M.TARGET_COLS, preds))

def main():
    tree = load(module_dir(__file__) / "models")

    # Scenario 1: Mountain descent. Rotors are glowing hot (>450C).
    state_1 = dict(
        t_cabin=26.0, t_engine=95.0, t_ambient=20.0, irradiance=400, 
        occupants=2, setpoint=22.0, speed=15.0, soc=0.60,
        t_tire_fl=95.0, t_tire_fr=96.0, t_tire_rl=90.0, t_tire_rr=91.0,
        t_rotor_fl=480.0, t_rotor_fr=490.0, t_rotor_rl=450.0, t_rotor_rr=455.0
    )
    
    # Scenario 2: City driving, normal temps, but low battery (Module 3 dependency)
    state_2 = dict(
        t_cabin=35.0, t_engine=100.0, t_ambient=35.0, irradiance=1000, 
        occupants=1, setpoint=22.0, speed=5.0, soc=0.03,
        t_tire_fl=40.0, t_tire_fr=41.0, t_tire_rl=39.0, t_tire_rr=40.0,
        t_rotor_fl=60.0, t_rotor_fr=62.0, t_rotor_rl=55.0, t_rotor_rr=56.0
    )

    for i, state in enumerate([state_1, state_2]):
        logger.info(f"--- Thermal Scenario {i+1} ---")
        max_rotor = max(state["t_rotor_fl"], state["t_rotor_fr"], state["t_rotor_rl"], state["t_rotor_rr"])
        logger.info(f"State: Cabin={state['t_cabin']}C, Engine={state['t_engine']}C, Max Rotor={max_rotor}C, SoC={state['soc']*100:.0f}%")
        
        # Dependency Check
        if state["soc"] < 0.05:
            logger.warning("!!! MODULE 3 INTERLOCK: Battery < 5%. Thermal System Shut Down. !!!")
            actions = {k: 0.0 for k in M.TARGET_COLS}
        else:
            actions = predict(tree, state)

        # Report to Driver
        logger.info(f"  -> Liquid Pump Duty  : {actions['pump_duty']:.2f}")
        logger.info(f"  -> Radiator Fan Duty : {actions['fan_duty']:.2f}")
        logger.info(f"  -> Aero Vents Open   : {actions['vent_open']:.2f} (RAM Air)")
        logger.info(f"  -> AC Compressor Duty: {actions['compressor_duty']:.2f}")
        
        # Brake Advisory Check
        if round(actions["disc_brake_advisory"]) == 1:
            logger.warning("!!! BRAKE FADE RISK DETECTED !!!")
            logger.warning("Rotors exceeding 450C. Discontinue use of friction disc brakes immediately!")
            logger.warning("Switch to regenerative / lower-gear engine braking to prevent brake failure.")

if __name__ == "__main__":
    main()