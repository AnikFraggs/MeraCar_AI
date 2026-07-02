"""Demo: Energy mode selection, priority queue, consent, and emergency bypass."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from ..common.utils import module_dir, setup_logger
from ..common.physics_constants import POWERTRAIN_EV_ONLY, POWERTRAIN_HYBRID
from . import physics as ph, model as M

logger = setup_logger("Module3_Infer")

def load_all(model_dir):
    mode_clf = joblib.load(model_dir / "energy_clf.joblib")
    fuel_reg = joblib.load(model_dir / "fuel_rf.joblib")
    return mode_clf, fuel_reg

def request_user_consent(actions: dict) -> bool:
    logger.info("--- ENERGY OPTIMIZATION CONSENT REQUIRED ---")
    logger.info("AegisDrive recommends entering Eco Mode to conserve range.")
    logger.info("The following non-critical systems will be altered:")
    for sys_name, state in actions.items():
        if sys_name not in ["mode_override", "emergency_signal", "headlights"]:
            if state not in ["ENABLED", "NORMAL", "NORMAL_FREQ"]:
                logger.info(f"  - {sys_name.replace('_', ' ').title()}: {state}")
    
    logger.info("Simulating Driver clicking 'Accept Eco Mode'...")
    return True

def main():
    mode_clf, fuel_reg = load_all(module_dir(__file__) / "models")

    # Scenario: Hybrid Car, Night time, Extreme terrain, Battery LOW (but not worst case)
    soc = 0.20 # 20%
    raw_fuel_sensor = 15.0 
    speed = 15.0
    accel = 0.5
    terrain = 3 # Extreme
    eng_temp = 95.0
    is_night = True
    powertrain = POWERTRAIN_HYBRID

    # 1. Predict true fuel
    X_fuel = pd.DataFrame([[speed, accel, terrain, raw_fuel_sensor]], columns=M.FUEL_FEATURES)
    true_fuel = float(fuel_reg.predict(X_fuel)[0])
    logger.info(f"Raw Fuel: {raw_fuel_sensor:.1f}L -> AI True Fuel: {true_fuel:.1f}L | SoC: {soc*100:.1f}% | Night: {is_night}")

    # 2. Get Priority Actions
    actions, emergency = ph.get_energy_priorities(terrain, soc, true_fuel, speed, eng_temp, is_night, powertrain)

    # 3. Emergency Bypass or Consent Flow
    if emergency:
        logger.warning("!!! WORST CASE DETECTED: CRITICAL BATTERY !!!")
        logger.warning("Bypassing User Consent. Activating Module 6 Emergency AI.")
        logger.warning(f"Headlights forced {'ON' if is_night else 'OFF'}.")
    else:
        needs_consent = any("DISABLE" in str(v) or "ECO" in str(v) or "REDUCED" in str(v) for v in actions.values())
        if needs_consent:
            consent_granted = request_user_consent(actions)
            if not consent_granted:
                logger.info("Consent denied. Systems remain normal.")
                actions = {k: "NORMAL" for k in actions.keys()}

    # 4. Determine Final Mode
    final_mode: int = 0 # Initialize
    if actions.get("mode_override") is not None:
        final_mode = int(actions["mode_override"])
        logger.info(f"EV Benchmark Triggered: Overriding mode to {M.MODE_NAMES[final_mode]}")
    else:
        scenario = dict(soc=soc, irradiance=10, temp_c=20, speed=speed, accel=accel,
                        incline_rad=np.radians(2), traffic=0.2, weather_clear=False,
                        solar_w=0, power_demand_w=8000, terrain_type=terrain,
                        powertrain_type=powertrain, is_night=int(is_night))
        row = pd.DataFrame([scenario])[M.FEATURE_COLS]
        # FIX: Cast numpy.int64 to standard int for dictionary lookup
        final_mode = int(mode_clf.predict(row)[0])

    logger.info(f"Final Powertrain Mode: {M.MODE_NAMES[final_mode]}")

if __name__ == "__main__":
    main()