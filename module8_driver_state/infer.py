"""Demo: Driver classification, dynamic profile generation, and Module 1 integration."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from ..common.utils import module_dir, setup_logger
from ..common.physics_constants import MAX_SPEED_UNDERAGE_MPS, MAX_SPEED_NORMAL_MPS
from . import data_gen, model as M, physics as ph

logger = setup_logger("Module8_Infer")

def load(model_dir):
    return joblib.load(model_dir / "driver_clf.joblib")

def assess_driver(clf, features: dict) -> M.DriverProfile:
    row = pd.DataFrame([features])[M.FEATURE_COLS]
    state = int(clf.predict(row)[0])
    
    # Calculate Physics Modifiers
    safety_mod = ph.calculate_safety_modifier(state, features["eye_closure"])
    max_speed = ph.get_speed_limit(features["driver_age"], bool(features["license_verified"]), MAX_SPEED_NORMAL_MPS, MAX_SPEED_UNDERAGE_MPS)
    
    throttle_limit = 1.0
    if state == 3: throttle_limit = 0.5 # Aggressive accel -> limit throttle
    if features["driver_age"] < 18: throttle_limit = min(throttle_limit, 0.8)

    # Medical Emergency Check (Unconscious)
    medical_emergency = False
    if state == 1 and features["eye_closure"] > 0.6 and features["steering_entropy"] < 0.1:
        medical_emergency = True

    return M.DriverProfile(
        state=state, state_name=M.STATE_NAMES[state],
        safety_modifier=safety_mod, max_speed_mps=max_speed,
        throttle_limit=throttle_limit, medical_emergency=medical_emergency
    )

def main():
    clf = load(module_dir(__file__) / "models")
    rng = np.random.default_rng(8)
    
    # Scenario 1: Underage, unlicensed driver
    logger.info("--- Scenario 1: Underage / Unlicensed Driver ---")
    feat_1 = data_gen.make_window(rng, state=0)
    feat_1["driver_age"] = 16
    feat_1["license_verified"] = 0
    profile_1 = assess_driver(clf, feat_1)
    logger.info(f"State: {profile_1.state_name}")
    logger.info(f"Max Speed Allowed: {profile_1.max_speed_mps:.1f} m/s (0 if unlicensed)")
    
    # Scenario 2: Drowsy adult driver
    logger.info("--- Scenario 2: Drowsy Adult Driver ---")
    feat_2 = data_gen.make_window(rng, state=1)
    feat_2["driver_age"] = 35
    feat_2["license_verified"] = 1
    profile_2 = assess_driver(clf, feat_2)
    logger.info(f"State: {profile_2.state_name}")
    logger.info(f"Intervention: {M.INTERVENTION[profile_2.state]}")
    
    # Module 1 Integration Payload
    logger.info("Transmitting DriverProfile to Module 1 (Adaptive Driving):")
    logger.info(f"  -> Safety Distance Modifier: {profile_2.safety_modifier}x (+30%)")
    logger.info(f"  -> Max Speed Limit         : {profile_2.max_speed_mps:.1f} m/s")
    logger.info(f"  -> Throttle Limit          : {profile_2.throttle_limit * 100:.0f}%")
    
    # Scenario 3: Medical Emergency
    logger.info("--- Scenario 3: Medical Emergency (Unconscious) ---")
    feat_3 = data_gen.make_window(rng, state=1)
    feat_3["driver_age"] = 65
    feat_3["license_verified"] = 1
    feat_3["eye_closure"] = 0.8 # Eyes closed 80% of time
    feat_3["steering_entropy"] = 0.05 # No steering corrections
    profile_3 = assess_driver(clf, feat_3)
    
    if profile_3.medical_emergency:
        logger.warning("!!! MEDICAL EMERGENCY DETECTED !!!")
        logger.warning("Driver unresponsive. Triggering Module 6 Emergency AI.")
        logger.warning("Taking control of vehicle to pull over safely.")

if __name__ == "__main__":
    main()