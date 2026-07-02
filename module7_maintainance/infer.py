"""Demo: Predictive RUL, Casual Warnings, and Maintenance Tracker Loop."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from ..common.utils import module_dir, setup_logger
from ..common.physics_constants import POWERTRAIN_ICE_ONLY, MAINTENANCE_INTERVALS
from . import data_gen, model as M

logger = setup_logger("Module7_Infer")

def load(model_dir, comp):
    return joblib.load(model_dir / f"maintenance_{comp}.joblib")

def predict(bundle, sensor_row: dict):
    feats = bundle["features"]
    X = pd.DataFrame([sensor_row])[feats].to_numpy(np.float32)
    rul = float(bundle["reg"].predict(X)[0])
    Xs = bundle["a_scaler"].transform(X)
    is_anom = bundle["det"].predict(Xs)[0] == -1
    return rul, is_anom

def main():
    model_dir = module_dir(__file__) / "models"
    rng = np.random.default_rng(3)
    
    # Initialize Tracker (Simulate an ICE vehicle that has driven 42,000km and 5,200km since last PUC)
    tracker = M.MaintenanceTracker(powertrain_type=POWERTRAIN_ICE_ONLY)
    tracker.mileage_km = 42000.0
    tracker.last_radiator_change = 0.0 # Never changed
    tracker.last_pollution_check_km = 36800.0 # 5,200km ago
    
    # Simulate Module 5 flagging brake fade 4 times
    for _ in range(4): tracker.trigger_brake_fade()
    
    # Simulate low fuel (From Module 3)
    fuel_pct = 12.0 
    soc = 0.80

    logger.info("--- AEGISDRIVE MAINTENANCE REPORT ---")
    
    # 1. Casual Warnings (Fuel / EV)
    if fuel_pct < 15.0:
        logger.info("CASUAL WARNING: Fuel low. Please refuel soon.")
    if soc < 0.15:
        logger.info("CASUAL WARNING: Battery low. Please charge EV soon.")

    # 2. ML Component RUL Predictions
    logger.info("Component Health (ML Predictions):")
    for comp in M.COMPONENTS:
        bundle = load(model_dir, comp)
        recs = data_gen.simulate_unit(rng, comp)
        late_sample = recs[int(len(recs) * 0.9)] # Near failure
        rul, is_anom = predict(bundle, late_sample)
        flag = " <-- ANOMALY DETECTED" if is_anom else ""
        logger.info(f"  {comp:10s}: Predicted RUL = {rul:7.0f} h {flag}")

    # 3. Rule-Based Maintenance Alerts
    logger.info("Maintenance Tracker Alerts:")
    alerts = tracker.get_alerts()
    if not alerts:
        logger.info("  All systems serviced. No pending maintenance.")
    else:
        for alert in alerts:
            logger.warning(f"  - {alert}")
        
        # Simulate User Interaction (Confirming cleanup)
        logger.info("Simulating Driver confirming Pollution Check and Radiator Service...")
        tracker.record_service("pollution_check")
        tracker.record_service("radiator_fluid")
        
        # Re-check alerts
        logger.info("Post-Service Alerts:")
        remaining_alerts = tracker.get_alerts()
        if len(remaining_alerts) < len(alerts):
            logger.info("  Service confirmed. Pollution and Radiator alerts cleared.")
        if remaining_alerts:
            for alert in remaining_alerts:
                logger.info(f"  - {alert}")

if __name__ == "__main__":
    main()