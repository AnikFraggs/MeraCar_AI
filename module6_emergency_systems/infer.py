"""Demo: Emergency detection, XGBoost POI routing, and 20s cancellation window."""
from __future__ import annotations
import joblib
import time
import numpy as np
import pandas as pd
from ..common.utils import module_dir, setup_logger
from . import data_gen, model as M, physics as ph

logger = setup_logger("Module6_Infer")

def load_all(model_dir):
    clf = joblib.load(model_dir / "emergency_clf.joblib")
    scorer = joblib.load(model_dir / "poi_scorer.joblib")
    return clf, scorer

def assess_incident(clf, telemetry: dict):
    row = pd.DataFrame([telemetry])[M.FEATURE_COLS]
    return int(clf.predict(row)[0])

def find_best_hospital(scorer, router, start_node, telemetry, weather_score):
    """Gets route options, scores them with XGBoost, returns the best one."""
    options = router.get_route_options(start_node, weather_score)
    
    # Prepare features for XGBoost
    X = [[opt["distance_km"], opt["traffic_level"], telemetry["fuel_pct"]/100.0, 
          opt["hospital_capacity"], opt["road_condition"], opt["weather_score"]] for opt in options]
    
    # Predict response times
    preds = scorer.predict(X)
    
    for i, opt in enumerate(options):
        opt["predicted_response_mins"] = preds[i]
        
    # Sort by predicted response time (lowest is best)
    best_opt = min(options, key=lambda x: x["predicted_response_mins"])
    return best_opt

def emergency_protocol(incident: int, telemetry: dict, gps: dict, scorer):
    incident_name = M.INCIDENT_NAMES[incident]
    logger.warning(f"!!! INCIDENT DETECTED: {incident_name.upper()} !!!")
    
    dead_reckon = ph.dead_reckon_distance(telemetry["speed"], time_elapsed_s=5.0)
    
    logger.info("Initiating 20-second driver cancellation window...")
    for i in range(20, 0, -1):
        logger.info(f"Dispatching in {i}s... (Press Cancel on Dashboard to abort)")
        time.sleep(1)
        
    logger.warning("No cancellation received. Proceeding with Emergency Dispatch.")
    
    # Map Routing & POI Scoring
    router = ph.OfflineMapRouter()
    start_node = router.gps_to_graph_node(gps["lat"], gps["lon"])
    weather = 0.8 # Simulate bad weather
    
    if incident in [1, 2, 4]: # Crash, Rollover, Unconscious -> Hospital
        best_hospital = find_best_hospital(scorer, router, start_node, telemetry, weather)
        logger.info(f"XGBoost POI Analysis: Selected {best_hospital['hospital_id']} (Est. Response: {best_hospital['predicted_response_mins']:.1f} mins)")
        logger.info(f"  -> Distance: {best_hospital['distance_km']}km, Traffic: {best_hospital['traffic_level']}, Capacity: {best_hospital['hospital_capacity']*100:.0f}% full")
    else:
        best_hospital = {"hospital_id": "police_station", "predicted_response_mins": 10.0}
        logger.info("Incident type routed to nearest Police Station.")
        
    payload = M.build_emergency_message(incident, telemetry, gps, best_hospital, dead_reckon)
    logger.info(f"Transmitting Payload:\n{M.to_json(payload)}")

def main():
    clf, scorer = load_all(module_dir(__file__) / "models")
    rng = np.random.default_rng(5)
    
    # Simulate a Crash (Label 1)
    tele = data_gen.sample(rng, label=1)
    tele.pop("incident", None)
    tele["soc"] = 0.45                 
    tele["tire_pressure_kpa"] = 230.0  
    
    gps = {"lat": 12.97160, "lon": 77.59456, "heading_deg": 215.0}
    
    incident = assess_incident(clf, tele)
    
    if incident != 0:
        emergency_protocol(incident, tele, gps, scorer)
    else:
        logger.info("Telemetry normal. No action required.")

if __name__ == "__main__":
    main()