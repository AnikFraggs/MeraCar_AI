"""Model config for Module 7 — ML RUL + Rule-based Maintenance Tracker."""
from __future__ import annotations
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from ..common.physics_constants import POWERTRAIN_EV_ONLY

COMPONENTS = ["brakes", "battery", "engine", "suspension", "steering"]
TARGET_COL = "rul_hours"
NON_FEATURES = {"health", "rul_hours"}

def feature_cols(df):
    return [c for c in df.columns if c not in NON_FEATURES]

def build_regressor():
    return GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42)

def build_anomaly_detector():
    return IsolationForest(n_estimators=200, contamination=0.08, random_state=42)

class MaintenanceTracker:
    """Tracks rule-based maintenance, pollution loops, and user confirmations."""
    def __init__(self, powertrain_type: int):
        self.powertrain_type = powertrain_type
        self.mileage_km = 0.0
        self.last_radiator_change = 0.0
        self.last_pollution_check_km = 0.0
        self.last_pollution_check_month = 0
        self.brake_fade_count = 0
        
    def add_mileage(self, km: float):
        self.mileage_km += km

    def record_service(self, service_type: str):
        if service_type == "radiator_fluid": self.last_radiator_change = self.mileage_km
        if service_type == "pollution_check": 
            self.last_pollution_check_km = self.mileage_km
            # In a real system, this would pull the current month from a clock
            self.last_pollution_check_month += 6 

    def trigger_brake_fade(self):
        """Called by Module 5/4 when disc_brake_advisory is active."""
        self.brake_fade_count += 1

    def get_alerts(self) -> list[str]:
        from ..common.physics_constants import MAINTENANCE_INTERVALS
        alerts = []
        
        # 1. Radiator Fluid
        if (self.mileage_km - self.last_radiator_change) > MAINTENANCE_INTERVALS["radiator_fluid_km"]:
            alerts.append("Radiator fluid change required. Please confirm after service.")
            
        # 2. Pollution Loop (ICE / Hybrid only)
        if self.powertrain_type != POWERTRAIN_EV_ONLY:
            km_since_pollution = self.mileage_km - self.last_pollution_check_km
            if km_since_pollution > MAINTENANCE_INTERVALS["pollution_check_km"]:
                alerts.append("POLLUTION CHECK (PUC) DUE. Vehicle emissions must be checked. Please confirm after cleanup.")
                
        # 3. Brake Fade Consistency (From Module 5)
        if self.brake_fade_count > 3:
            alerts.append("Consistent brake overheating detected. Rotor lubrication & brake pad change recommended.")
            
        return alerts