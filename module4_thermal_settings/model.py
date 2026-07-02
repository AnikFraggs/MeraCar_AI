"""Model config for Module 4 — Decision Tree for Liquid Cooling & Brake Thermal Control."""
from __future__ import annotations
from sklearn.tree import DecisionTreeRegressor

FEATURE_COLS = [
    # Base Thermal
    "t_cabin", "t_engine", "t_ambient", "irradiance", "occupants", 
    "setpoint", "speed", "soc",
    # Module 5 Integration: Tire & Rotor Temps (Front Left/Right, Rear Left/Right)
    "t_tire_fl", "t_tire_fr", "t_tire_rl", "t_tire_rr",
    "t_rotor_fl", "t_rotor_fr", "t_rotor_rl", "t_rotor_rr"
]
TARGET_COLS = [
    "pump_duty",        # Liquid circulation pump (0.0 to 1.0)
    "fan_duty",         # Radiator cooling fan (0.0 to 1.0)
    "vent_open",        # Sleek aerodynamic RAM air vents (0.0 to 1.0)
    "compressor_duty",  # AC Compressor (0.0 to 1.0)
    "disc_brake_advisory" # 0.0 = Normal, 1.0 = Stop using disc brakes (brake fade risk)
]

def build_model():
    # Increased max depth slightly to handle the 16 input features
    return DecisionTreeRegressor(max_depth=10, random_state=42)