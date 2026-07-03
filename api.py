"""AegisDrive FastAPI Backend - Feeds live data to the React Dashboard."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import random
import math
import time
import os
import sys

# Fix import paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from MeraCar_AI.module2_suspension import physics as ph2
from MeraCar_AI.module4_thermal import physics as ph4
from MeraCar_AI.module6_emergency import physics as ph6
from MeraCar_AI.common.physics_constants import TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_ROCKY, TERRAIN_EXTREME

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/api/telemetry")
def get_telemetry():
    t = time.time() % 100
    speed = 20 + 10 * math.sin(t * 0.5)
    eng_heat = 30 + 40 * math.sin(t * 0.2)
    is_drowsy = math.sin(t * 0.5) > 0.5
    
    # Module 2 Physics
    terrain_id = TERRAIN_ROCKY
    k_base = ph2.TERRAIN_K_BASE[terrain_id]
    k_eff, heater = ph2.compensate_k_with_heaters(k_base, eng_heat)
    
    # Module 4 Physics
    vent_open = 1.0 if speed > 15 else 0.2
    
    # Module 5 Physics
    tire_pressure = 180 + 40 * math.sin(t) # Fluctuates
    tire_alert = tire_pressure < 150
    
    # Module 6 Physics
    router = ph6.OfflineMapRouter()
    distances = router.dijkstra("crash_node")
    
    # Module 3 & 7 Physics
    fuel_pct = 15 + 10 * math.sin(t * 0.1)
    low_fuel = fuel_pct < 20
    emergency = fuel_pct < 12 or speed > 35 or tire_alert

    return {
        "module1_8": {
            "speed": speed,
            "is_drowsy": is_drowsy,
            "safety_modifier": 1.3 if is_drowsy else 1.0,
            "driver_age": 45,
            "reaction_latency": 2.1 if is_drowsy else 0.9,
            "eye_closure": 0.5 if is_drowsy else 0.05,
            "steering_entropy": 0.8 if is_drowsy else 0.2
        },
        "module2": {
            "terrain": "Rocky",
            "eng_heat": eng_heat,
            "k_eff": k_eff,
            "heater_on": eng_heat < 60
        },
        "module3_7": {
            "fuel_pct": fuel_pct,
            "low_fuel": low_fuel,
            "emergency": emergency
        },
        "module4": {
            "speed": speed,
            "vent_open": vent_open,
            "fan_speed_rpm": 1000 + speed * 100
        },
        "module5": {
            "tire_pressure": tire_pressure,
            "alert": tire_alert
        },
        "module6": {
            "hosp1_dist": distances.get("hosp_1", 0),
            "hosp2_dist": distances.get("hosp_2", 0),
            "police_dist": distances.get("police", 0),
            "best_route": "Hospital 2"
        }
    }