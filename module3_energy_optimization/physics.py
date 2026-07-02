"""Photovoltaics, battery, propulsion, and priority queue energy models."""
from __future__ import annotations
import numpy as np
import heapq
from ..common.physics_constants import (
    G, RHO_AIR, DEFAULT_VEHICLE, TERRAIN_EXTREME, 
    POWERTRAIN_EV_ONLY, POWERTRAIN_ICE_ONLY
)

def solar_power(irradiance: float, panel_area: float = 1.5, panel_eff: float = 0.22, temp_c: float = 25.0) -> float:
    temp_derate = 1.0 - 0.004 * max(0.0, temp_c - 25.0)
    return max(0.0, irradiance * panel_area * panel_eff * temp_derate)

def propulsion_power(speed: float, accel: float, incline_rad: float, mass: float = DEFAULT_VEHICLE.mass) -> float:
    f_drag = 0.5 * RHO_AIR * DEFAULT_VEHICLE.cd * DEFAULT_VEHICLE.frontal_area * speed ** 2
    f_roll = DEFAULT_VEHICLE.crr * mass * G * np.cos(incline_rad)
    f_grade = mass * G * np.sin(incline_rad)
    f_inertia = mass * accel
    return (f_drag + f_roll + f_grade + f_inertia) * speed

def true_fuel_consumption(speed: float, accel: float, terrain_type: int) -> float:
    base_burn = 5.0 + (speed * 0.2) + (max(0, accel) * 2.0)
    terrain_penalty = 1.0 + (terrain_type * 0.15)
    return base_burn * terrain_penalty

def expert_mode(soc, solar_w, power_demand_w, traffic, route_grade, weather_clear, powertrain_type):
    soc_eff = soc + 0.05 * (solar_w / 200.0)
    high_demand = power_demand_w > 35_000 or route_grade > 0.05

    if powertrain_type == POWERTRAIN_EV_ONLY:
        if soc_eff < 0.15: return 0 
        return 0
    
    if powertrain_type == POWERTRAIN_ICE_ONLY:
        return 2

    if soc_eff < 0.2 and high_demand: return 2  # Engine
    if soc_eff > 0.6 and (not high_demand or traffic > 0.6): return 0  # EV
    if soc_eff < 0.35: return 2 if high_demand else 1
    return 1  # Hybrid

def build_priority_queue() -> list:
    pq = []
    heapq.heappush(pq, (8, "rear_view_mirror_motor", "ENABLED"))
    heapq.heappush(pq, (9, "active_suspension", "NORMAL"))
    heapq.heappush(pq, (10, "engine_cooling", "NORMAL"))
    heapq.heappush(pq, (11, "tire_check_ai", "NORMAL_FREQ"))
    heapq.heappush(pq, (12, "cabin_cooling", "NORMAL"))
    heapq.heappush(pq, (13, "sunroof_ai", "ENABLED"))
    return pq

def get_energy_priorities(terrain_type: int, soc: float, fuel_level: float, speed: float, 
                          eng_temp: float, is_night: bool, powertrain_type: int) -> tuple[dict, bool]:
    """Explicitly typed to return a dictionary of actions and a boolean emergency flag."""
    actions: dict = {}
    emergency_triggered: bool = False
    
    # WORST CASE: Battery dead
    if soc < 0.05 and powertrain_type != POWERTRAIN_ICE_ONLY:
        emergency_triggered = True
        actions["emergency_signal"] = True
        actions["headlights"] = "ON" if is_night else "OFF"
        actions["active_suspension"] = "DISABLED"
        actions["sunroof_ai"] = "DISABLED"
        actions["cabin_cooling"] = "DISABLED"
        actions["mode_override"] = 0 if powertrain_type == POWERTRAIN_EV_ONLY else 2
        return actions, emergency_triggered

    # Normal Priority Queue Processing
    pq = build_priority_queue()
    energy_critical = (soc < 0.30) or (fuel_level < 15.0) or (terrain_type == TERRAIN_EXTREME)
    
    while pq:
        priority, sys_name, default_state = heapq.heappop(pq)
        
        if energy_critical:
            if sys_name == "rear_view_mirror_motor":
                actions[sys_name] = "DISABLED (movement)" if speed > 1.0 else "ENABLED"
            elif sys_name == "active_suspension": actions[sys_name] = "REDUCED_PERF"
            elif sys_name == "sunroof_ai": actions[sys_name] = "DISABLED"
            elif sys_name == "cabin_cooling": actions[sys_name] = "ECO_MODE"
            elif sys_name == "tire_check_ai": actions[sys_name] = "LOWER_FREQ"
            elif sys_name == "engine_cooling": actions[sys_name] = "REDUCED_IF_SAFE" if eng_temp < 105.0 else "NORMAL"
            else: actions[sys_name] = default_state
        else:
            actions[sys_name] = default_state

    actions["mode_override"] = 0 if (soc > 0.40 and fuel_level < 40.0 and powertrain_type != POWERTRAIN_ICE_ONLY) else None
    
    return actions, emergency_triggered