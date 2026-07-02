"""Longitudinal & Lateral driving physics for 360-degree adaptive control."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..common.physics_constants import G, RHO_AIR, VehicleParams, MU_DRY, REACTION_TIME

def aero_drag(v: float, cd: float = 0.29, area: float = 2.2) -> float:
    return 0.5 * RHO_AIR * cd * area * v * v

def rolling_resistance(mass: float, incline_rad: float, crr: float = 0.012) -> float:
    return crr * mass * G * np.cos(incline_rad)

def grade_force(mass: float, incline_rad: float) -> float:
    return mass * G * np.sin(incline_rad)

def max_decel_on_surface(mu: float, incline_rad: float = 0.0) -> float:
    return mu * G * np.cos(incline_rad) + G * np.sin(incline_rad)

def braking_distance(v: float, mu: float, incline_rad: float = 0.0,
                     reaction_time: float = REACTION_TIME) -> float:
    a = max(max_decel_on_surface(mu, incline_rad), 0.5)
    return v * reaction_time + v * v / (2.0 * a)

def safe_following_distance(v: float, time_gap: float = 1.8, d_min: float = 2.0, driver_modifier: float = 1.0) -> float:
    """Constant-time-gap safe distance [m]: d = (d_min + v * time_gap) * driver_modifier."""
    # Module 8 passes 1.3 for drowsy drivers, increasing the gap by 30%
    return (d_min + v * time_gap) * driver_modifier

def lateral_safety_benchmark(veh: VehicleParams) -> float:
    """Calculates the minimum safe distance from the centerline to a side object [m]."""
    # Half the track width + mirror offset
    return (veh.track_width / 2.0) + veh.mirror_offset

def check_vibration_threshold(vibration_rms: float, veh: VehicleParams) -> bool:
    """Returns True if lateral vibration indicates impending side collision/scrub."""
    threshold = veh.vibration_baseline + veh.vibration_offset
    return vibration_rms > threshold

def limit_brake_for_rear_safety(target_brake: float, rear_distance: float, rear_rel_speed: float) -> float:
    """If a car is tailgating us (close rear distance), we cannot brake at 100% 
    or we will be rear-ended. Limits brake force proportionally."""
    # If rear car is far away, don't limit brakes
    if rear_distance > 20.0:
        return target_brake
    
    # If rear car is approaching fast and very close, reduce braking by up to 50%
    # to give them time to react (classic ACC collision mitigation)
    severity = np.clip(rear_distance / 15.0, 0.5, 1.0)
    return target_brake * severity

@dataclass
class IDMParams:
    v0: float = 30.0
    T: float = 1.6
    a: float = 1.5
    b: float = 2.5
    delta: float = 4.0
    s0: float = 2.0

def idm_acceleration(v: float, gap: float, dv: float, p: IDMParams) -> float:
    gap = max(gap, 0.1)
    s_star = p.s0 + max(0.0, v * p.T + (v * dv) / (2.0 * np.sqrt(p.a * p.b)))
    accel = p.a * (1.0 - (v / p.v0) ** p.delta - (s_star / gap) ** 2)
    return float(accel)

def split_command(accel: float, mass: float, v: float, incline_rad: float,
                  veh: VehicleParams = VehicleParams()) -> tuple[float, float]:
    f_drag = aero_drag(v, veh.cd, veh.frontal_area)
    f_roll = rolling_resistance(mass, incline_rad, veh.crr)
    f_grade = grade_force(mass, incline_rad)
    f_required = mass * accel + f_drag + f_roll + f_grade
    
    if f_required >= 0:
        throttle = np.clip(f_required / (veh.max_engine_force * veh.drivetrain_eff), 0.0, 1.0)
        return float(throttle), 0.0
    brake = np.clip(-f_required, 0.0, veh.max_brake_force)
    return 0.0, float(brake)