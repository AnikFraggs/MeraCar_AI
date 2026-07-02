"""Kinematic features and driver profile physics for Module 8."""
from __future__ import annotations
import numpy as np

def jerk(accel_series, dt):
    return np.gradient(np.asarray(accel_series, float), dt)

def steering_entropy(steering_series):
    d = np.diff(np.asarray(steering_series, float))
    if len(d) < 2: return 0.0
    return float(np.std(d))

def harsh_event_rate(jerk_series, threshold=8.0):
    j = np.abs(np.asarray(jerk_series, float))
    return float(np.mean(j > threshold))

def calculate_safety_modifier(state: int, eye_closure: float) -> float:
    """Returns a multiplier for Module 1 safety distances."""
    modifier = 1.0
    if state == 1: # Fatigued
        modifier = 1.3
    if state == 4: # Distracted
        modifier = 1.2
        
    # Extreme drowsiness (eyes closed >40% of time)
    if eye_closure > 0.4:
        modifier = 1.5 
        
    return modifier

def get_speed_limit(age: int, license_verified: bool, max_normal: float, max_underage: float) -> float:
    """Determines max allowed speed based on driver profile."""
    if not license_verified:
        return 0.0 # Car shouldn't move
    if age < 18:
        return max_underage
    if age >= 75: # Elderly drivers get slightly lower limit for safety
        return max_normal * 0.9
    return max_normal