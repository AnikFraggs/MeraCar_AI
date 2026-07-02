"""Tyre & Brake physics: gas laws, thermal dynamics, wear, and fluid dynamics."""
from __future__ import annotations
import numpy as np
from ..common.physics_constants import G, KELVIN, TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_ROCKY, TERRAIN_EXTREME

def pressure_from_temperature(p_ref_kpa, t_ref_c, t_now_c):
    """Gay-Lussac's law (ideal gas, const V): P/T = const (absolute units)."""
    return p_ref_kpa * (t_now_c + KELVIN) / (t_ref_c + KELVIN)

def normalize_pressure(p_kpa, t_c, t_ref_c=20.0):
    """Normalizes tire pressure to a reference temp (e.g., 20C).
    This isolates true air loss from ambient temperature changes."""
    return p_kpa * (t_ref_c + KELVIN) / (t_c + KELVIN)

def rolling_resistance_coeff(pressure_kpa, crr0=0.012, p_nominal=240.0):
    return crr0 * (p_nominal / max(pressure_kpa, 50.0)) ** 0.4

def rolling_loss_power(crr, mass, speed):
    return crr * mass * G * speed

def leak_pressure_drop(pressure_kpa, leak_rate_kpa_per_hr, dt_hr):
    return max(0.0, pressure_kpa - leak_rate_kpa_per_hr * dt_hr)

def puncture_hole_size(pressure_kpa: float, leak_rate_kpa_per_hr: float, tire_volume_m3: float = 0.05) -> float:
    """Estimates puncture hole diameter [mm] using Bernoulli's fluid dynamics.
    Assumes air escapes at speed of sound for simplicity in choked flow."""
    if leak_rate_kpa_per_hr <= 0: return 0.0
    # Convert kPa/hr to m^3/s of volume loss
    p_pa = pressure_kpa * 1000.0
    delta_v_m3_s = (leak_rate_kpa_per_hr * 1000.0) / 3600.0 / p_pa * tire_volume_m3
    # Bernoulli: v = sqrt(2*P/rho). A = Q / v
    rho_air = 1.225
    v_air = np.sqrt(max(1.0, (2 * p_pa) / rho_air))
    area_m2 = delta_v_m3_s / v_air
    diameter_mm = 2.0 * np.sqrt(area_m2 / np.pi) * 1000.0
    return float(np.clip(diameter_mm, 0.0, 20.0))

def calculate_rotor_temp(speed: float, brake_intensity: float, t_ambient: float, t_rotor_prev: float, dt: float = 1.0) -> float:
    """Simulates brake rotor temperature. brake_intensity is 0.0 to 1.0."""
    # Heat generation scales with speed squared and brake force
    q_gen = brake_intensity * (speed ** 2) * 0.5
    # Cooling scales with speed (RAM air) and temp difference
    q_cool = (speed * 0.8) + 5.0
    delta_t = (q_gen - q_cool * (t_rotor_prev - t_ambient)) * dt * 0.01
    return t_rotor_prev + delta_t

def wear_increment(distance_km: float, pressure_kpa: float, harshness: float, terrain_type: int, p_nominal=240.0) -> float:
    """Tread depth loss [mm], worsened by under-inflation, harsh driving, and terrain."""
    base = distance_km / 10_000.0
    inflation_factor = (p_nominal / max(pressure_kpa, 50.0)) ** 0.5
    
    # Terrain multipliers
    terrain_wear = {TERRAIN_PLAINS: 1.0, TERRAIN_DESERT: 1.2, TERRAIN_ROCKY: 1.8, TERRAIN_EXTREME: 2.5}
    terrain_factor = terrain_wear.get(terrain_type, 1.0)
    
    return base * inflation_factor * (1.0 + 0.8 * harshness) * terrain_factor