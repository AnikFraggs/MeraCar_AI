"""Quarter-car, transmission, and thermal physics for Module 2."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.signal import lsim
from common.physics_constants import (
    TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_ROCKY, TERRAIN_EXTREME
)
@dataclass
class QuarterCar:
    m_s: float = 400.0
    m_u: float = 40.0
    k_t: float = 200_000.0
    c_t: float = 0.0

# --- TERRAIN & THERMAL PHYSICS ---
TERRAIN_K_BASE = {
    TERRAIN_PLAINS: 120000.0,
    TERRAIN_DESERT: 90000.0,
    TERRAIN_ROCKY: 65000.0,
    TERRAIN_EXTREME: 45000.0,
}

def compensate_k_with_heaters(k_base: float, engine_heat: float, target_susp_temp: float = 60.0) -> tuple[float, float]:
    if engine_heat < target_susp_temp:
        heater_power = target_susp_temp - engine_heat
        effective_temp = target_susp_temp
    else:
        heater_power = 0.0
        effective_temp = engine_heat
        
    delta_t = effective_temp - 60.0
    if delta_t < 0:
        k_modifier = 1.0 - (0.001 * abs(delta_t))
    else:
        k_modifier = 1.0 + (0.0005 * delta_t)
        
    k_commanded = k_base * k_modifier
    return k_commanded, heater_power

def auto_gear_logic(speed: float, load_factor: float, throttle_direction: float) -> int:
    if throttle_direction < 0.0 and speed < 2.0:
        return 0 # Reverse
    
    shift_1to2 = 5.0 + (load_factor * 2.0)
    shift_2to3 = 10.0 + (load_factor * 2.0)
    shift_3to4 = 16.0 + (load_factor * 2.0)
    shift_4to5 = 22.0 + (load_factor * 2.0)
    
    if speed < shift_1to2: return 1
    if speed < shift_2to3: return 2
    if speed < shift_3to4: return 3
    if speed < shift_4to5: return 4
    if speed < 28.0: return 5
    return 6

# --- SUSPENSION SIMULATION ---
def natural_frequency(k: float, m: float) -> float:
    return np.sqrt(k / m) / (2 * np.pi)

def damping_ratio(c: float, k: float, m: float) -> float:
    return c / (2.0 * np.sqrt(k * m))

def road_profile(t: np.ndarray, roughness: float, speed: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    z = np.zeros_like(t)
    for wavelength in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        f = speed / wavelength
        amp = roughness * (wavelength ** 0.5) * 0.002
        phase = rng.uniform(0, 2 * np.pi)
        z += amp * np.sin(2 * np.pi * f * t + phase)
    return z

def _state_space(qc: QuarterCar, k_eff: float, c_eff: float):
    m_s, m_u, k_t, c_t = qc.m_s, qc.m_u, qc.k_t, qc.c_t
    A = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [-k_eff / m_s, -c_eff / m_s, k_eff / m_s, c_eff / m_s],
        [0.0, 0.0, 0.0, 1.0],
        [k_eff / m_u, c_eff / m_u, -(k_eff + k_t) / m_u, -(c_eff + c_t) / m_u],
    ])
    B = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [k_t / m_u, c_t / m_u]])
    return A, B

def simulate(qc: QuarterCar, k_eff: float, c_eff: float, *, roughness: float, speed: float, t_end: float = 4.0, dt: float = 0.002, seed: int = 0):
    t = np.arange(0, t_end, dt)
    z_r = road_profile(t, roughness, speed, seed)
    z_r_dot = np.gradient(z_r, dt)
    A, B = _state_space(qc, k_eff, c_eff)
    C = np.eye(4); D = np.zeros((4, 2))
    u = np.column_stack([z_r, z_r_dot])
    _, y, _ = lsim((A, B, C, D), U=u, T=t)
    
    z_s, z_s_dot, z_u, z_u_dot = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
    a_s = (k_eff * (z_u - z_s) + c_eff * (z_u_dot - z_s_dot)) / qc.m_s
    
    comfort_rms = float(np.sqrt(np.mean(a_s ** 2)))
    tyre_defl_rms = float(np.sqrt(np.mean((z_r - z_u) ** 2))) # Restored this metric
    
    return {
        "comfort_rms_accel": comfort_rms,
        "tyre_defl_rms": tyre_defl_rms
    }

def optimal_setting(qc: QuarterCar, terrain_type: int, speed: float, engine_heat: float, seed: int = 0):
    k_base = TERRAIN_K_BASE[terrain_type]
    k_eff, heater_power = compensate_k_with_heaters(k_base, engine_heat)
    roughness = {
        TERRAIN_PLAINS: 1.0, 
        TERRAIN_DESERT: 3.0, 
        TERRAIN_ROCKY: 5.0, 
        TERRAIN_EXTREME: 7.0
    }[terrain_type]
    
    # FIX: Initialize best with the FIRST iteration to prevent NoneType unpacking errors
    zeta_list = np.linspace(0.2, 0.8, 13)
    c_init = zeta_list[0] * 2.0 * np.sqrt(k_eff * qc.m_s)
    m_init = simulate(qc, k_eff, c_init, roughness=roughness, speed=speed, seed=seed)
    
    w_grip = 300.0 + 60.0 * speed
    best_cost = m_init["comfort_rms_accel"] + w_grip * m_init["tyre_defl_rms"]
    best = (c_init, m_init)
    
    # Iterate through the rest of the zeta values
    for zeta in zeta_list[1:]:
        c = zeta * 2.0 * np.sqrt(k_eff * qc.m_s)
        m = simulate(qc, k_eff, c, roughness=roughness, speed=speed, seed=seed)
        cost = m["comfort_rms_accel"] + w_grip * m["tyre_defl_rms"]
        
        # FIX: Safeguard against NaN values breaking the logic
        if np.isfinite(cost) and cost < best_cost:
            best_cost, best = cost, (c, m)
            
    c_eff, m = best
    return {"k_eff": k_eff, "c_eff": c_eff, "heater_power": heater_power, **m}
