"""Liquid cooling, RAM air, and thermal physics for Module 4."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..common.physics_constants import STEFAN_BOLTZMANN, KELVIN

@dataclass
class LiquidCoolingThermal:
    # Cabin properties
    cabin_air_mass: float = 3.0         
    cabin_cp: float = 1005.0            
    cabin_ua: float = 60.0              
    
    # Engine/Liquid Loop properties
    engine_mass: float = 150.0          # Engine block mass [kg]
    engine_cp: float = 500.0            # Specific heat of metal [J/(kg*K)]
    engine_base_heat: float = 15000.0   # Waste heat generated at idle/cruise [W]
    radiator_capacity: float = 8000.0   # Max heat dissipation of liquid radiator [W]
    
    @property
    def cabin_capacitance(self) -> float:
        return self.cabin_air_mass * self.cabin_cp
        
    @property
    def engine_capacitance(self) -> float:
        return self.engine_mass * self.engine_cp

def step_thermal(model: LiquidCoolingThermal, t_cabin: float, t_engine: float, t_ambient: float, 
                 speed: float, soc: float, occupants: int, irradiance: float,
                 pump_duty: float, fan_duty: float, vent_open: float, compressor_duty: float, dt: float = 1.0):
    """Advances thermal states by dt seconds. Returns (next_t_cabin, next_t_engine)."""
    
    # MODULE 3 DEPENDENCY: If battery is critically dead, system shuts down.
    if soc < 0.05:
        # No power for pumps, fans, or compressors. Only natural convection remains.
        pump_duty, fan_duty, vent_open, compressor_duty = 0.0, 0.0, 0.0, 0.0

    # --- ENGINE & LIQUID COOLING LOOP ---
    q_engine_gen = model.engine_base_heat
    
    # Radiator dissipates heat via liquid pump and fan. 
    # RAM AIR effect: speed naturally pushes air through radiator, scaling with speed^2
    ram_air = (speed ** 2) * 5.0 
    q_radiator = model.radiator_capacity * pump_duty * (fan_duty * 2000.0 + ram_air) / 2000.0
    q_radiator = min(q_radiator, q_engine_gen) # Can't remove more heat than generated
    
    # --- CABIN THERMAL ---
    # Heat transfers from engine to cabin (simplified conduction)
    q_eng_to_cabin = 15.0 * (t_engine - t_cabin)
    
    q_shell = model.cabin_ua * (t_ambient - t_cabin)
    q_solar = irradiance * 2.5 * 0.6  # Glass area * transmittance
    q_people = 100.0 * occupants
    
    # AC Compressor (only affects cabin)
    q_ac = -4000.0 * compressor_duty
    
    # Aerodynamic Vents (Sleek openings). Free cooling at speed.
    q_vent = -vent_open * (speed * 10.0) * max(0.0, t_cabin - t_ambient)

    q_net_cabin = q_shell + q_solar + q_people + q_ac + q_vent + q_eng_to_cabin
    q_net_engine = q_engine_gen - q_radiator - q_eng_to_cabin
    
    next_t_cabin = t_cabin + (q_net_cabin / model.cabin_capacitance) * dt
    next_t_engine = t_engine + (q_net_engine / model.engine_capacitance) * dt
    
    return next_t_cabin, next_t_engine

def thermal_energy_use(soc: float, pump_duty: float, fan_duty: float, compressor_duty: float, dt: float = 1.0) -> float:
    """Calculates electrical energy used [J] by the thermal system."""
    if soc < 0.05: return 0.0
    p_pump = 150.0 * pump_duty      # Liquid pump draws ~150W max
    p_fan = 200.0 * fan_duty        # Radiator fan draws ~200W max
    p_comp = 1500.0 * compressor_duty 
    return (p_pump + p_fan + p_comp) * dt