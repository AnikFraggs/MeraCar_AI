"""Physical constants and default vehicle parameters shared across all 8 modules."""
from dataclasses import dataclass

# Universal
G = 9.81                 # gravitational acceleration [m/s^2]
RHO_AIR = 1.225          # air density at sea level [kg/m^3]
STEFAN_BOLTZMANN = 5.670374419e-8  # [W/m^2/K^4]
R_GAS = 287.058          # specific gas constant for dry air [J/(kg*K)]
KELVIN = 273.15          # 0 C in Kelvin

# Thermodynamics (For Module 4 - Smart Cabin Thermal Management)
CP_AIR = 1005.0          # Specific heat capacity of air [J/(kg*K)]
CV_AIR = 718.0           # Specific heat capacity at constant volume [J/(kg*K)]
R_IDEAL_GAS = 8.314      # Universal gas constant [J/(mol*K)]
MOLAR_MASS_AIR = 0.02896 # kg/mol
DEW_POINT_CONST_A = 17.27
DEW_POINT_CONST_B = 237.7 # Celsius

# Tire Mechanics (For Module 5 - Bernoulli Puncture Physics)
TIRE_VOLUME_SUV = 0.05   # Approx volume of a standard SUV tire in m^3
TIRE_TEMP_REF = 20.0     # Reference temp for tire pressure normalization (Celsius)

# Standardized Terrain Types (Used by Modules 2, 3, 4, 7)
TERRAIN_PLAINS = 0
TERRAIN_DESERT = 1
TERRAIN_ROCKY = 2
TERRAIN_EXTREME = 3
# Maintenance Intervals (For Module 7)
MAINTENANCE_INTERVALS = {
    "radiator_fluid_km": 40000,      # Radiator coolant change
    "brake_pad_min_mm": 3.0,         # Min pad thickness before change
    "tire_tread_min_mm": 2.0,        # Min tread depth before change
    "pollution_check_km": 5000,      # PUC check interval for ICE/Hybrid
    "pollution_check_months": 6      # PUC check time limit
}
# Powertrain Types (For Module 3 Energy Management)
POWERTRAIN_EV_ONLY = 0
POWERTRAIN_ICE_ONLY = 1 # Internal Combustion Engine Only
POWERTRAIN_HYBRID = 2
MAX_SPEED_UNDERAGE_MPS = 22.0  # ~80 km/h limit for new/underage drivers
MAX_SPEED_NORMAL_MPS = 33.0    # ~120 km/h standard highway limit
DROWSY_DISTANCE_MODIFIER = 1.3 # +30% following distance
@dataclass
class VehicleParams:
    mass: float = 1600.0
    sprung_mass: float = 400.0
    unsprung_mass: float = 40.0
    frontal_area: float = 2.2
    cd: float = 0.29
    crr: float = 0.012
    wheel_radius: float = 0.32
    max_brake_force: float = 12000.0
    max_engine_force: float = 4500.0
    drivetrain_eff: float = 0.90
    
    # 360-Degree Dimensions (For Module 1 & 6)
    length: float = 4.5        # [m] Bumper to bumper
    width: float = 1.8         # [m] Chassis width excluding mirrors
    mirror_offset: float = 0.2 # [m] How far mirrors stick out on EACH side
    track_width: float = 1.6   # [m] Distance between left and right wheels
    center_of_gravity_height: float = 0.55 # [m] Hierarchy for rollover physics (Module 6)
    
    # Lateral Vibration Benchmark (For Module 1)
    vibration_baseline: float = 0.5 
    vibration_offset: float = 0.8 

# Default instances
DEFAULT_VEHICLE = VehicleParams()
HEAVY_EV = VehicleParams(mass=2200.0, max_engine_force=6000.0)

# Tyre / friction defaults
MU_DRY = 0.9      # tyre-road friction, dry asphalt
MU_WET = 0.5
MU_SNOW = 0.25

# Driver model defaults
REACTION_TIME = 1.2   # human reaction time [s]