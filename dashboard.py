"""AegisDrive AI Data Dashboard.
Visualizes the physics and AI impact of Modules 1, 2, 4, 6, and 8.
Run: python dashboard.py
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
import sys
import os

# Add project root to path so we can import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import module physics & data generators
from module2_suspension_braking_sunroof import physics as ph2
from module4_thermal_settings import physics as ph4
from module6_emergency_systems import physics as ph6
from common.physics_constants import TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_ROCKY, TERRAIN_EXTREME

# Setup dark theme
plt.style.use('dark_background')
fig = plt.figure(figsize=(18, 10))
fig.suptitle('AegisDrive AI: Physics-Driven Vehicle Dynamics Dashboard', color='cyan', fontsize=16, fontweight='bold')

# ==========================================
# CHART 1: Module 1 & 8 - Cumulative Naive Bayes Accident Prediction
# ==========================================
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_title('Module 1 & 8: Cumulative Accident Risk vs Age (Naive Bayes)', color='white')

# 1. Generate synthetic historical data with ALL Module 8 features
np.random.seed(42)
n_samples = 2000
ages = np.random.randint(18, 80, n_samples)
eye_visibility = np.random.uniform(0.5, 1.0, n_samples) # 0.5=bad vision, 1.0=perfect
license_verified = np.random.choice([0, 1], n_samples, p=[0.1, 0.9])
reaction_latency = np.random.uniform(0.7, 2.5, n_samples) # Seconds
eye_closure = np.random.uniform(0.0, 0.5, n_samples) # PERCLOS (Drowsiness)
safety_modifier = np.random.choice([1.0, 1.3], n_samples) # 1.3 = AI Active (+30% gap)

# Physics-based risk calculation: Risk increases with age, latency, drowsiness. Decreases with vision, license, AI.
risk_score = (
    (0.4 * ages) + 
    (40 * (1 - eye_visibility)) + 
    (50 * (1 - license_verified)) + 
    (15 * (reaction_latency - 0.7)) + 
    (60 * eye_closure) - 
    (40 * (safety_modifier - 1.0)) # AI reduces risk heavily
)
# Accident occurs if risk score + noise > threshold
accidents = ((risk_score + np.random.normal(0, 10, n_samples)) > 50).astype(int)

# 2. Train Naive Bayes Classifier
X_train = np.column_stack([ages, eye_visibility, license_verified, reaction_latency, eye_closure, safety_modifier])
nb = GaussianNB()
nb.fit(X_train, accidents)

# 3. Predict probability curves across age range (18-80)
age_range = np.arange(18, 80, 1)

# Scenario A: High-Risk Driver (Unlicensed, Bad Eyes, Distracted, NO AI)
X_high_risk = np.column_stack([
    age_range, 
    np.full_like(age_range, 0.6),   # Poor eye visibility
    np.zeros_like(age_range),       # Unlicensed
    np.full_like(age_range, 2.0),   # Slow reaction (distracted)
    np.full_like(age_range, 0.4),   # High eye closure (drowsy)
    np.ones_like(age_range)         # AI Modifier 1.0 (OFF)
])
probs_high_risk = nb.predict_proba(X_high_risk)[:, 1]

# Scenario B: Normal Driver (Licensed, Good Eyes, Alert, NO AI)
X_normal = np.column_stack([
    age_range, 
    np.full_like(age_range, 0.9),   # Good eye visibility
    np.ones_like(age_range),        # Licensed
    np.full_like(age_range, 0.9),   # Normal reaction
    np.full_like(age_range, 0.05),  # Low eye closure (alert)
    np.ones_like(age_range)         # AI Modifier 1.0 (OFF)
])
probs_normal = nb.predict_proba(X_normal)[:, 1]

# Scenario C: Normal Driver + AegisDrive AI Active
X_ai = np.column_stack([
    age_range, 
    np.full_like(age_range, 0.9),   # Good eye visibility
    np.ones_like(age_range),        # Licensed
    np.full_like(age_range, 0.9),   # Normal reaction
    np.full_like(age_range, 0.05),  # Low eye closure (alert)
    np.full_like(age_range, 1.3)    # AI Modifier 1.3 (ON - +30% safety gap)
])
probs_ai = nb.predict_proba(X_ai)[:, 1]

# Plot the cumulative curves
ax1.plot(age_range, probs_high_risk * 100, color='red', label='High-Risk Driver (No AI)', linewidth=2, linestyle='--')
ax1.plot(age_range, probs_normal * 100, color='orange', label='Normal Driver (No AI)', linewidth=2)
ax1.plot(age_range, probs_ai * 100, color='lime', label='Normal Driver + AegisDrive AI', linewidth=2)

ax1.set_xlabel('Driver Age', color='white')
ax1.set_ylabel('Predicted Accident Probability (%)', color='white')
ax1.legend()
ax1.grid(alpha=0.3)

# ==========================================
# CHART 2: Module 2 - Terrain vs Suspension Stiffness (k)
# ==========================================
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_title('Module 2: Effective Stiffness (k) vs Terrain & Heat', color='white')

terrains = ['Plains', 'Desert', 'Rocky', 'Extreme']
terrain_ids = [TERRAIN_PLAINS, TERRAIN_DESERT, TERRAIN_ROCKY, TERRAIN_EXTREME]
k_cold = []
k_hot = []

for t_id in terrain_ids:
    k_base = ph2.TERRAIN_K_BASE[t_id]
    k_c, _ = ph2.compensate_k_with_heaters(k_base, engine_heat=20.0)
    k_h, _ = ph2.compensate_k_with_heaters(k_base, engine_heat=110.0)
    k_cold.append(k_c / 1000) 
    k_hot.append(k_h / 1000)

x = np.arange(len(terrains))
width = 0.35
ax2.bar(x - width/2, k_cold, width, label='Cold Start (Heater ON)', color='cyan')
ax2.bar(x + width/2, k_hot, width, label='Hot Engine (Heater OFF)', color='orange')
ax2.set_xticks(x)
ax2.set_xticklabels(terrains)
ax2.set_ylabel('Stiffness k (kN/m)', color='white')
ax2.legend()
ax2.grid(alpha=0.3, axis='y')

# ==========================================
# CHART 3: Module 4 - Thermal Management Over Time
# ==========================================
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_title('Module 4: Engine Temp - AI Cooling vs Passive', color='white')

thermal_model = ph4.LiquidCoolingThermal()
time_steps = np.arange(0, 120, 2) 
t_engine_ai = [90.0]
t_engine_passive = [90.0]
speed = 25.0 

for t in time_steps[1:]:
    next_ai, _ = ph4.step_thermal(thermal_model, 25, t_engine_ai[-1], 30, speed, 0.8, 2, 800,
                                  pump_duty=1.0, fan_duty=1.0, vent_open=1.0, compressor_duty=0.0)
    next_p, _ = ph4.step_thermal(thermal_model, 25, t_engine_passive[-1], 30, speed, 0.8, 2, 800,
                                 pump_duty=0.0, fan_duty=0.0, vent_open=0.0, compressor_duty=0.0)
    t_engine_ai.append(next_ai)
    t_engine_passive.append(next_p)

ax3.plot(time_steps, t_engine_ai, color='lime', label='AI Liquid Cooling + RAM Air', linewidth=2)
ax3.plot(time_steps, t_engine_passive, color='red', label='Passive (No Pumps/Fans)', linewidth=2)
ax3.axhline(y=105, color='white', linestyle='--', label='Overheat Threshold')
ax3.set_xlabel('Time (seconds)', color='white')
ax3.set_ylabel('Engine Temperature (°C)', color='white')
ax3.legend()
ax3.grid(alpha=0.3)

# ==========================================
# CHART 4: Module 6 - Offline Emergency Routing
# ==========================================
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_title('Module 6: Offline Dijkstra Routing & Response Times', color='white')

router = ph6.OfflineMapRouter()
start_node = "crash_node"
distances = router.dijkstra(start_node)

hospitals = ['Hospital 1', 'Hospital 2', 'Police Sta.']
nodes = ['hosp_1', 'hosp_2', 'police']
dists = [distances[n] for n in nodes]
response_times = [18.5, 12.2, 8.0] 

ax4.scatter(0, 0, color='red', s=150, zorder=5, label='Crash Site')
ax4.scatter(dists, [0, 0, 0], color=['green', 'green', 'blue'], s=100, zorder=5)

for i, (h, d, r) in enumerate(zip(hospitals, dists, response_times)):
    ax4.plot([0, d], [0, 0], color='gray', linestyle='--')
    ax4.text(d, 0.5, f'{h}\n{d}km / {r}min', color='white', ha='center', fontsize=8)

ax4.plot([0, dists[1]], [0, 0], color='lime', linewidth=3, label='Best AI Route (Hosp 2)')
ax4.set_xlim(-2, max(dists)+2)
ax4.set_ylim(-2, 2)
ax4.set_xlabel('Distance (km)', color='white')
ax4.set_yticks([])
ax4.legend(loc='upper right')
ax4.grid(alpha=0.3, axis='x')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()