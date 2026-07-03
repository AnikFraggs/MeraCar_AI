                                          MeraCar_AI: Physics-Driven Vehicle Dynamics & Safety Management System v1.00
MeraCar_AI is an intelligent, physics-informed vehicle control platform that optimizes safety, energy efficiency, ride comfort, thermal management, and emergency response. By bridging classical physics equations with advanced machine learning architectures, the system acts as a fully integrated digital twin capable of predicting and controlling real-world vehicle dynamics.

Table of Contents:
1>System Architecture
2>Understanding the "Early Stopping" Behavior
3>Module Breakdown & AI Models
4>Installation & Setup
5>Training the Models

<img width="1087" height="816" alt="image" src="https://github.com/user-attachments/assets/34a70d28-d32b-46f1-94a5-645f2642b9e9" /> 
System Architecture
The project is structured into a central common core and 8 specialized modules. Instead of treating AI as a "black box," AegisDrive uses Physics-Informed Neural Networks (PINNs) and hybrid ML pipelines.

The common/ folder acts as the central nervous system, containing standardized SI physics constants, the Extended Kalman Filter for sensor fusion, and shared PyTorch utilities. All modules rely on this shared physical truth to ensure that AI predictions never violate the laws of physics.

Understanding the "Early Stopping" Behavior
When running the training scripts, you will notice that the LSTM and MLP models often halt training before reaching the maximum number of epochs. This is not a bug, but an intentional MLOps feature called Early Stopping.

We implemented a strict patience=5 threshold. During training, the model evaluates its Mean Squared Error (MSE) against a validation dataset after every epoch.

If the validation loss stops improving for 5 consecutive epochs, the system automatically halts training.
Why? Because the AI has successfully converged on the optimal physics model. Continuing to train would lead to overfitting, where the model simply memorizes the synthetic data points rather than learning the actual kinematic/thermodynamic rules, rendering it useless in real-world scenarios.
Upon stopping, the system automatically restores the best-performing weights (lowest validation loss) and saves them to the models/ directory.

The accuracy of the model is very high as it works on highly standard data training set.
Module Breakdown & AI Models:

Module 1: Intelligent Adaptive Driving (360° Collision Avoidance)
Calculates a 360-degree predictive safety envelope, factoring in front gaps, rear cameras, and lateral vibration sensors.
<img width="850" height="447" alt="image" src="https://github.com/user-attachments/assets/c21201cc-88f6-4f71-9fc2-938d39687939" />

Physics Used: Kinematics, Newton's Laws, Aerodynamic Drag, Intelligent Driver Model (IDM).
AI Models:
LSTM Regressor (PyTorch): Predicts a 3-step horizon (0.3s future) for braking force, throttle, and lateral shift to avoid side collisions.
Feature Engineering: Gradient clipping and multi-step horizon flattening are used to stabilize the LSTM against sudden sensor spikes (e.g., potholes).

Module 2: AI Suspension, Gearing & Sunroof Controller
A holistic chassis controller that adjusts suspension stiffness (
k
) based on terrain, thermally compensates for spring softening using inbuilt heaters, and automatically shifts gears.

Physics Used: Hooke's Law, Quarter-Car Mass-Spring-Damper ODEs, Thermal Degradation of Steel.
AI Models:
Multi-Layer Perceptron (PyTorch): Predicts continuous effective stiffness (
k 
eff
​
 
), damping (
c 
eff
​
 
), and heater power.
Random Forest (scikit-learn): Discrete classification for automatic gear selection (including reverse) and sunroof open/closed states based on weather and cabin temp.

Module 3: Hybrid Energy Optimization & Load Shedding
Manages powertrain modes (EV/ICE/Hybrid) and implements a priority-based load shedding system to prevent battery depletion.

Physics Used: Photovoltaic power modeling, Battery State of Charge (SoC) dynamics, Rolling/Aero resistance.
AI Models:
HistGradientBoosting Classifier (scikit-learn): Determines the optimal powertrain mode.
Random Forest Regressor: De-noises the raw fuel sensor signal to predict the true remaining fuel, compensating for tank sloshing and inclines.
Algorithm: A Min-Heap Priority Queue safely disables non-critical systems (Priority 8-13) while hard-protecting Module 1 (Brakes/Steering).

Module 4: Smart Cabin & Brake Thermal Management
Controls a PC-style liquid cooling loop and aerodynamic RAM air vents. Integrates with Module 5 to monitor brake rotor temperatures and prevent brake fade.
<img width="885" height="440" alt="image" src="https://github.com/user-attachments/assets/fcb969e4-681a-4735-adb6-62d576adcb36" />

Physics Used: Thermodynamics, Stefan-Boltzmann Radiation, Fluid Dynamics (RAM Air pressure).
AI Models:
Decision Tree Regressor (scikit-learn): Chosen for its interpretability and safety. It maps engine/rotor temps to exact pump duties, fan speeds, and vent states. It physically interlocks with Module 3 (shuts down if SoC < 5%).

Module 5: Tire Health & Puncture Monitoring
Detects slow leaks and punctures independent of ambient weather changes, and tracks brake rotor thermals.

Physics Used: Gay-Lussac’s Gas Law (Pressure-Temperature normalization), Bernoulli’s Fluid Dynamics (Puncture hole size estimation).
AI Models:
LSTM Regressor (PyTorch): Forecasts future tire pressure and tread depth.
Random Forest Classifier: Classifies leak types (Healthy, Slow Leak, Puncture) based on normalized pressure deltas over time.

Module 6: Emergency AI & Offline Routing
Detects crashes, rollovers, or medical emergencies, and uses offline digital maps to route ambulances to the best equipped hospital, not just the closest.

Physics Used: Static Stability Factor (SSF) for rollover, Dead-Reckoning (Kinematic distance prediction without GPS).
AI Models:
Random Forest Classifier: Multi-class incident detection (Crash, Rollover, Theft, Unconscious, etc.).
Gradient Boosting Regressor (XGBoost equivalent): Scores hospitals based on traffic, road conditions, weather, and hospital capacity to predict the lowest estimated response time.
Algorithm: Dijkstra's Algorithm on a localized offline graph structure.

Module 7: Predictive Maintenance
Tracks component degradation and schedules rule-based maintenance loops (PUC pollution checks, radiator fluid swaps).

Physics Used: Exponential wear acceleration based on stress/heat cycles.
AI Models:
Gradient Boosting Regressor: Predicts Remaining Useful Life (RUL) in operating hours for brakes, battery, engine, suspension, and steering.
Isolation Forest: Unsupervised anomaly detection to catch sensor signatures indicative of impending failure.
Rule-Based State Machine: Tracks user-confirmed service intervals for pollution loops.

Module 8: Driver Assistance & Profiling
Profiles the driver's physical state and dynamically modifies Module 1's safety thresholds.
<img width="830" height="468" alt="image" src="https://github.com/user-attachments/assets/4ae1dcf4-8309-4147-ab57-bb755a802993" />

Physics Used: Jerk (derivative of acceleration), Steering Entropy.
AI Models:
Random Forest Classifier: Detects states (Normal, Fatigued, Harsh Braking, Aggressive, Distracted).
Naive Bayes: Used in dashboard visualization to prove cumulative accident risk reduction across age demographics when the AI's +30% safety gap is enforced.

Train Modules Individually
If you want to train specific modules, use the -m flag followed by the package path:
python -m MeraCar_AI.module1_ai_assisted_driving.train
python -m MeraCar_AI.module2_suspension_braking_sunroof.train
python -m MeraCar_AI.module3_energy_optimization.train
python -m MeraCar_AI.module4_thermal_settings.train
python -m MeraCar_AI.module5_tyre_health_management.train
python -m MeraCar_AI.module6_emergency_systems.train
python -m MeraCar_AI.module7_maintainance.train
python -m MeraCar_AI.module8_driver_state.train
Upon completion, each module will generate a models/ directory containing its trained .pt (PyTorch) and .joblib (scikit-learn) weights, ready for inference.

This is a prerelease as only trained models are prepared new version expected in a month !!
