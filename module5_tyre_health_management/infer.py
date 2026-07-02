"""Demo: Tire pressure, brake temps, leak detection, and Module 4 integration."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import torch
from ..common.utils import get_device, module_dir, setup_logger
from ..common.torch_helper import load_checkpoint
from . import physics as ph, model as M

logger = setup_logger("Module5_Infer")

def load_all(model_dir):
    net = M.build_forecaster()
    net, _ = load_checkpoint(net, model_dir / "tire_lstm.pt", device=get_device())
    x_sc = joblib.load(model_dir / "x_scaler.joblib")
    y_sc = joblib.load(model_dir / "y_scaler.joblib")
    leak_clf = joblib.load(model_dir / "leak_rf.joblib")
    return net, x_sc, y_sc, leak_clf

def main():
    net, x_sc, y_sc, leak_clf = load_all(module_dir(__file__) / "models")

    # 1. AI BOOT SEQUENCE: Ask initial value
    logger.info("--- AEGISDRIVE TIRE AI INITIALIZED ---")
    initial_pressure = 240.0  # Simulate driver entering 240 kPa after filling air
    initial_temp = 20.0
    baseline_norm = ph.normalize_pressure(initial_pressure, initial_temp)
    logger.info(f"Baseline set to {initial_pressure} kPa at {initial_temp}C (Normalized: {baseline_norm:.1f} kPa)")

    # 2. SIMULATE DRIVING (1 week later, cold weather hit, and ran over a nail)
    logger.info("--- RUNNING WEEKLY CHECK ---")
    current_pressure = 195.0
    current_temp = -5.0  # Cold weather drops pressure naturally
    current_rotor = 85.0
    speed = 25.0
    terrain = 2 # Rocky
    
    # Normalize current pressure to reference temp to see if it's a real leak
    current_norm = ph.normalize_pressure(current_pressure, current_temp)
    p_delta = baseline_norm - current_norm
    time_elapsed = 168.0 # 1 week in hours

    # 3. Leak Classification
    X_leak = np.array([[p_delta, time_elapsed, -5.0, speed]], np.float32)
    leak_class = int(leak_clf.predict(X_leak)[0])
    
    # 4. Bernoulli Hole Size Calculation
    leak_rate_est = p_delta / time_elapsed if p_delta > 0 else 0.0
    hole_size_mm = ph.puncture_hole_size(current_pressure, leak_rate_est)

    # 5. Report to Driver & Module 4
    logger.info(f"Raw Pressure: {current_pressure} kPa ({current_temp}C)")
    logger.info(f"Normalized Pressure: {current_norm:.1f} kPa (Isolated from weather)")
    
    if leak_class == 2:
        logger.warning(f"!!! PUNCTURE DETECTED !!!")
        logger.warning(f"Estimated Hole Size: {hole_size_mm:.2f} mm")
    elif leak_class == 1:
        logger.warning(f"!!! SLOW LEAK DETECTED !!!")
    else:
        logger.info("Tire pressure is healthy.")

    # 6. Module 4 Integration Payload (Tire & Rotor Temps)
    # In a real system, this array is passed directly to module4_thermal/infer.py
    module4_payload = {
        "t_tire_fl": current_temp + 15.0, "t_tire_fr": current_temp + 16.0,
        "t_tire_rl": current_temp + 14.0, "t_tire_rr": current_temp + 14.5,
        "t_rotor_fl": current_rotor, "t_rotor_fr": current_rotor + 5.0,
        "t_rotor_rl": current_rotor - 10.0, "t_rotor_rr": current_rotor - 8.0
    }
    logger.info(f"Transmitting Wheel Thermal Array to Module 4: {module4_payload}")

if __name__ == "__main__":
    main()