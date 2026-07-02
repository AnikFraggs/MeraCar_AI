"""Train every module end-to-end. Generates datasets as needed, trains, saves weights.

Run from the parent of adaptive_vehicle_ai/:
    python train_all.py
Optionally pass module numbers to train a subset:
    python train_all.py 1 3 7
"""
from __future__ import annotations
import sys
import time
import traceback
import importlib

MODULES = {
    1: ("myCar_AI.module1_ai_assisted_driving.train", "Intelligent Adaptive Driving"),
    2: ("myCar_AI.module2_suspension_braking_sunroof", "AI Suspension Controller"),
    3: ("myCar_AI.module3_energy_optimization.train", "Hybrid Energy Optimization"),
    4: ("myCar_AI.module4_thermal_settings.train", "Smart Cabin Thermal Management"),
    5: ("myCar_AI.module5_tyre_health_management.train", "Tire Health Monitoring"),
    6: ("myCar_AI.module6_emergency_systems.train", "Emergency AI"),
    7: ("myCar_AI.module7_maintainance.train", "Predictive Maintenance"),
    8: ("myCar_AI.module8_driver_state.train", "Driver Assistance"),
}

def run(selected):
    for num in selected:
        mod_path, name = MODULES[num]
        print("\n" + "=" * 70)
        print(f"  MODULE {num}: {name}")
        print("=" * 70)
        t0 = time.time()
        try:
            mod = importlib.import_module(mod_path)
            mod.main()
            print(f"[SUCCESS] Module {num} trained in {time.time() - t0:.1f}s")
        except Exception:
            print(f"[ERROR] Module {num} failed:")
            traceback.print_exc()

def main():
    args = [int(a) for a in sys.argv[1:] if a.isdigit()]
    selected = args or list(MODULES.keys())
    print(f"Training modules: {selected}")
    run(selected)
    print("\nAll requested modules processed.")

if __name__ == "__main__":
    main()