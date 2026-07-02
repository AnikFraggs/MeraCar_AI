"""Train every module end-to-end."""
from __future__ import annotations
import sys
import time
import traceback
#import importlib

# CHANGE THESE STRINGS TO MATCH YOUR EXACT FOLDER NAMES AND FILE NAMES
MODULES = {
    1: ("MeraCar_AI.module1_ai_assisted_driving.train", "Intelligent Adaptive Driving"),
    2: ("MeraCar_AI.module2_suspension_braking_sunroof.train", "AI Suspension Controller"),
    3: ("MeraCar_AI.module3_energy_optimization.train", "Hybrid Energy Optimization"),
    4: ("MeraCar_AI.module4_thermal_settings.train", "Smart Cabin Thermal Management"),
    5: ("MeraCar_AI.module5_tyre_health_management.train", "Tire Health Monitoring"),
    6: ("MeraCar_AI.module6_emergency_systems.train", "Emergency AI"),
    7: ("MeraCar_AI.module7_maintainance.train", "Predictive Maintenance"),
    8: ("MeraCar_AI.module8_driver_state.train", "Driver Assistance"),
}

def run(selected):
    import importlib
    for num in selected:
        mod_path, name = MODULES[num]
        print("\n" + "=" * 70)
        print(f"  MODULE {num}: {name}")
        print("=" * 70)
        t0 = time.time()
        try:
            # If your files are in the root directory, just use mod_path
            # If they are in a parent folder, use f"parent_folder.{mod_path}"
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

if __name__ == "__main__":
    main()
