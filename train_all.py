"""Train every module end-to-end."""
from __future__ import annotations
import sys
import os
import time
import traceback
import importlib

# Get the directory of this script and add it to Python Path
# This fixes the "attempted relative import beyond top-level package" error
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

# MATCH THESE EXACTLY TO YOUR FOLDER NAMES
MODULES = {
    1: ("module1_ai_assisted_driving.train", "Intelligent Adaptive Driving"),
    2: ("module2_suspension_braking_sunroof.train", "AI Suspension Controller"),
    3: ("module3_energy_optimization.train", "Hybrid Energy Optimization"),
    4: ("module4_thermal_settings.train", "Smart Cabin Thermal Management"),
    5: ("module5_tyre_health_management.train", "Tire Health Monitoring"),
    6: ("module6_emergency_systems.train", "Emergency AI"),
    7: ("module7_maintainance.train", "Predictive Maintenance"),
    8: ("module8_driver_state.train", "Driver Assistance"),
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
            # Check if main() exists before calling it
            if hasattr(mod, 'main'):
                mod.main()
                print(f"[SUCCESS] Module {num} trained in {time.time() - t0:.1f}s")
            else:
                print(f"[ERROR] Module {num} train.py does not have a main() function!")
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
