"""Demo: Multi-Modal inference for Module 2."""
from __future__ import annotations
import joblib
import numpy as np
import torch
from common.utils import get_device, module_dir, setup_logger
from common.torch_helper import load_checkpoint
from . import model as M

logger = setup_logger("Module2_Infer")

def load_all_models(model_dir):
    # 1. Load MLP
    net = M.build_suspension_mlp()
    net, _ = load_checkpoint(net, model_dir / "suspension_mlp.pt", device=get_device())
    x_scaler = joblib.load(model_dir / "mlp_x_scaler.joblib")
    y_scaler = joblib.load(model_dir / "mlp_y_scaler.joblib")
    
    # 2. Load RFs
    gear_clf = joblib.load(model_dir / "gear_rf.joblib")
    sun_clf = joblib.load(model_dir / "sunroof_rf.joblib")
    return net, x_scaler, y_scaler, gear_clf, sun_clf

def main():
    base = module_dir(__file__)
    net, x_sc, y_sc, gear_clf, sun_clf = load_all_models(base / "models")

    # Simulated inputs: Cold morning, Rocky terrain, Reversing
    terrain = 2  # Rocky
    speed = 1.5  # Moving slowly
    load = 1.2
    eng_heat = 35.0 
    throttle_dir = -1.0  # Reversing

    # 1. MLP Suspension Inference
    X_mlp = np.array([[terrain, abs(speed), load, eng_heat]], np.float32)
    xb = torch.tensor(x_sc.transform(X_mlp), dtype=torch.float32)
    with torch.no_grad():
        mlp_out = y_sc.inverse_transform(net(xb).cpu().numpy())[0]
    k_eff, c_eff, heater_pwr = mlp_out

    # 2. RF Gear Inference
    X_gear = np.array([[abs(speed), load, throttle_dir]], np.float32)
    gear_pred = int(gear_clf.predict(X_gear)[0])  # Cast to int

    # 3. RF Sunroof Inference
    # [outside_temp, cabin_temp, humidity, rain, sunlight, aqi, speed, passenger_pref]
    X_sun = np.array([[15.0, 18.0, 60.0, 0, 800, 90, abs(speed), 1]], np.float32)
    sunroof_pred = int(sun_clf.predict(X_sun)[0]) # Cast to int

    logger.info(f"State: Terrain={terrain}, Speed={speed}m/s, EngHeat={eng_heat}C, Reverse Intent={throttle_dir<0}")
    logger.info(f"  -> MLP: k={k_eff:,.0f} N/m, c={c_eff:,.0f} Ns/m, Heater={heater_pwr:.1f}W (compensating cold)")
    logger.info(f"  -> RF Gear: {gear_pred} (0=Reverse, 1-6=Forward)")
    logger.info(f"  -> RF Sunroof: {'OPEN' if sunroof_pred else 'CLOSED'}")

if __name__ == "__main__":
    main()
