"""Load the trained Module 1 model and run a demo prediction on a fresh episode."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
import torch
from common.utils import get_device, module_dir, setup_logger
from common.torch_helper import load_checkpoint
from . import model as M

logger = setup_logger("Module1_Infer")

def load(model_dir):
    net = M.build_model()
    net, meta = load_checkpoint(net, model_dir / "driving_lstm_360.pt", device=get_device())
    x_scaler = joblib.load(model_dir / "x_scaler_360.joblib")
    y_scaler = joblib.load(model_dir / "y_scaler_360.joblib")
    return net, x_scaler, y_scaler, meta

def predict(net, x_scaler, y_scaler, window_df: pd.DataFrame):
    X = x_scaler.transform(window_df[M.FEATURE_COLS].to_numpy(np.float32))
    xb = torch.tensor(X[None], dtype=torch.float32, device=next(net.parameters()).device)
    with torch.no_grad():
        out = net(xb).cpu().numpy()
        
    n_targets = len(M.TARGET_COLS)
    horizon = M.HORIZON
    out_reshaped = out.reshape(-1, n_targets) 
    inv = y_scaler.inverse_transform(out_reshaped)
    predictions = inv.reshape(horizon, n_targets)
    return predictions

def main():
    base = module_dir(__file__)
    model_dir = base / "models"
    net, x_scaler, y_scaler, _ = load(model_dir)

    from . import data_gen
    rng = np.random.default_rng(7)
    df = pd.DataFrame(data_gen.simulate_episode(rng))
    window = df.iloc[-M.WINDOW:]
    predictions = predict(net, x_scaler, y_scaler, window)

    last = window.iloc[-1]
    logger.info(f"Demo 360 state: Front={last.exact_front_gap:.1f}m, Rear={last.rear_min_distance:.1f}m, Left={last.left_side_distance:.1f}m, Vib={last.engine_vibration_rms:.2f}")
    
    for step in range(M.HORIZON):
        logger.info(f"Pred t+{step+1}: Brake={predictions[step, 0]:.0f}N, Throttle={predictions[step, 1]:.2f}, LateralShift={predictions[step, 3]:.2f}m")

if __name__ == "__main__":
    main()
