"""Train Module 1's LSTM. Generates data if missing, scales, trains, saves weights."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from common.utils import set_seed, get_device, module_dir, ensure_dirs, setup_logger
from common.torch_helper import make_loaders, train_regression, save_checkpoint
from . import data_gen, model as M

logger = setup_logger("Module1_Train")

def load_or_generate(data_dir):
    csv = data_dir / "driving_360.csv"
    if not csv.exists():
        logger.info("Dataset missing — generating 360-degree data...")
        data_gen.main()
    return pd.read_csv(csv)

def main(epochs: int = 30):
    set_seed(42)
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)
    df = load_or_generate(data_dir)

    X = df[M.FEATURE_COLS].to_numpy(np.float32)
    y = df[M.TARGET_COLS].to_numpy(np.float32)

    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y)
    Xs = x_scaler.transform(X).astype(np.float32)
    ys = y_scaler.transform(y).astype(np.float32)

    train_loader, val_loader = make_loaders(Xs, ys, window=M.WINDOW, horizon=M.HORIZON, batch_size=128)
    device = get_device()
    net = M.build_model()
    logger.info(f"Training 360 LSTM on {device} ...")
    
    net, hist = train_regression(net, train_loader, val_loader, epochs=epochs, device=device, clip_grad=1.0, patience=5, verbose=True)

    save_checkpoint(net, model_dir / "driving_lstm_360.pt", meta={"features": M.FEATURE_COLS, "targets": M.TARGET_COLS, "window": M.WINDOW, "horizon": M.HORIZON})
    joblib.dump(x_scaler, model_dir / "x_scaler_360.joblib")
    joblib.dump(y_scaler, model_dir / "y_scaler_360.joblib")
    logger.info(f"Saved model + scalers -> {model_dir}")

if __name__ == "__main__":
    main()
