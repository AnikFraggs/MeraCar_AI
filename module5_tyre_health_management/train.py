"""Train Module 5's tyre LSTM + Leak Random Forest."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from ..common.utils import set_seed, get_device, module_dir, ensure_dirs, setup_logger
from ..common.torch_helper import make_loaders, train_regression, save_checkpoint
from . import data_gen, model as M

logger = setup_logger("Module5_Train")

def load_or_generate(data_dir):
    csv = data_dir / "tire_brake.csv"
    if not csv.exists():
        logger.info("Dataset missing — generating...")
        data_gen.main()
    return pd.read_csv(csv)

def main(epochs: int = 25):
    set_seed(42)
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)
    df = load_or_generate(data_dir)

    # 1. LSTM Forecaster
    X = df[M.FEATURE_COLS].to_numpy(np.float32)
    y = df[M.TARGET_COLS].to_numpy(np.float32)
    x_scaler = StandardScaler().fit(X)
    y_scaler = StandardScaler().fit(y)
    Xs = x_scaler.transform(X).astype(np.float32)
    ys = y_scaler.transform(y).astype(np.float32)

    train_loader, val_loader = make_loaders(Xs, ys, window=M.WINDOW, batch_size=128)
    net = M.build_forecaster()
    logger.info(f"Training LSTM on {get_device()} ...")
    net, _ = train_regression(net, train_loader, val_loader, epochs=epochs, device=get_device(), clip_grad=1.0, patience=5)
    save_checkpoint(net, model_dir / "tire_lstm.pt")
    joblib.dump(x_scaler, model_dir / "x_scaler.joblib")
    joblib.dump(y_scaler, model_dir / "y_scaler.joblib")

    # 2. Leak Classifier (Random Forest)
    X_leak = df[M.LEAK_FEATURES].to_numpy(np.float32)
    y_leak = df[M.LEAK_TARGETS].to_numpy(np.int32)
    leak_clf = M.build_leak_classifier()
    logger.info("Training Leak Random Forest...")
    leak_clf.fit(X_leak, y_leak)
    joblib.dump(leak_clf, model_dir / "leak_rf.joblib")
    logger.info(f"Saved models -> {model_dir}")

if __name__ == "__main__":
    main()