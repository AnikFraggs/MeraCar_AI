"""Train Module 2's Multi-Modal models (MLP + 2 Random Forests)."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from ..common.utils import set_seed, get_device, module_dir, ensure_dirs, setup_logger
from ..common.torch_helper import make_loaders, train_regression, save_checkpoint
from . import data_gen, model as M

logger = setup_logger("Module2_Train")

def load_or_generate(data_dir):
    csv = data_dir / "multimodal_chassis.csv"
    if not csv.exists():
        logger.info("Dataset missing — generating...")
        data_gen.main()
    return pd.read_csv(csv)

def main(epochs: int = 200):
    set_seed(42)
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)
    df = load_or_generate(data_dir)

    # ==========================================
    # 1. Train MLP for Suspension (PyTorch)
    # ==========================================
    logger.info("Preparing Suspension MLP data...")
    X_mlp = df[M.MLP_FEATURES].to_numpy(np.float32)
    y_mlp = df[M.MLP_TARGETS].to_numpy(np.float32)
    
    x_scaler = StandardScaler().fit(X_mlp)
    y_scaler = StandardScaler().fit(y_mlp)
    Xs = x_scaler.transform(X_mlp).astype(np.float32)
    ys = y_scaler.transform(y_mlp).astype(np.float32)

    train_loader, val_loader = make_loaders(Xs, ys, batch_size=32)
    device = get_device()
    net = M.build_suspension_mlp()
    logger.info(f"Training Suspension MLP on {device}...")
    net, _ = train_regression(
        net, train_loader, val_loader, 
        epochs=epochs, lr=1e-3, device=device, 
        clip_grad=1.0, patience=10, verbose=True
    )
    save_checkpoint(net, model_dir / "suspension_mlp.pt")
    joblib.dump(x_scaler, model_dir / "mlp_x_scaler.joblib")
    joblib.dump(y_scaler, model_dir / "mlp_y_scaler.joblib")
    logger.info("Saved MLP and scalers.")

    # ==========================================
    # 2. Train Random Forest for Auto Gear
    # ==========================================
    logger.info("Training Gear Random Forest...")
    X_gear = df[M.RF_GEAR_FEATURES].to_numpy(np.float32)
    y_gear = df[M.RF_GEAR_TARGETS].to_numpy(np.int32).ravel()
    gear_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    gear_clf.fit(X_gear, y_gear)
    joblib.dump(gear_clf, model_dir / "gear_rf.joblib")
    logger.info("Saved Gear RF.")

    # ==========================================
    # 3. Train Random Forest for Sunroof
    # ==========================================
    logger.info("Training Sunroof Random Forest...")
    X_sun = df[M.RF_SUNROOF_FEATURES].to_numpy(np.float32)
    y_sun = df[M.RF_SUNROOF_TARGETS].to_numpy(np.int32).ravel()
    sun_clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    sun_clf.fit(X_sun, y_sun)
    joblib.dump(sun_clf, model_dir / "sunroof_rf.joblib")
    logger.info("Saved Sunroof RF.")

if __name__ == "__main__":
    main()
