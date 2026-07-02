"""Train Module 4's Liquid Cooling & Brake Decision Tree."""
from __future__ import annotations
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ..common.utils import module_dir, ensure_dirs, setup_logger
from . import data_gen, model as M

logger = setup_logger("Module4_Train")

def load_or_generate(data_dir):
    csv = data_dir / "liquid_thermal_brakes.csv"
    if not csv.exists():
        logger.info("Dataset missing — generating...")
        data_gen.main()
    return pd.read_csv(csv)

def main():
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)
    df = load_or_generate(data_dir)

    X = df[M.FEATURE_COLS]
    y = df[M.TARGET_COLS]
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tree = M.build_model()
    logger.info("Training Decision Tree Regressor...")
    tree.fit(X_tr, y_tr)
    
    pred = tree.predict(X_te)
    mse = mean_squared_error(y_te, pred)
    logger.info(f"Decision Tree Test MSE: {mse:.4f}")

    joblib.dump(tree, model_dir / "thermal_tree.joblib")
    logger.info(f"Saved model -> {model_dir / 'thermal_tree.joblib'}")

if __name__ == "__main__":
    main()