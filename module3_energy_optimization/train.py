"""Train Module 3's Powertrain Classifier & Fuel Regressor."""
from __future__ import annotations
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from . import data_gen, model as M

logger = setup_logger("Module3_Train")

def load_or_generate(data_dir):
    csv = data_dir / "energy_fuel.csv"
    if not csv.exists():
        logger.info("Dataset missing — generating...")
        data_gen.main()
    return pd.read_csv(csv)

def main():
    set_seed(42)
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)
    df = load_or_generate(data_dir)

    # 1. Train Mode Classifier
    X = df[M.FEATURE_COLS]
    y = df[M.TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    mode_clf = M.build_mode_clf()
    logger.info("Training Mode Classifier...")
    mode_clf.fit(X_tr, y_tr)
    acc = accuracy_score(y_te, mode_clf.predict(X_te))
    logger.info(f"Mode Classifier Accuracy: {acc:.3f}")

    # 2. Train Fuel Regressor (Random Forest)
    Xf = df[M.FUEL_FEATURES]
    yf = df[M.FUEL_TARGET]
    Xf_tr, Xf_te, yf_tr, yf_te = train_test_split(Xf, yf, test_size=0.2, random_state=42)
    fuel_reg = M.build_fuel_regressor()
    logger.info("Training Fuel Random Forest...")
    fuel_reg.fit(Xf_tr, yf_tr)
    mae = mean_absolute_error(yf_te, fuel_reg.predict(Xf_te))
    logger.info(f"Fuel Regressor MAE: {mae:.2f} Liters")

    # Save models
    joblib.dump(mode_clf, model_dir / "energy_clf.joblib")
    joblib.dump(fuel_reg, model_dir / "fuel_rf.joblib")
    logger.info(f"Saved models -> {model_dir}")

if __name__ == "__main__":
    main()