"""Train one RUL regressor + anomaly detector per component."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from . import data_gen, model as M

logger = setup_logger("Module7_Train")

def load_or_generate(data_dir, comp):
    csv = data_dir / f"maintenance_{comp}.csv"
    if not csv.exists():
        logger.info(f"Dataset for {comp} missing — generating all...")
        data_gen.main()
    return pd.read_csv(csv)

def main():
    set_seed(42)
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)

    for comp in M.COMPONENTS:
        df = load_or_generate(data_dir, comp)
        feats = M.feature_cols(df)
        X, y = df[feats].to_numpy(np.float32), df[M.TARGET_COL].to_numpy(np.float32)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
        reg = M.build_regressor()
        reg.fit(X_tr, y_tr)
        pred = reg.predict(X_te)
        mae, r2 = mean_absolute_error(y_te, pred), r2_score(y_te, pred)

        healthy = df[df["health"] > 0.8][feats].to_numpy(np.float32)
        a_scaler = StandardScaler().fit(healthy)
        det = M.build_anomaly_detector().fit(a_scaler.transform(healthy))

        joblib.dump({"reg": reg, "det": det, "a_scaler": a_scaler, "features": feats},
                    model_dir / f"maintenance_{comp}.joblib")
        logger.info(f"{comp:10s}: RUL MAE={mae:7.1f} h  R2={r2:.3f}  -> saved")

if __name__ == "__main__":
    main()