"""Train Module 8's driver-state classifier."""
from __future__ import annotations
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from . import data_gen, model as M

logger = setup_logger("Module8_Train")

def load_or_generate(data_dir):
    csv = data_dir / "driver.csv"
    if not csv.exists():
        logger.info("Dataset missing — generating...")
        data_gen.main()
    return pd.read_csv(csv)

def main():
    set_seed(42)
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)
    df = load_or_generate(data_dir)

    X, y = df[M.FEATURE_COLS], df[M.TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = M.build_model()
    logger.info("Training driver-state classifier ...")
    clf.fit(X_tr, y_tr)

    pred = clf.predict(X_te)
    acc = accuracy_score(y_te, pred)
    logger.info(f"Test accuracy = {acc:.3f}")
    joblib.dump(clf, model_dir / "driver_clf.joblib")
    logger.info(f"Saved model -> {model_dir / 'driver_clf.joblib'}")

if __name__ == "__main__":
    main()
