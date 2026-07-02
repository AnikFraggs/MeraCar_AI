"""Train Module 6's Incident Classifier & POI Scorer."""
from __future__ import annotations
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error
from ..common.utils import set_seed, module_dir, ensure_dirs, setup_logger
from . import data_gen, model as M

logger = setup_logger("Module6_Train")

def load_or_generate(data_dir):
    inc_csv = data_dir / "emergency.csv"
    poi_csv = data_dir / "poi_routes.csv"
    if not inc_csv.exists() or not poi_csv.exists():
        logger.info("Dataset missing — generating...")
        data_gen.main()
    return pd.read_csv(inc_csv), pd.read_csv(poi_csv)

def main():
    set_seed(42)
    base = module_dir(__file__)
    data_dir, model_dir = ensure_dirs(base)
    inc_df, poi_df = load_or_generate(data_dir)

    # 1. Train Incident Classifier
    X, y = inc_df[M.FEATURE_COLS], inc_df[M.TARGET_COL]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = M.build_model()
    logger.info("Training incident classifier ...")
    clf.fit(X_tr, y_tr)
    logger.info(f"Incident Classifier Accuracy: {accuracy_score(y_te, clf.predict(X_te)):.3f}")

    # 2. Train POI Scorer (XGBoost)
    Xp = poi_df[M.POI_SCORER_FEATURES]
    yp = poi_df[M.POI_SCORER_TARGET]
    Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(Xp, yp, test_size=0.2, random_state=42)
    scorer = M.build_poi_scorer()
    logger.info("Training POI XGBoost Scorer ...")
    scorer.fit(Xp_tr, yp_tr)
    mae = mean_absolute_error(yp_te, scorer.predict(Xp_te))
    logger.info(f"POI Scorer MAE: {mae:.2f} mins")

    # Save models
    joblib.dump(clf, model_dir / "emergency_clf.joblib")
    joblib.dump(scorer, model_dir / "poi_scorer.joblib")
    logger.info(f"Saved models -> {model_dir}")

if __name__ == "__main__":
    main()