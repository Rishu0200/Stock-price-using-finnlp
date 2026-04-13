from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import xgboost as xgb
from config.config import settings
from config.constants import XGB_MODEL_FILE
from src.model_trainer.base_trainer import BaseTrainer
from utils.logging_utils import get_logger

log = get_logger(__name__)


class XGBoostTrainer(BaseTrainer):
    """
    XGBoost classifier with TimeSeriesSplit CV and backtest integration.

    Usage
    -----
    trainer = XGBoostTrainer()
    results = trainer.fit(feature_matrix_df)

    results["eval_metrics"]  →  accuracy, AUC, classification report
    results["cv_results"]    →  per-fold accuracies
    results["backtest"]      →  portfolio DataFrame
    """

    def __init__(self, params: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        cfg = settings.model
        self._params = params or {
            "n_estimators"     : cfg.xgb_n_estimators,
            "max_depth"        : cfg.xgb_max_depth,
            "learning_rate"    : cfg.xgb_learning_rate,
            "subsample"        : cfg.xgb_subsample,
            "colsample_bytree" : cfg.xgb_colsample,
            "min_child_weight" : cfg.xgb_min_child_weight,
            "gamma"            : cfg.xgb_gamma,
            "reg_alpha"        : cfg.xgb_reg_alpha,
            "reg_lambda"       : cfg.xgb_reg_lambda,
            "eval_metric"      : "logloss",
            "random_state"     : self.random_seed,
            "verbosity"        : 0,
        }

    # ── BaseTrainer implementation ────────────────────────────

    def _train_final(self) -> None:
        """Fit XGBoost on the full training set."""
        params = dict(self._params)
        # Handle class imbalance automatically
        n_neg = (self.y_train == 0).sum()
        n_pos = (self.y_train == 1).sum()
        if n_pos > 0:
            params["scale_pos_weight"] = n_neg / n_pos

        log.info("Training XGBoost on %d samples (n_estimators=%d) …",
                 len(self.X_train), params["n_estimators"])

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=False,
        )
        log.info("XGBoost training complete.")

    def _cv_fold_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a lightweight fold model for CV."""
        fold_params = dict(self._params)
        fold_params["n_estimators"] = 200    # faster for CV
        self.model = xgb.XGBClassifier(**fold_params)
        self.model.fit(X, y, verbose=False)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (binary predictions, probability of class 1)."""
        if self.model is None:
            raise RuntimeError("Model not trained. Call fit() first.")
        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs

    def feature_importance(self) -> dict[str, float]:
        """Return dict of feature_name → importance score."""
        if self.model is None:
            raise RuntimeError("Model not trained.")
        return dict(zip(self.feature_cols, self.model.feature_importances_))

    def save(self, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or settings.paths.models_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / XGB_MODEL_FILE
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        log.info("XGBoost model saved → %s", path)

        # Also save the scaler
        scaler_path = out_dir / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)
        log.info("Scaler saved → %s", scaler_path)

        return path

    def load(self, model_path: Path) -> None:
        model_path = Path(model_path)
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        scaler_path = model_path.parent / "feature_scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
            log.info("Scaler loaded from %s", scaler_path)

        log.info("XGBoost model loaded from %s", model_path)
