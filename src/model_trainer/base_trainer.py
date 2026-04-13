# ──────────────────────────────────────────────────────────────
#  src/model_trainer/base_trainer.py
#  Abstract base class for all model trainers.
#  XGBoostTrainer and LSTMTrainer both inherit from this.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from config.config import settings
from config.constants import ALL_FEATURES, TARGET_COL, DATE_COL
from utils.backtest_utils import run_backtest
from utils.evaluation_utils import compute_classification_metrics, compute_financial_metrics
from utils.logging_utils import get_logger

log = get_logger(__name__)


class BaseTrainer(ABC):
    """
    Shared scaffolding for XGBoost and LSTM trainers:
      - train/test split
      - feature scaling
      - TimeSeriesSplit cross-validation
      - evaluation + backtest
      - save/load
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        target_col: str                = TARGET_COL,
        train_ratio: float | None      = None,
        n_cv_splits: int | None        = None,
        random_seed: int | None        = None,
    ):
        self.feature_cols = feature_cols or ALL_FEATURES
        self.target_col   = target_col
        self.train_ratio  = train_ratio  or settings.model.train_ratio
        self.n_cv_splits  = n_cv_splits  or settings.model.n_cv_splits
        self.random_seed  = random_seed  or settings.model.random_seed
        self.scaler       = StandardScaler()

        # These are set after train() is called
        self.model        = None
        self.train_df: pd.DataFrame | None = None
        self.test_df:  pd.DataFrame | None = None
        self.X_train: np.ndarray | None    = None
        self.X_test:  np.ndarray | None    = None
        self.y_train: np.ndarray | None    = None
        self.y_test:  np.ndarray | None    = None

    # ── Template method ───────────────────────────────────────

    def fit(self, df: pd.DataFrame) -> dict:
        """
        Full training pipeline:
          1. Split data
          2. Scale features
          3. Cross-validate
          4. Train final model
          5. Evaluate on held-out test set
          6. Run backtest

        Returns a results dict with metrics and backtest DataFrame.
        """
        self._split_and_scale(df)
        cv_results   = self.cross_validate()
        self._train_final()
        eval_metrics = self._evaluate()
        bt_df        = self._backtest(df)

        return {
            "cv_results"  : cv_results,
            "eval_metrics": eval_metrics,
            "backtest"    : bt_df,
        }

    # ── Abstract methods (subclasses implement) ───────────────

    @abstractmethod
    def _train_final(self) -> None:
        """Train the model on the full training set."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (predictions, probabilities) for array X."""

    @abstractmethod
    def save(self, out_dir: Path | None = None) -> Path:
        """Persist the trained model to disk."""

    @abstractmethod
    def load(self, model_path: Path) -> None:
        """Load a previously saved model from disk."""

    # ── Shared helpers ────────────────────────────────────────

    def _split_and_scale(self, df: pd.DataFrame) -> None:
        """Chronological 80/20 train/test split + StandardScaler fit."""
        df = df.sort_values(DATE_COL).reset_index(drop=True)

        # Intersect requested features with available columns
        available     = [c for c in self.feature_cols if c in df.columns]
        missing       = set(self.feature_cols) - set(available)
        if missing:
            log.warning("Missing feature columns (will be ignored): %s", missing)
        self.feature_cols = available

        split_idx     = int(len(df) * self.train_ratio)
        self.train_df = df.iloc[:split_idx].copy()
        self.test_df  = df.iloc[split_idx:].copy()

        self.X_train  = self.scaler.fit_transform(self.train_df[self.feature_cols].values)
        self.X_test   = self.scaler.transform(self.test_df[self.feature_cols].values)
        self.y_train  = self.train_df[self.target_col].values
        self.y_test   = self.test_df[self.target_col].values

        log.info(
            "Split: train=%d rows  test=%d rows  features=%d",
            len(self.train_df), len(self.test_df), len(self.feature_cols),
        )

    def cross_validate(self) -> dict:
        """
        TimeSeriesSplit cross-validation.
        Subclasses may override to add model-specific verbosity.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)
        fold_accs: list[float] = []

        X_all = np.vstack([self.X_train, self.X_test])
        y_all = np.concatenate([self.y_train, self.y_test])

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_all), 1):
            self._cv_fold_train(X_all[train_idx], y_all[train_idx])
            preds, _ = self.predict(X_all[val_idx])
            acc = (preds == y_all[val_idx]).mean()
            fold_accs.append(acc)
            log.info("Fold %d/%d — val_accuracy=%.4f", fold, self.n_cv_splits, acc)

        mean_acc = float(np.mean(fold_accs))
        log.info("CV mean accuracy: %.4f ± %.4f", mean_acc, float(np.std(fold_accs)))
        return {"fold_accuracies": fold_accs, "mean_accuracy": mean_acc}

    def _cv_fold_train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a fold model. Subclasses override if needed."""
        pass

    def _evaluate(self) -> dict:
        """Evaluate on held-out test set."""
        preds, probs = self.predict(self.X_test)
        metrics = compute_classification_metrics(
            self.y_test, preds, probs, label=self.__class__.__name__
        )
        return metrics

    def _backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run backtest on test set predictions."""
        preds, _ = self.predict(self.X_test)
        actual   = self.test_df["target_return"].values
        dates    = self.test_df[DATE_COL].reset_index(drop=True)
        return run_backtest(preds, actual, dates)
