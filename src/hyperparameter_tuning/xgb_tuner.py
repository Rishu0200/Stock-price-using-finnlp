from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from config.config import settings
from utils.logging_utils import get_logger

log = get_logger(__name__)

# Optional Optuna import
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    log.warning("Optuna not installed. Falling back to RandomizedSearchCV.")


class XGBTuner:
    """
    Hyperparameter search for XGBoost classifier.

    Two backends:
      • Optuna (preferred)  — Bayesian optimisation, smarter search
      • RandomizedSearchCV  — fallback, no extra dependencies

    Usage
    -----
    tuner = XGBTuner(n_trials=50)
    best_params = tuner.tune(X_train, y_train)
    # Then pass best_params to XGBoostTrainer(params=best_params)
    """

    def __init__(
        self,
        n_trials: int  = 50,
        n_cv_splits: int = 5,
        random_seed: int | None = None,
        n_jobs: int    = -1,
    ):
        self.n_trials    = n_trials
        self.n_cv_splits = n_cv_splits
        self.random_seed = random_seed or settings.model.random_seed
        self.n_jobs      = n_jobs
        self.best_params: dict = {}
        self.study       = None   # Optuna study object (if used)

    def tune(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Run hyperparameter search and return best params dict.
        Use Optuna if available, else RandomizedSearchCV.
        """
        if _OPTUNA_AVAILABLE:
            return self._tune_optuna(X_train, y_train)
        return self._tune_random(X_train, y_train)

    # ── Optuna backend ────────────────────────────────────────

    def _tune_optuna(self, X: np.ndarray, y: np.ndarray) -> dict:
        log.info("Starting Optuna XGBoost tuning (%d trials) …", self.n_trials)
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        def objective(trial: "optuna.Trial") -> float:
            params = {
                "n_estimators"     : trial.suggest_int("n_estimators", 100, 600),
                "max_depth"        : trial.suggest_int("max_depth", 2, 8),
                "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
                "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "min_child_weight" : trial.suggest_int("min_child_weight", 1, 10),
                "gamma"            : trial.suggest_float("gamma", 0.0, 1.0),
                "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "scale_pos_weight" : (y == 0).sum() / max((y == 1).sum(), 1),
                "eval_metric"      : "logloss",
                "random_state"     : self.random_seed,
                "verbosity"        : 0,
            }

            fold_accs = []
            for train_idx, val_idx in tscv.split(X):
                model = xgb.XGBClassifier(**params)
                model.fit(X[train_idx], y[train_idx], verbose=False)
                preds = model.predict(X[val_idx])
                fold_accs.append((preds == y[val_idx]).mean())

            return float(np.mean(fold_accs))

        self.study = optuna.create_study(direction="maximize",
                                         sampler=optuna.samplers.TPESampler(seed=self.random_seed))
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)

        self.best_params = self.study.best_params
        log.info("Optuna best accuracy: %.4f | params: %s",
                 self.study.best_value, self.best_params)
        return self.best_params

    # ── RandomizedSearchCV backend ────────────────────────────

    def _tune_random(self, X: np.ndarray, y: np.ndarray) -> dict:
        log.info("Starting RandomizedSearchCV XGBoost tuning (%d iterations) …", self.n_trials)
        tscv = TimeSeriesSplit(n_splits=self.n_cv_splits)

        param_dist = {
            "n_estimators"     : [100, 200, 300, 400, 500],
            "max_depth"        : [2, 3, 4, 5, 6, 7, 8],
            "learning_rate"    : [0.005, 0.01, 0.05, 0.1, 0.2, 0.3],
            "subsample"        : [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree" : [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight" : [1, 2, 3, 5, 7, 10],
            "gamma"            : [0.0, 0.1, 0.2, 0.5, 1.0],
            "reg_alpha"        : [1e-4, 0.01, 0.1, 1.0, 10.0],
            "reg_lambda"       : [1e-4, 0.01, 0.1, 1.0, 10.0],
        }

        base_model = xgb.XGBClassifier(
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
            eval_metric="logloss",
            random_state=self.random_seed,
            verbosity=0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = RandomizedSearchCV(
                estimator  = base_model,
                param_distributions = param_dist,
                n_iter     = self.n_trials,
                scoring    = "accuracy",
                cv         = tscv,
                n_jobs     = self.n_jobs,
                random_state = self.random_seed,
                refit      = False,
            )
            search.fit(X, y)

        self.best_params = search.best_params_
        log.info("RandomizedSearch best accuracy: %.4f | params: %s",
                 search.best_score_, self.best_params)
        return self.best_params
