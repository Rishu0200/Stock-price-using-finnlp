# ──────────────────────────────────────────────────────────────
#  src/prediction_pipeline/predictor.py
#  Full real-time prediction pipeline:
#    1. Fetch latest OHLCV data
#    2. Fetch latest news headlines
#    3. Run FinBERT sentiment
#    4. Engineer features
#    5. Load saved model and predict next-day direction
#  This is the single entry point for inference / serving.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from config.config import settings
from config.constants import (
    ALL_FEATURES, DATE_COL, FEATURE_FILE_TPL,
    LSTM_MODEL_FILE, SCALER_FILE, XGB_MODEL_FILE,
)
from src.data_ingestion.news_fetcher import NewsFetcher
from src.data_ingestion.stock_fetcher import StockFetcher
from src.data_transformation.data_merger import DataMerger
from src.data_transformation.feature_engineering import FeatureEngineer
from src.data_transformation.sentiment_pipeline import FinBERTPipeline
from utils.date_utils import get_trading_days
from utils.logging_utils import get_logger

log = get_logger(__name__)


class StockPredictor:
    """
    End-to-end prediction pipeline for next-day stock direction.

    Usage — single prediction
    -------------------------
    predictor = StockPredictor(ticker="AAPL", model_type="xgboost")
    result = predictor.predict()
    # result = {"date": "2024-06-05", "direction": "UP", "probability": 0.63}

    Usage — batch from saved feature matrix
    ----------------------------------------
    result = predictor.predict_from_file("artifacts/AAPL_feature_matrix.csv")
    """

    def __init__(
        self,
        ticker: str      | None = None,
        model_type: Literal["xgboost", "lstm"] | None = None,
        models_dir: Path | None = None,
    ):
        self.ticker     = (ticker or settings.stock.ticker).upper()
        self.model_type = model_type or settings.model.model_type
        self.models_dir = models_dir or settings.paths.models_dir

        self._model      = None
        self._scaler     = None
        self._feature_cols: list[str] = ALL_FEATURES

    # ── Public API ────────────────────────────────────────────

    def predict(
        self,
        lookback_days: int = 90,
    ) -> dict:
        """
        Fetch fresh data, build features, and predict next-day direction.

        Parameters
        ----------
        lookback_days : how many calendar days of history to fetch

        Returns
        -------
        dict: date, direction ("UP" / "DOWN"), probability, model_type
        """
        self._ensure_model_loaded()

        end_date   = datetime.today().strftime("%Y-%m-%d")
        start_date = (datetime.today() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")

        log.info("Running real-time prediction for %s (lookback %dd) …",
                 self.ticker, lookback_days)

        # ── 1. Fetch price data ───────────────────────────────
        stock = StockFetcher(ticker=self.ticker, start=start_date, end=end_date)
        ohlcv = stock.fetch()

        # ── 2. Fetch news ─────────────────────────────────────
        news  = NewsFetcher(ticker=self.ticker)
        news_df = news.fetch()

        # ── 3. FinBERT sentiment ──────────────────────────────
        trading_days = get_trading_days(start_date, end_date)
        merger       = DataMerger(ticker=self.ticker)
        sentiment_df = pd.DataFrame()

        if not news_df.empty:
            aligned_news = merger.align_news(news_df, trading_days)
            if not aligned_news.empty:
                finbert      = FinBERTPipeline()
                sentiment_df = finbert.run(aligned_news,
                                           text_col="headline",
                                           date_col="trade_date")

        # ── 4. Feature engineering ────────────────────────────
        fe           = FeatureEngineer()
        price_feats  = fe.transform(ohlcv)
        feature_matrix = merger.merge(price_feats, sentiment_df)

        if feature_matrix.empty:
            raise ValueError("Feature matrix is empty — not enough data to predict.")

        # Use the most recent row (latest trading day's features)
        latest_row   = feature_matrix.iloc[[-1]]
        available    = [c for c in self._feature_cols if c in latest_row.columns]
        X_latest     = self._scaler.transform(latest_row[available].values)

        pred, prob = self._model_predict(X_latest)
        direction  = "UP" if pred[0] == 1 else "DOWN"
        pred_date  = latest_row["date"].iloc[0]

        result = {
            "ticker"     : self.ticker,
            "as_of_date" : str(pred_date.date() if hasattr(pred_date, "date") else pred_date),
            "direction"  : direction,
            "probability": round(float(prob[0]), 4),
            "model_type" : self.model_type,
        }
        log.info("Prediction: %s | direction=%s | prob=%.4f",
                 self.ticker, direction, prob[0])
        return result

    def predict_from_file(
        self,
        feature_file: str | Path | None = None,
        n_rows: int = 10,
    ) -> pd.DataFrame:
        """
        Run predictions on a saved feature matrix CSV.
        Returns last n_rows of predictions as a DataFrame.
        """
        self._ensure_model_loaded()

        if feature_file is None:
            feature_file = (settings.paths.artifacts_dir /
                            FEATURE_FILE_TPL.format(ticker=self.ticker))

        feature_file = Path(feature_file)
        if not feature_file.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_file}")

        df = pd.read_csv(feature_file, parse_dates=[DATE_COL])
        log.info("Loaded feature matrix: %d rows from %s", len(df), feature_file)

        available = [c for c in self._feature_cols if c in df.columns]
        X         = self._scaler.transform(df[available].values)
        preds, probs = self._model_predict(X)

        result_df = df[[DATE_COL]].copy()
        result_df["prediction"]  = preds
        result_df["direction"]   = np.where(preds == 1, "UP", "DOWN")
        result_df["probability"] = probs.round(4)

        return result_df.tail(n_rows).reset_index(drop=True)

    # ── Model loading ─────────────────────────────────────────

    def _ensure_model_loaded(self) -> None:
        if self._model is not None:
            return
        self._load_model()
        self._load_scaler()

    def _load_model(self) -> None:
        if self.model_type == "xgboost":
            path = self.models_dir / XGB_MODEL_FILE
            if not path.exists():
                raise FileNotFoundError(f"XGBoost model not found: {path}. Run training first.")
            import pickle
            with open(path, "rb") as f:
                self._model = pickle.load(f)
            log.info("XGBoost model loaded from %s", path)

        elif self.model_type == "lstm":
            import torch
            from src.model_trainer.lstm_trainer import LSTMTrainer
            path = self.models_dir / LSTM_MODEL_FILE
            if not path.exists():
                raise FileNotFoundError(f"LSTM model not found: {path}. Run training first.")
            trainer = LSTMTrainer()
            trainer.load(path)
            self._model      = trainer
            self._lstm_mode  = True
            log.info("LSTM model loaded from %s", path)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _load_scaler(self) -> None:
        path = self.models_dir / SCALER_FILE
        if not path.exists():
            log.warning("Scaler not found at %s — using identity transform.", path)
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            return
        with open(path, "rb") as f:
            self._scaler = pickle.load(f)

    def _model_predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dispatch to the right model's predict method."""
        if self.model_type == "lstm":
            return self._model.predict(X)
        # XGBoost
        probs = self._model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return preds, probs
