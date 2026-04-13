# ──────────────────────────────────────────────────────────────
#  src/data_transformation/data_merger.py
#  Aligns news headlines to correct trading days, merges price
#  features + sentiment features, adds lag/rolling features,
#  and produces the final feature matrix saved to artifacts/.
#  Implements Phase 5 Sections 7, 9, 10, 11.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import settings
from config.constants import (
    ALL_FEATURES,
    DATE_COL,
    META_COLS,
    PRICE_FEATURES,
    SENTIMENT_FEATURES,
    SENTIMENT_NEUTRAL_FILL,
    TARGET_COL,
    TARGET_RETURN,
    FEATURE_FILE_TPL,
)
from utils.date_utils import assign_trade_date, get_trading_days
from utils.logging_utils import get_logger

log = get_logger(__name__)


class DataMerger:
    """
    Merges price features and sentiment features into a single feature matrix.

    Steps performed:
      1. Align each headline to its correct trading day (no look-ahead)
      2. Left-join sentiment onto price features (every trading day kept)
      3. Fill missing sentiment with neutral defaults
      4. Add lag + rolling sentiment features
      5. Drop NaN warm-up rows

    Usage
    -----
    merger = DataMerger(ticker="AAPL")
    final_df = merger.merge(price_df, sentiment_df, news_df)
    merger.save(final_df)
    """

    def __init__(self, ticker: str | None = None):
        self.ticker = (ticker or settings.stock.ticker).upper()

    # ── Public API ────────────────────────────────────────────

    def align_news(
        self,
        news_df: pd.DataFrame,
        trading_days: pd.DatetimeIndex,
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Assign each headline to its correct trading day.

        Rule (Phase 5 Section 7):
          Published before market close on day T  →  trade_date = T
          Published after  market close on day T  →  trade_date = T+1
        """
        log.info("Aligning %d headlines to trading days …", len(news_df))
        df = news_df.copy()
        df["trade_date"] = df[date_col].apply(
            lambda dt: assign_trade_date(dt, trading_days) if dt is not None else None
        )
        before = len(df)
        df = df.dropna(subset=["trade_date"])
        log.info(
            "Aligned %d/%d headlines (dropped %d with no valid trading day).",
            len(df), before, before - len(df),
        )
        return df

    def merge(
        self,
        price_df: pd.DataFrame,
        sentiment_df: pd.DataFrame,
        news_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Full merge pipeline. Returns the final feature matrix.

        Parameters
        ----------
        price_df     : output of FeatureEngineer.transform()
        sentiment_df : output of FinBERTPipeline.run() (already per-day)
        news_df      : raw headlines (optional — used only for alignment logging)
        """
        price_df = price_df.copy()
        sentiment_df = sentiment_df.copy()

        # Ensure date columns are datetime
        price_df["date"]     = pd.to_datetime(price_df["date"])
        if not sentiment_df.empty:
            sentiment_df["date"] = pd.to_datetime(sentiment_df["date"])

        # Left join — keep every trading day even without news
        log.info("Merging price (%d rows) + sentiment (%d rows) …",
                 len(price_df), len(sentiment_df))

        master = price_df.merge(sentiment_df, on="date", how="left")

        # Fill missing sentiment with neutral values
        for col, fill_val in SENTIMENT_NEUTRAL_FILL.items():
            if col in master.columns:
                master[col] = master[col].fillna(fill_val)
            else:
                master[col] = fill_val

        # Add lag + rolling sentiment features
        master = self._add_sentiment_lags(master)

        # Drop NaN warm-up rows
        key_cols = ["return_1d", "rsi_14", "target_return"]
        key_cols = [c for c in key_cols if c in master.columns]
        master   = master.dropna(subset=key_cols).reset_index(drop=True)

        log.info("Final feature matrix: %d rows × %d columns.", *master.shape)
        return master

    def save(self, df: pd.DataFrame, out_dir: Path | None = None) -> Path:
        """Save the feature matrix to artifacts/ directory."""
        out_dir = out_dir or settings.paths.artifacts_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = FEATURE_FILE_TPL.format(ticker=self.ticker)
        path = out_dir / filename
        df.to_csv(path, index=False)
        log.info("Feature matrix saved → %s  (%d rows × %d cols)", path, *df.shape)
        return path

    # ── Private helpers ───────────────────────────────────────

    def _add_sentiment_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lag-1/2/3 and rolling-3/5 sentiment features."""
        df = df.copy().sort_values("date").reset_index(drop=True)

        if "sentiment_conf_wt" not in df.columns:
            return df

        for lag in [1, 2, 3]:
            df[f"sentiment_lag{lag}"] = df["sentiment_conf_wt"].shift(lag)

        df["sentiment_roll3"] = df["sentiment_conf_wt"].rolling(3).mean().shift(1)
        df["sentiment_roll5"] = df["sentiment_conf_wt"].rolling(5).mean().shift(1)

        return df
