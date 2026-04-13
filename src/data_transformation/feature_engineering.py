from __future__ import annotations
import numpy as np
import pandas as pd
import ta
from config.constants import (
    BB_STD, BB_WINDOW, CLOSE_COL, HIGH_COL, LOW_COL,
    MA_LONG, MA_MID, MA_SHORT,
    MACD_FAST, MACD_SIGNAL, MACD_SLOW,
    RSI_WINDOW, VOLUME_COL, VOLUME_MA,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


class FeatureEngineer:
    """
    Transforms raw OHLCV DataFrame into a rich feature matrix.

    Usage
    -----
    fe = FeatureEngineer()
    features_df = fe.transform(ohlcv_df)
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full feature engineering pipeline.

        Parameters
        ----------
        df : OHLCV DataFrame with DatetimeIndex and columns
             [Open, High, Low, Close, Volume]

        Returns
        -------
        DataFrame with price features + target variable (no NaN rows).
        """
        log.info("Engineering price features from %d OHLCV rows …", len(df))
        prices = df.copy()

        prices = self._add_return_features(prices)
        prices = self._add_moving_averages(prices)
        prices = self._add_technical_indicators(prices)
        prices = self._add_volume_features(prices)
        prices = self._add_target(prices)
        prices = self._add_lag_features(prices)
        prices = self._final_cleanup(prices)

        log.info("Feature engineering complete: %d rows × %d columns.", *prices.shape)
        return prices

    # ── Return features ───────────────────────────────────────

    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["return_1d"]  = df[CLOSE_COL].pct_change(1)
        df["return_3d"]  = df[CLOSE_COL].pct_change(3)
        df["return_5d"]  = df[CLOSE_COL].pct_change(5)
        df["return_10d"] = df[CLOSE_COL].pct_change(10)

        df["volatility_5d"]  = df["return_1d"].rolling(5).std()
        df["volatility_20d"] = df["return_1d"].rolling(20).std()

        df["intraday_range"] = (df[HIGH_COL] - df[LOW_COL]) / df[CLOSE_COL]
        return df

    # ── Moving averages ───────────────────────────────────────

    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f"ma_{MA_SHORT}"]  = df[CLOSE_COL].rolling(MA_SHORT).mean()
        df[f"ma_{MA_MID}"]   = df[CLOSE_COL].rolling(MA_MID).mean()
        df[f"ma_{MA_LONG}"]  = df[CLOSE_COL].rolling(MA_LONG).mean()

        # Ratio of fast MA to slow MA — captures trend direction
        df["ma_5_vs_20"]  = df[f"ma_{MA_SHORT}"]  / df[f"ma_{MA_MID}"]  - 1
        df["ma_20_vs_50"] = df[f"ma_{MA_MID}"]   / df[f"ma_{MA_LONG}"] - 1
        return df

    # ── Technical indicators ──────────────────────────────────

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # RSI
        df["rsi_14"] = ta.momentum.RSIIndicator(
            close=df[CLOSE_COL], window=RSI_WINDOW
        ).rsi()

        # MACD
        macd_obj = ta.trend.MACD(
            close=df[CLOSE_COL],
            window_fast=MACD_FAST,
            window_slow=MACD_SLOW,
            window_sign=MACD_SIGNAL,
        )
        df["macd"]      = macd_obj.macd()
        df["macd_sig"]  = macd_obj.macd_signal()
        df["macd_hist"] = macd_obj.macd_diff()

        # Bollinger Bands
        bb_obj = ta.volatility.BollingerBands(
            close=df[CLOSE_COL], window=BB_WINDOW, window_dev=BB_STD
        )
        df["bb_upper"] = bb_obj.bollinger_hband()
        df["bb_lower"] = bb_obj.bollinger_lband()
        df["bb_mid"]   = bb_obj.bollinger_mavg()
        df["bb_pct"]   = bb_obj.bollinger_pband()   # %B: position within bands [0, 1]

        return df

    # ── Volume features ───────────────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["vol_ma_20"]    = df[VOLUME_COL].rolling(VOLUME_MA).mean()
        df["volume_ratio"] = df[VOLUME_COL] / (df["vol_ma_20"] + 1e-9)
        return df

    # ── Target variable ───────────────────────────────────────

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        target_return    : continuous next-day return (regression target)
        target_direction : binary 1 = price went up, 0 = price went down
        """
        df["target_return"]    = df[CLOSE_COL].pct_change(1).shift(-1)
        df["target_direction"] = (df["target_return"] > 0).astype(int)
        return df

    # ── Lag features ──────────────────────────────────────────

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Shift key features forward in time so the model sees past values only.
        shift(n) moves row T's value to row T+n — i.e., row T now sees T-n's data.
        """
        df = df.copy().sort_index()

        for lag in [1, 2, 3, 5]:
            df[f"return_lag{lag}"] = df["return_1d"].shift(lag)

        df["return_past_week"] = df["return_5d"].shift(1)

        return df

    # ── Cleanup ───────────────────────────────────────────────

    def _final_cleanup(self, df: pd.DataFrame) -> pd.DataFrame:
        """Reset index to a 'date' column and drop NaN rows."""
        df = df.reset_index()
        df = df.rename(columns={"index": "date"}) if "date" not in df.columns else df
        df["date"] = pd.to_datetime(df["date"])

        # Drop last row (target_return is NaN — no tomorrow yet)
        # Drop first N rows (rolling windows require warm-up)
        df = df.dropna(subset=["target_return", "return_1d", "rsi_14"]).reset_index(drop=True)
        return df
