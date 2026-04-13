from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf
from config.config import settings
from config.constants import (
    CLOSE_COL, HIGH_COL, LOW_COL, OPEN_COL, VOLUME_COL,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


class StockFetcher:
    """
    Downloads OHLCV data for one or more tickers from Yahoo Finance.

    Usage
    -----
    fetcher = StockFetcher(ticker="AAPL", start="2023-01-01", end="2024-12-31")
    df      = fetcher.fetch()             # returns cleaned DataFrame
    fetcher.save(df)                      # saves to data/raw/AAPL_ohlcv.csv
    """

    def __init__(
        self,
        ticker: str     | None = None,
        start: str      | None = None,
        end: str        | None = None,
        interval: str          = "1d",
        auto_adjust: bool      = True,
    ):
        self.ticker      = (ticker or settings.stock.ticker).upper()
        self.start       = start  or settings.stock.start_date
        self.end         = end    or settings.stock.end_date
        self.interval    = interval
        self.auto_adjust = auto_adjust

    # ── Public API ────────────────────────────────────────────

    def fetch(self) -> pd.DataFrame:
        """Download and clean OHLCV data. Returns a DataFrame indexed by date."""
        log.info("Downloading %s price data (%s → %s) …", self.ticker, self.start, self.end)

        raw = yf.download(
            self.ticker,
            start       = self.start,
            end         = self.end,
            interval    = self.interval,
            auto_adjust = self.auto_adjust,
            progress    = False,
        )

        if raw.empty:
            raise ValueError(f"yfinance returned empty DataFrame for {self.ticker}. "
                             "Check the ticker symbol and date range.")

        df = self._clean(raw)
        log.info("Fetched %d trading days for %s.", len(df), self.ticker)
        return df

    def fetch_benchmarks(self, extra_tickers: list[str] | None = None) -> pd.DataFrame:
        """
        Download Close prices for the main ticker plus benchmark ETFs (SPY, QQQ).
        Useful for relative-strength feature engineering.
        """
        tickers = [self.ticker] + (extra_tickers or ["SPY", "QQQ"])
        log.info("Downloading benchmarks: %s", tickers)

        raw = yf.download(
            tickers,
            start       = self.start,
            end         = self.end,
            auto_adjust = True,
            progress    = False,
        )[CLOSE_COL]

        raw.columns = [str(c) for c in raw.columns]
        raw.index   = pd.to_datetime(raw.index)
        raw.index.name = "date"
        return raw.dropna()

    def get_ticker_info(self) -> dict:
        """Return metadata for the ticker (company name, sector, P/E, etc.)."""
        info = yf.Ticker(self.ticker).info
        return {
            "symbol"    : self.ticker,
            "name"      : info.get("longName"),
            "sector"    : info.get("sector"),
            "industry"  : info.get("industry"),
            "market_cap": info.get("marketCap"),
            "pe_ratio"  : info.get("trailingPE"),
            "52w_high"  : info.get("fiftyTwoWeekHigh"),
            "52w_low"   : info.get("fiftyTwoWeekLow"),
        }

    def save(self, df: pd.DataFrame, out_dir: Path | None = None) -> Path:
        """Save the fetched OHLCV DataFrame as CSV and return the file path."""
        out_dir = out_dir or settings.paths.data_dir / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.ticker}_ohlcv.csv"
        df.to_csv(path)
        log.info("Saved OHLCV data → %s", path)
        return path

    # ── Private helpers ───────────────────────────────────────

    def _clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Flatten multi-level columns, rename, sort, and forward-fill."""
        df = raw.copy()

        # yfinance sometimes returns MultiIndex columns — flatten them
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        # Ensure expected columns exist
        expected = [OPEN_COL, HIGH_COL, LOW_COL, CLOSE_COL, VOLUME_COL]
        missing  = [c for c in expected if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns from yfinance: {missing}")

        df.index     = pd.to_datetime(df.index)
        df.index.name = "date"
        df           = df.sort_index()
        df           = df[expected]           # keep only OHLCV
        df           = df.ffill().dropna()    # forward-fill gaps (holidays)

        return df
