from __future__ import annotations
import re
import time
from pathlib import Path
import feedparser
import pandas as pd
from config.config import settings
from config.constants import FINANCIAL_RSS_FEEDS, NEWSDATA_REQUEST_DELAY, TICKER_KEYWORDS
from utils.date_utils import parse_date_flexible
from utils.logging_utils import get_logger

log = get_logger(__name__)

# ── Optional import for NewsData.io client ────────────────────
try:
    from newsdataapi import NewsDataApiClient
    _NEWSDATA_AVAILABLE = True
except ImportError:
    _NEWSDATA_AVAILABLE = False
    log.warning("newsdataapi not installed. NewsData.io source unavailable.")


class NewsFetcher:
    """
    Fetches and cleans financial news headlines for a given ticker.

    Usage
    -----
    fetcher = NewsFetcher(ticker="AAPL")
    df = fetcher.fetch()          # tries NewsData.io first, falls back to RSS
    fetcher.save(df)              # saves to data/raw/AAPL_news_raw.csv
    """

    def __init__(
        self,
        ticker: str     | None = None,
        api_key: str    | None = None,
        max_pages: int          = 3,
        language: str           = "en",
        category: str           = "business",
    ):
        self.ticker    = (ticker or settings.stock.ticker).upper()
        self.api_key   = api_key or settings.api.newsdata_api_key
        self.max_pages = max_pages
        self.language  = language
        self.category  = category
        self.keywords  = TICKER_KEYWORDS.get(self.ticker, [self.ticker.lower()])

    # ── Public API ────────────────────────────────────────────

    def fetch(self) -> pd.DataFrame:
        """
        Fetch headlines from NewsData.io if a key is available,
        otherwise fall back to RSS feeds.
        Returns a DataFrame with columns: [date, headline, source, url].
        """
        frames: list[pd.DataFrame] = []

        if self.api_key and _NEWSDATA_AVAILABLE:
            nd_df = self._fetch_newsdata()
            if not nd_df.empty:
                frames.append(nd_df)
                log.info("NewsData.io: %d headlines fetched.", len(nd_df))

        rss_df = self._fetch_rss_all()
        if not rss_df.empty:
            frames.append(rss_df)
            log.info("RSS: %d headlines fetched.", len(rss_df))

        if not frames:
            log.warning("No headlines fetched for %s.", self.ticker)
            return pd.DataFrame(columns=["date", "headline", "source", "url"])

        combined = pd.concat(frames, ignore_index=True)
        combined = self._clean_and_deduplicate(combined)
        log.info("Total after dedup: %d headlines for %s.", len(combined), self.ticker)
        return combined

    def save(self, df: pd.DataFrame, out_dir: Path | None = None) -> Path:
        out_dir = out_dir or settings.paths.data_dir / "raw"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{self.ticker}_news_raw.csv"
        df.to_csv(path, index=False)
        log.info("Saved raw news → %s", path)
        return path

    # ── NewsData.io ───────────────────────────────────────────

    def _fetch_newsdata(self) -> pd.DataFrame:
        """Paginate through NewsData.io API results for the ticker."""
        client = NewsDataApiClient(apikey=self.api_key)
        records: list[dict] = []
        page = None

        for page_num in range(self.max_pages):
            try:
                kwargs: dict = {
                    "q"        : self.ticker,
                    "language" : self.language,
                    "category" : self.category,
                }
                if page:
                    kwargs["page"] = page

                response = client.news_api(**kwargs)

                if response.get("status") != "success":
                    log.warning("NewsData.io returned status: %s", response.get("status"))
                    break

                for article in response.get("results", []):
                    records.append({
                        "date"    : article.get("pubDate"),
                        "headline": article.get("title", ""),
                        "source"  : article.get("source_id", "newsdata"),
                        "url"     : article.get("link", ""),
                    })

                page = response.get("nextPage")
                if not page:
                    break

                time.sleep(NEWSDATA_REQUEST_DELAY)

            except Exception as exc:
                log.error("NewsData.io fetch error on page %d: %s", page_num, exc)
                break

        return pd.DataFrame(records)

    # ── RSS feeds ─────────────────────────────────────────────

    def _fetch_rss_all(self) -> pd.DataFrame:
        """Fetch all RSS feeds in parallel (sequentially) and merge."""
        frames: list[pd.DataFrame] = []
        for feed_name, feed_url in FINANCIAL_RSS_FEEDS.items():
            df = self._fetch_single_rss(feed_name, feed_url)
            if not df.empty:
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        return self._filter_by_keywords(combined)

    def _fetch_single_rss(self, feed_name: str, feed_url: str) -> pd.DataFrame:
        """Parse a single RSS feed and return a normalized DataFrame."""
        try:
            parsed = feedparser.parse(feed_url)
            records = []
            for entry in parsed.entries:
                title   = entry.get("title", "")
                link    = entry.get("link", "")
                pub_str = entry.get("published", entry.get("updated", ""))
                records.append({
                    "date"    : pub_str,
                    "headline": title,
                    "source"  : feed_name,
                    "url"     : link,
                })
            return pd.DataFrame(records)
        except Exception as exc:
            log.debug("RSS fetch failed for %s: %s", feed_name, exc)
            return pd.DataFrame()

    def _filter_by_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only rows whose headline mentions any ticker keyword."""
        if df.empty:
            return df
        pattern = "|".join(re.escape(k) for k in self.keywords)
        mask    = df["headline"].str.lower().str.contains(pattern, na=False)
        return df[mask].reset_index(drop=True)

    # ── Cleaning ──────────────────────────────────────────────

    def _clean_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean headlines and drop duplicates."""
        df = df.copy()
        df["headline"] = df["headline"].apply(self._clean_headline)
        df = df[df["headline"].str.len() >= 10]   # drop very short / empty

        # Normalise date column
        df["date"] = df["date"].apply(
            lambda s: parse_date_flexible(str(s)) if pd.notna(s) else None
        )
        df = df.dropna(subset=["date"])

        # Deduplicate on normalised headline
        df["_norm"] = df["headline"].str.lower().str.strip()
        df = df.drop_duplicates(subset=["_norm"]).drop(columns=["_norm"])
        return df.reset_index(drop=True)

    @staticmethod
    def _clean_headline(text: str) -> str:
        """Strip HTML, special characters, and common boilerplate."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"<[^>]+>", " ", text)            # HTML tags
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = re.sub(r"http\S+", "", text)              # URLs
        text = re.sub(r"[^\w\s\-\.,!?\'\"$%]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # Strip common boilerplate suffixes
        for suffix in [" - Reuters", " | Reuters", " - CNBC", " - MarketWatch"]:
            text = text.replace(suffix, "")
        return text.strip()
