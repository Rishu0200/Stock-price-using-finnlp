from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from config.config import settings
from config.constants import (
    CONF_HIGH,
    CONF_MEDIUM,
    FINBERT_LABELS,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_NAME,
    FINBERT_NEG_LABEL,
    FINBERT_NEU_LABEL,
    FINBERT_POS_LABEL,
    SENTIMENT_NEUTRAL_FILL,
)
from utils.logging_utils import get_logger

log = get_logger(__name__)


class FinBERTPipeline:
    """
    Wraps ProsusAI/finbert for batch sentiment inference on financial headlines.

    Usage
    -----
    pipe = FinBERTPipeline()
    sentiment_df = pipe.run(news_df, text_col="headline", date_col="trade_date")
    # Returns one row per trading day with aggregated sentiment features.
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        device: str | None     = None,
        conf_high: float       = CONF_HIGH,
        conf_medium: float     = CONF_MEDIUM,
    ):
        self.model_name  = model_name  or settings.finbert.model_name
        self.batch_size  = batch_size  or settings.finbert.batch_size
        self.conf_high   = conf_high
        self.conf_medium = conf_medium
        self.device      = self._resolve_device(device)
        self._pipe       = None   # lazy-loaded on first call to run()

    # ── Public API ────────────────────────────────────────────

    def run(
        self,
        news_df: pd.DataFrame,
        text_col: str  = "headline",
        date_col: str  = "trade_date",
        batch_size: int | None = None,
        min_conf: float = 0.0,
    ) -> pd.DataFrame:
        """
        Run FinBERT on all headlines and return per-day aggregated features.

        Parameters
        ----------
        news_df    : DataFrame with at least [text_col, date_col] columns
        text_col   : column name containing raw headline text
        date_col   : column name containing the aligned trading date
        batch_size : override default batch size
        min_conf   : minimum confidence threshold (0.0 = keep all)

        Returns
        -------
        DataFrame indexed by trading date with columns:
          sentiment_mean, sentiment_conf_wt, sentiment_std,
          pos_ratio, neg_ratio, neu_ratio, headline_count
        """
        if news_df.empty:
            log.warning("Empty news DataFrame passed to FinBERTPipeline.run().")
            return pd.DataFrame()

        self._ensure_loaded()
        bs = batch_size or self.batch_size

        texts = news_df[text_col].fillna("").tolist()
        log.info("Running FinBERT on %d headlines (batch=%d) …", len(texts), bs)

        raw_results = self._batch_infer(texts, bs)

        # Attach predictions back to news_df
        df = news_df[[date_col, text_col]].copy()
        df["label"]      = [r["label"]  for r in raw_results]
        df["confidence"] = [r["score"]  for r in raw_results]

        # Filter by minimum confidence
        if min_conf > 0.0:
            df = df[df["confidence"] >= min_conf]

        # Map label → numeric score: positive=+1, negative=-1, neutral=0
        label_map = {FINBERT_POS_LABEL: 1.0, FINBERT_NEG_LABEL: -1.0, FINBERT_NEU_LABEL: 0.0}
        df["score"] = df["label"].map(label_map).fillna(0.0)

        # Confidence weighting: high-conf headlines count more
        df["weight"] = df["confidence"].apply(self._confidence_weight)
        df["weighted_score"] = df["score"] * df["weight"]

        # Aggregate per trading day
        daily = self._aggregate_by_day(df, date_col)
        log.info("Sentiment features computed for %d trading days.", len(daily))
        return daily

    # ── Private helpers ───────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load the FinBERT model on first use."""
        if self._pipe is not None:
            return
        log.info("Loading FinBERT model: %s (device=%s) …", self.model_name, self.device)
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model     = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if self.device == "cuda" else -1,
            max_length=FINBERT_MAX_LENGTH,
            truncation=True,
        )
        log.info("FinBERT loaded.")

    def _batch_infer(self, texts: list[str], batch_size: int) -> list[dict]:
        """Run inference in batches, return list of {label, score} dicts."""
        results = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            out   = self._pipe(batch, truncation=True, max_length=FINBERT_MAX_LENGTH)
            results.extend(out)
        return results

    def _confidence_weight(self, conf: float) -> float:
        """Assign higher weight to more confident predictions."""
        if conf >= self.conf_high:
            return 1.5
        if conf >= self.conf_medium:
            return 1.0
        return 0.5

    def _aggregate_by_day(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """Aggregate per-headline scores to one row per trading day."""
        records = []
        for trade_date, grp in df.groupby(date_col):
            w_sum = grp["weight"].sum() + 1e-9
            records.append({
                "date"             : trade_date,
                "sentiment_mean"   : grp["score"].mean(),
                "sentiment_conf_wt": (grp["weighted_score"].sum() / w_sum),
                "sentiment_std"    : grp["score"].std(ddof=0),
                "pos_ratio"        : (grp["label"] == FINBERT_POS_LABEL).mean(),
                "neg_ratio"        : (grp["label"] == FINBERT_NEG_LABEL).mean(),
                "neu_ratio"        : (grp["label"] == FINBERT_NEU_LABEL).mean(),
                "headline_count"   : len(grp),
            })
        result = pd.DataFrame(records)
        if not result.empty:
            result["date"] = pd.to_datetime(result["date"])
        return result

    @staticmethod
    def _resolve_device(device: str | None) -> str:
        if device:
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"
