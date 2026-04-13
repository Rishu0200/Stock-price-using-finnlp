from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.config import settings
from config.constants import ALL_FEATURES, FEATURE_FILE_TPL
from utils.logging_utils import get_logger

log = get_logger(__name__, log_file=settings.paths.logs_dir / "app.log")

# ── FastAPI app ───────────────────────────────────────────────

app = FastAPI(
    title       = "Stock Predictor FinBERT",
    description = "Next-day stock direction prediction using FinBERT sentiment + XGBoost/LSTM",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Pydantic schemas ──────────────────────────────────────────

class PredictionResponse(BaseModel):
    ticker:      str
    as_of_date:  str
    direction:   str
    probability: float
    model_type:  str


class HealthResponse(BaseModel):
    status:  str
    version: str


class TrainRequest(BaseModel):
    ticker:     str  = "AAPL"
    start_date: str  = "2023-01-01"
    end_date:   str  = "2024-12-31"
    model_type: str  = "xgboost"
    tune_hp:    bool = False


class TrainResponse(BaseModel):
    ticker:       str
    model_type:   str
    accuracy:     float
    auc:          float | None
    message:      str


# ── Routes ────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def root():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(
    ticker:     str = Query(default="AAPL", description="Stock ticker symbol"),
    model_type: str = Query(default="xgboost", description="xgboost or lstm"),
    lookback:   int = Query(default=90, ge=30, le=365, description="Lookback days for data fetch"),
):
    """
    Fetch live data and predict next-day direction for the given ticker.
    Returns direction (UP/DOWN) and probability.
    """
    try:
        from src.prediction_pipeline.predictor import StockPredictor
        predictor = StockPredictor(ticker=ticker, model_type=model_type)
        result    = predictor.predict(lookback_days=lookback)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error("Prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/predict/batch", tags=["Prediction"])
def predict_batch(
    ticker:     str = Query(default="AAPL"),
    model_type: str = Query(default="xgboost"),
    n_rows:     int = Query(default=10, ge=1, le=100),
):
    """
    Run predictions on the saved feature matrix and return the last n_rows.
    """
    try:
        from src.prediction_pipeline.predictor import StockPredictor
        predictor = StockPredictor(ticker=ticker, model_type=model_type)
        df = predictor.predict_from_file(n_rows=n_rows)
        return df.to_dict(orient="records")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.error("Batch prediction error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse, tags=["Training"])
def train(req: TrainRequest):
    """
    Trigger the full training pipeline:
    data ingestion → feature engineering → model training → save.
    """
    try:
        results = run_training_pipeline(
            ticker     = req.ticker,
            start_date = req.start_date,
            end_date   = req.end_date,
            model_type = req.model_type,
            tune_hp    = req.tune_hp,
        )
        metrics = results.get("eval_metrics", {})
        return {
            "ticker"    : req.ticker,
            "model_type": req.model_type,
            "accuracy"  : metrics.get("accuracy", 0.0),
            "auc"       : metrics.get("auc"),
            "message"   : "Training complete.",
        }
    except Exception as e:
        log.error("Training error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ── Training pipeline function ────────────────────────────────

def run_training_pipeline(
    ticker:     str  = "AAPL",
    start_date: str  = "2023-01-01",
    end_date:   str  = "2024-12-31",
    model_type: str  = "xgboost",
    tune_hp:    bool = False,
) -> dict:
    """
    Full training pipeline:
      1. Ingest stock prices
      2. Ingest news headlines
      3. FinBERT sentiment
      4. Feature engineering
      5. Merge features
      6. [Optional] Hyperparameter tuning
      7. Train model
      8. Save model + scaler
    """
    import pickle

    from src.data_ingestion.news_fetcher import NewsFetcher
    from src.data_ingestion.stock_fetcher import StockFetcher
    from src.data_transformation.data_merger import DataMerger
    from src.data_transformation.feature_engineering import FeatureEngineer
    from src.data_transformation.sentiment_pipeline import FinBERTPipeline
    from utils.date_utils import get_trading_days

    log.info("=" * 60)
    log.info("Training pipeline: %s | %s → %s | model=%s",
             ticker, start_date, end_date, model_type)
    log.info("=" * 60)

    # ── Step 1: Stock prices ──────────────────────────────────
    stock    = StockFetcher(ticker=ticker, start=start_date, end=end_date)
    ohlcv_df = stock.fetch()
    stock.save(ohlcv_df)

    # ── Step 2: News headlines ────────────────────────────────
    news_fetcher = NewsFetcher(ticker=ticker)
    news_df      = news_fetcher.fetch()
    if not news_df.empty:
        news_fetcher.save(news_df)

    # ── Step 3: FinBERT sentiment ─────────────────────────────
    trading_days = get_trading_days(start_date, end_date)
    merger       = DataMerger(ticker=ticker)
    sentiment_df = _get_sentiment(news_df, trading_days, merger)

    # ── Step 4: Price feature engineering ─────────────────────
    fe           = FeatureEngineer()
    price_feats  = fe.transform(ohlcv_df)

    # ── Step 5: Merge features ────────────────────────────────
    feature_matrix = merger.merge(price_feats, sentiment_df)
    merger.save(feature_matrix)

    # ── Step 6: Optional HP tuning ────────────────────────────
    tuned_params = {}
    if tune_hp:
        tuned_params = _run_hp_tuning(feature_matrix, model_type)

    # ── Step 7: Train model ───────────────────────────────────
    results = _train_model(feature_matrix, model_type, tuned_params)

    log.info("Pipeline complete. Accuracy=%.4f",
             results.get("eval_metrics", {}).get("accuracy", 0))
    return results


def _get_sentiment(news_df, trading_days, merger) -> "pd.DataFrame":
    import pandas as pd
    if news_df.empty:
        log.warning("No news headlines — using neutral sentiment.")
        return pd.DataFrame()
    aligned = merger.align_news(news_df, trading_days)
    if aligned.empty:
        return pd.DataFrame()
    finbert = FinBERTPipeline()
    return finbert.run(aligned, text_col="headline", date_col="trade_date")


def _run_hp_tuning(feature_matrix, model_type: str) -> dict:
    from config.constants import ALL_FEATURES, TARGET_COL
    from sklearn.preprocessing import StandardScaler

    available = [c for c in ALL_FEATURES if c in feature_matrix.columns]
    X = StandardScaler().fit_transform(feature_matrix[available].values)
    y = feature_matrix[TARGET_COL].values
    split = int(len(X) * 0.8)

    if model_type == "xgboost":
        from src.hyperparameter_tuning.xgb_tuner import XGBTuner
        tuner = XGBTuner(n_trials=40)
        return tuner.tune(X[:split], y[:split])
    elif model_type == "lstm":
        from src.hyperparameter_tuning.lstm_tuner import LSTMTuner
        tuner = LSTMTuner(n_trials=20, epochs_per_trial=10)
        return tuner.tune(X[:split], y[:split], X[split:], y[split:])
    return {}


def _train_model(feature_matrix, model_type: str, params: dict) -> dict:
    if model_type == "xgboost":
        from src.model_trainer.xgboost_trainer import XGBoostTrainer
        trainer = XGBoostTrainer(params=params if params else None)
    elif model_type == "lstm":
        from src.model_trainer.lstm_trainer import LSTMTrainer
        trainer = LSTMTrainer(model_params=params if params else None)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    results = trainer.fit(feature_matrix)
    trainer.save()
    return results


# ── CLI entrypoint ────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(description="Stock Predictor FinBERT")
    parser.add_argument("--mode",       choices=["train", "predict", "serve"], default="serve")
    parser.add_argument("--ticker",     default="AAPL")
    parser.add_argument("--start",      default="2023-01-01")
    parser.add_argument("--end",        default="2024-12-31")
    parser.add_argument("--model",      choices=["xgboost", "lstm"], default="xgboost")
    parser.add_argument("--tune",       action="store_true")
    parser.add_argument("--host",       default="0.0.0.0")
    parser.add_argument("--port",       type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "train":
        run_training_pipeline(
            ticker     = args.ticker,
            start_date = args.start,
            end_date   = args.end,
            model_type = args.model,
            tune_hp    = args.tune,
        )

    elif args.mode == "predict":
        from src.prediction_pipeline.predictor import StockPredictor
        predictor = StockPredictor(ticker=args.ticker, model_type=args.model)
        result = predictor.predict()
        print("\n" + "=" * 50)
        print(f"  Ticker     : {result['ticker']}")
        print(f"  As of      : {result['as_of_date']}")
        print(f"  Direction  : {result['direction']}")
        print(f"  Probability: {result['probability']:.4f}")
        print(f"  Model      : {result['model_type']}")
        print("=" * 50 + "\n")

    elif args.mode == "serve":
        import uvicorn
        uvicorn.run("main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    _cli()
