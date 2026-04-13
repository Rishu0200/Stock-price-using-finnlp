# 📈 Stock Predictor FinBERT

> **Next-day stock direction prediction** combining FinBERT sentiment analysis on real financial news with XGBoost and LSTM classifiers.

---

## 🗺️ Project Structure

```
stock_predictor_finbert/
│
├── main.py                            # FastAPI app + CLI entrypoint
├── requirements.txt
├── Dockerfile                         # Multi-stage build for Render
├── docker-compose.yml                 # Local development
├── render.yaml                        # Render IaC config
├── .env                               # Environment variables (never commit)
├── .gitignore
│
├── config/
│   ├── config.py                      # Runtime settings (loaded from .env)
│   └── constants.py                   # Feature names, thresholds, column maps
│
├── src/
│   ├── data_ingestion/
│   │   ├── stock_fetcher.py           # yfinance OHLCV downloader
│   │   └── news_fetcher.py            # NewsData.io + RSS fallback
│   │
│   ├── data_transformation/
│   │   ├── feature_engineering.py     # Returns, MA, RSI, MACD, BB, lag features
│   │   ├── sentiment_pipeline.py      # FinBERT batch inference → daily scores
│   │   └── data_merger.py             # Date alignment + price/sentiment merge
│   │
│   ├── model_trainer/
│   │   ├── base_trainer.py            # Abstract base: split, scale, CV, evaluate
│   │   ├── xgboost_trainer.py         # XGBoost classifier
│   │   └── lstm_trainer.py            # Bidirectional LSTM with sequences
│   │
│   ├── hyperparameter_tuning/
│   │   ├── xgb_tuner.py              # Optuna / RandomizedSearchCV for XGBoost
│   │   └── lstm_tuner.py             # Optuna for LSTM architecture search
│   │
│   └── prediction_pipeline/
│       └── predictor.py              # End-to-end real-time + batch inference
│
├── utils/
│   ├── logging_utils.py              # Centralised logger factory
│   ├── date_utils.py                 # Trading-day alignment helpers
│   ├── evaluation_utils.py           # Metrics: accuracy, AUC, Sharpe, drawdown
│   └── backtest_utils.py             # Long-only backtest simulation
│
├── data/
│   ├── raw/                          # OHLCV CSVs, raw news CSVs
│   └── processed/                    # Intermediate cleaned data
│
├── models/                           # Saved model artifacts (.pkl, .pt)
├── artifacts/                        # Feature matrices (AAPL_feature_matrix.csv)
└── logs/                             # Application logs
```

---

## ⚡ Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/your-org/stock-predictor-finbert.git
cd stock-predictor-finbert
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure `.env`

```bash
cp .env .env.example   # already provided — fill in your values
```

| Variable | Required | Description |
|---|---|---|
| `NEWSDATA_API_KEY` | Optional | NewsData.io key (200 req/day free). Falls back to RSS if absent. |
| `TICKER` | Yes | Stock ticker, e.g. `AAPL` |
| `START_DATE` | Yes | Training start date `YYYY-MM-DD` |
| `END_DATE` | Yes | Training end date `YYYY-MM-DD` |
| `MODEL_TYPE` | Yes | `xgboost` or `lstm` |

### 3. Train the model

```bash
# Basic training (no hyperparameter tuning)
python main.py --mode train --ticker AAPL --model xgboost

# With Optuna hyperparameter tuning (takes longer)
python main.py --mode train --ticker AAPL --model xgboost --tune
```

### 4. Predict next-day direction

```bash
python main.py --mode predict --ticker AAPL --model xgboost
```

Output:
```
==================================================
  Ticker     : AAPL
  As of      : 2024-06-04
  Direction  : UP
  Probability: 0.6231
  Model      : xgboost
==================================================
```

### 5. Serve the REST API

```bash
python main.py --mode serve
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at `http://localhost:8000/docs`

---

## 🐳 Docker

```bash
# Build and run locally
docker compose up --build

# Run training inside container
docker compose run app python main.py --mode train --ticker AAPL

# Single prediction
docker compose run app python main.py --mode predict --ticker AAPL
```

---

## 🚀 Deploy to Render

1. Push this repo to GitHub.
2. Go to [render.com](https://render.com) → **New → Web Service**.
3. Connect your GitHub repo. Render auto-detects `render.yaml`.
4. Set secret env vars in the Render dashboard:
   - `NEWSDATA_API_KEY` — your NewsData.io key
5. Click **Deploy**. Render builds the Docker image and serves the API.

The `/health` endpoint is used as the health check — Render will wait for it to return `200 OK` before routing traffic.

---

## 🌐 REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `GET` | `/predict?ticker=AAPL&model_type=xgboost` | Live next-day prediction |
| `GET` | `/predict/batch?ticker=AAPL&n_rows=10` | Last N predictions from saved matrix |
| `POST` | `/train` | Trigger full training pipeline |
| `GET` | `/docs` | Swagger UI |

### Example prediction response

```json
{
  "ticker":      "AAPL",
  "as_of_date":  "2024-06-04",
  "direction":   "UP",
  "probability": 0.6231,
  "model_type":  "xgboost"
}
```

---

## 🧱 Pipeline Architecture

```
yfinance (OHLCV)
        │
        ▼
FeatureEngineer
  returns, MA, RSI, MACD, Bollinger Bands, volume ratio, lags
        │
        ▼                    NewsData.io / RSS
DataMerger ◄─────────────── NewsFetcher
  date alignment                │
  left-join on date             ▼
  fill neutral defaults    FinBERTPipeline
  sentiment lag features     batch inference
        │                    daily aggregation
        ▼
Feature Matrix (CSV)
        │
        ├──► XGBoostTrainer  ──► TimeSeriesSplit CV ──► backtest
        │
        └──► LSTMTrainer     ──► sliding window seqs ──► backtest
                │
                ▼
            StockPredictor (inference)
                │
                ▼
            FastAPI (REST)
                │
                ▼
            Render (cloud)
```

---

## 📐 Features Engineered

### Price Features (17)
`return_1d`, `return_3d`, `return_5d`, `volatility_5d`, `volatility_20d`,
`ma_5_vs_20`, `ma_20_vs_50`, `rsi_14`, `macd_hist`, `bb_pct`,
`intraday_range`, `volume_ratio`, `return_lag1–3`, `return_lag5`, `return_past_week`

### Sentiment Features (10)
`sentiment_conf_wt`, `sentiment_lag1–3`, `sentiment_roll3`, `sentiment_roll5`,
`sentiment_std`, `pos_ratio`, `neg_ratio`, `headline_count`

---

## 🔑 Key Design Decisions

- **No look-ahead bias** — news published after market close is aligned to the *next* trading day, not the same day (Phase 5 rule).
- **TimeSeriesSplit** — all cross-validation uses chronological folds; data is never shuffled.
- **Neutral fill** — trading days with zero headlines get `sentiment_conf_wt = 0.0` (neutral), not dropped.
- **Lazy model loading** — FinBERT model is only loaded when needed (saves RAM on startup).
- **Scaler persistence** — the `StandardScaler` fitted on training data is saved alongside the model to prevent train/serve skew.

---

## 📦 Tech Stack

| Component | Library |
|---|---|
| Stock prices | `yfinance` |
| News (primary) | `newsdataapi` (NewsData.io) |
| News (fallback) | `feedparser` (RSS) |
| Sentiment NLP | `transformers` + `ProsusAI/finbert` |
| Technical indicators | `ta` |
| Gradient boosting | `xgboost` |
| Deep learning | `torch` (LSTM) |
| HP tuning | `optuna` |
| API framework | `fastapi` + `uvicorn` |
| Deployment | Docker + Render |