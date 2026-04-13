# ── Market hours (US Eastern) ──────────────────────────────────
MARKET_OPEN_HOUR   = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR  = 16

# ── Price feature column names ─────────────────────────────────
PRICE_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "volatility_5d",
    "volatility_20d",
    "ma_5_vs_20",
    "ma_20_vs_50",
    "rsi_14",
    "macd_hist",
    "bb_pct",
    "intraday_range",
    "volume_ratio",
    "return_lag1",
    "return_lag2",
    "return_lag3",
    "return_lag5",
    "return_past_week",
]

# ── Sentiment feature column names ─────────────────────────────
SENTIMENT_FEATURES = [
    "sentiment_conf_wt",
    "sentiment_lag1",
    "sentiment_lag2",
    "sentiment_lag3",
    "sentiment_roll3",
    "sentiment_roll5",
    "sentiment_std",
    "pos_ratio",
    "neg_ratio",
    "headline_count",
]

# ── All features combined ──────────────────────────────────────
ALL_FEATURES = PRICE_FEATURES + SENTIMENT_FEATURES

# ── Column names ───────────────────────────────────────────────
TARGET_COL       = "target_direction"   # binary: 1=up, 0=down
TARGET_RETURN    = "target_return"      # continuous return
DATE_COL         = "date"
CLOSE_COL        = "Close"
OPEN_COL         = "Open"
HIGH_COL         = "High"
LOW_COL          = "Low"
VOLUME_COL       = "Volume"

META_COLS        = [DATE_COL, CLOSE_COL, TARGET_RETURN, TARGET_COL]

# ── FinBERT model ──────────────────────────────────────────────
FINBERT_MODEL_NAME    = "ProsusAI/finbert"
FINBERT_LABELS        = ["positive", "negative", "neutral"]
FINBERT_POS_LABEL     = "positive"
FINBERT_NEG_LABEL     = "negative"
FINBERT_NEU_LABEL     = "neutral"
FINBERT_MAX_LENGTH    = 512
FINBERT_DEFAULT_BATCH = 16

# Confidence tiers for headline weighting
CONF_HIGH   = 0.80
CONF_MEDIUM = 0.55

# ── Neutral sentiment fill values ──────────────────────────────
# Used when a trading day has zero headlines
SENTIMENT_NEUTRAL_FILL = {
    "sentiment_mean"    : 0.0,
    "sentiment_conf_wt" : 0.0,
    "sentiment_std"     : 0.0,
    "pos_ratio"         : 0.0,
    "neg_ratio"         : 0.0,
    "neu_ratio"         : 1.0,
    "headline_count"    : 0,
}

# ── RSS feed URLs (fallback when NewsData.io key is absent) ───
FINANCIAL_RSS_FEEDS = {
    "Reuters Business" : "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets"  : "https://feeds.reuters.com/reuters/markets",
    "CNBC Top News"    : "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "CNBC Finance"     : "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "MarketWatch"      : "http://feeds.marketwatch.com/marketwatch/topstories/",
    "Seeking Alpha"    : "https://seekingalpha.com/feed.xml",
    "Yahoo Finance"    : "https://finance.yahoo.com/rss/",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
}

# ── Ticker keyword maps (for RSS client-side filtering) ───────
TICKER_KEYWORDS: dict[str, list[str]] = {
    "AAPL": ["apple", "aapl", "tim cook", "iphone", "ipad", "macbook",
             "app store", "apple intelligence", "vision pro"],
    "MSFT": ["microsoft", "msft", "satya nadella", "azure", "copilot",
             "windows", "xbox", "linkedin"],
    "GOOGL": ["google", "googl", "alphabet", "sundar pichai", "gemini",
              "youtube", "waymo", "android"],
    "TSLA": ["tesla", "tsla", "elon musk", "cybertruck", "model 3",
             "model s", "model y", "autopilot", "giga"],
    "AMZN": ["amazon", "amzn", "andy jassy", "aws", "prime", "alexa",
             "whole foods", "kindle"],
    "META": ["meta", "facebook", "instagram", "whatsapp", "mark zuckerberg",
             "threads", "oculus", "llama"],
    "NVDA": ["nvidia", "nvda", "jensen huang", "cuda", "geforce", "h100",
             "blackwell", "gpu"],
}

# ── Model file names ────────────────────────────────────────────
XGB_MODEL_FILE   = "xgboost_model.pkl"
LSTM_MODEL_FILE  = "lstm_model.pt"
SCALER_FILE      = "feature_scaler.pkl"
FEATURE_FILE_TPL = "{ticker}_feature_matrix.csv"

# ── Technical indicator windows ────────────────────────────────
RSI_WINDOW     = 14
MACD_FAST      = 12
MACD_SLOW      = 26
MACD_SIGNAL    = 9
BB_WINDOW      = 20
BB_STD         = 2
MA_SHORT       = 5
MA_MID         = 20
MA_LONG        = 50
VOLUME_MA      = 20

# ── Backtesting ─────────────────────────────────────────────────
INITIAL_CAPITAL  = 10_000.0
TRANSACTION_COST = 0.001      # 0.1 % per trade
RISK_FREE_RATE   = 0.05       # annual, for Sharpe ratio
TRADING_DAYS_PER_YEAR = 252

# ── API limits ──────────────────────────────────────────────────
NEWSDATA_FREE_DAILY_LIMIT = 200
NEWSDATA_MAX_PAGES_DEFAULT = 3
NEWSDATA_REQUEST_DELAY     = 1.0   # seconds between requests
