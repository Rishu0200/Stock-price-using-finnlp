from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


# ─────────────────────────────────────────────────────────────
def _get(key: str, default: str = "") -> str:
    """Helper: get env var with a fallback default."""
    return os.getenv(key, default)


def _get_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, default))


def _get_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, default))


# ─────────────────────────────────────────────────────────────
@dataclass
class APIConfig:
    newsdata_api_key: str = field(default_factory=lambda: _get("NEWSDATA_API_KEY"))


@dataclass
class StockConfig:
    ticker: str     = field(default_factory=lambda: _get("TICKER", "AAPL"))
    start_date: str = field(default_factory=lambda: _get("START_DATE", "2023-01-01"))
    end_date: str   = field(default_factory=lambda: _get("END_DATE", "2024-12-31"))


@dataclass
class ModelConfig:
    model_type: str  = field(default_factory=lambda: _get("MODEL_TYPE", "xgboost"))
    train_ratio: float = field(default_factory=lambda: _get_float("TRAIN_RATIO", 0.80))
    random_seed: int   = field(default_factory=lambda: _get_int("RANDOM_SEED", 42))
    seq_len: int       = field(default_factory=lambda: _get_int("SEQ_LEN", 10))

    # ── XGBoost defaults ──────────────────────────────────────
    xgb_n_estimators: int     = 300
    xgb_max_depth: int        = 4
    xgb_learning_rate: float  = 0.05
    xgb_subsample: float      = 0.8
    xgb_colsample: float      = 0.8
    xgb_min_child_weight: int = 3
    xgb_gamma: float          = 0.1
    xgb_reg_alpha: float      = 0.1
    xgb_reg_lambda: float     = 1.0

    # ── LSTM defaults ─────────────────────────────────────────
    lstm_hidden_size: int  = 64
    lstm_num_layers: int   = 2
    lstm_dropout: float    = 0.3
    lstm_bidirectional: bool = False
    lstm_lr: float         = 1e-3
    lstm_weight_decay: float = 1e-4
    lstm_epochs: int       = 40
    lstm_batch_size: int   = 32

    # ── Cross-validation ──────────────────────────────────────
    n_cv_splits: int = 5


@dataclass
class FinBERTConfig:
    model_name: str    = field(default_factory=lambda: _get("FINBERT_MODEL", "ProsusAI/finbert"))
    batch_size: int    = field(default_factory=lambda: _get_int("FINBERT_BATCH_SIZE", 16))
    conf_high: float   = field(default_factory=lambda: _get_float("FINBERT_CONF_HIGH", 0.80))
    conf_medium: float = field(default_factory=lambda: _get_float("FINBERT_CONF_MEDIUM", 0.55))
    max_length: int    = 512


@dataclass
class AppConfig:
    host: str    = field(default_factory=lambda: _get("APP_HOST", "0.0.0.0"))
    port: int    = field(default_factory=lambda: _get_int("APP_PORT", 8000))
    env: str     = field(default_factory=lambda: _get("APP_ENV", "development"))
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))


@dataclass
class PathConfig:
    root: Path       = ROOT_DIR
    data_dir: Path   = field(default_factory=lambda: ROOT_DIR / _get("DATA_DIR", "data"))
    models_dir: Path = field(default_factory=lambda: ROOT_DIR / _get("MODELS_DIR", "models"))
    artifacts_dir: Path = field(default_factory=lambda: ROOT_DIR / _get("ARTIFACTS_DIR", "artifacts"))
    logs_dir: Path   = field(default_factory=lambda: ROOT_DIR / _get("LOGS_DIR", "logs"))

    def __post_init__(self):
        """Create directories if they do not exist."""
        for d in [self.data_dir / "raw", self.data_dir / "processed",
                  self.models_dir, self.artifacts_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ── Singleton settings object ─────────────────────────────────
@dataclass
class Settings:
    api: APIConfig      = field(default_factory=APIConfig)
    stock: StockConfig  = field(default_factory=StockConfig)
    model: ModelConfig  = field(default_factory=ModelConfig)
    finbert: FinBERTConfig = field(default_factory=FinBERTConfig)
    app: AppConfig      = field(default_factory=AppConfig)
    paths: PathConfig   = field(default_factory=PathConfig)


settings = Settings()
