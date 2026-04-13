# ──────────────────────────────────────────────────────────────
#  utils/evaluation_utils.py
#  Classification + financial performance metric helpers.
#  Both XGBoost and LSTM trainers call these to stay DRY.
# ──────────────────────────────────────────────────────────────

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from config.constants import INITIAL_CAPITAL, RISK_FREE_RATE, TRADING_DAYS_PER_YEAR
from utils.logging_utils import get_logger

log = get_logger(__name__)


# ── Classification metrics ──────────────────────────────────────

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    label: str = "Model",
) -> dict:
    """
    Compute accuracy, AUC-ROC, and a full classification report.

    Returns
    -------
    dict with keys: accuracy, auc, report, confusion_matrix
    """
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=["Down", "Up"], output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    result: dict = {
        "label"           : label,
        "accuracy"        : acc,
        "auc"             : None,
        "report"          : report,
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        try:
            result["auc"] = roc_auc_score(y_true, y_prob)
        except ValueError:
            log.warning("[%s] AUC computation failed (single class in y_true?)", label)

    log.info("[%s] Accuracy=%.4f  AUC=%s", label, acc,
             f"{result['auc']:.4f}" if result["auc"] else "N/A")

    return result


def print_metrics_table(metrics_dict: dict[str, dict]) -> None:
    """Pretty-print a comparison table for multiple models."""
    header = f"{'Model':<28} {'Accuracy':>10} {'AUC-ROC':>10}"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    for name, m in metrics_dict.items():
        auc_str = f"{m['auc']:.4f}" if m.get("auc") else "  N/A  "
        print(f"{name:<28} {m['accuracy']:>10.4f} {auc_str:>10}")
    print("=" * len(header))


# ── Financial / backtesting metrics ────────────────────────────

def compute_financial_metrics(
    bt_df: pd.DataFrame,
    name: str = "Strategy",
) -> dict:
    """
    Compute standard quant finance metrics from a backtest DataFrame.

    Expected columns in bt_df:
        strategy_cap : portfolio value each day under the model strategy
        bh_cap       : portfolio value under buy-and-hold benchmark
        is_invested  : bool — whether the strategy was in the market that day

    Returns
    -------
    dict with keys:
        total_return, annualised_return, max_drawdown,
        sharpe_ratio, win_rate, bh_total_return
    """
    if bt_df.empty:
        log.warning("[%s] Backtest DataFrame is empty.", name)
        return {}

    equity = bt_df["strategy_cap"].values

    # ── Total return ──────────────────────────────────────────
    total_return = (equity[-1] / equity[0]) - 1.0

    # ── Annualised return ─────────────────────────────────────
    n_days = len(equity)
    annualised_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / n_days) - 1

    # ── Max drawdown ──────────────────────────────────────────
    rolling_max = np.maximum.accumulate(equity)
    drawdowns   = (equity - rolling_max) / rolling_max
    max_drawdown = float(drawdowns.min())

    # ── Sharpe ratio ──────────────────────────────────────────
    daily_returns = np.diff(equity) / equity[:-1]
    excess_return = daily_returns - RISK_FREE_RATE / TRADING_DAYS_PER_YEAR
    sharpe = (
        np.sqrt(TRADING_DAYS_PER_YEAR) * excess_return.mean() / (excess_return.std() + 1e-9)
    )

    # ── Win rate (% of invested days that were profitable) ────
    if "is_invested" in bt_df.columns and "daily_return" in bt_df.columns:
        invested = bt_df[bt_df["is_invested"]]
        win_rate = (invested["daily_return"] > 0).mean() if len(invested) > 0 else float("nan")
    else:
        win_rate = float("nan")

    # ── Buy-and-hold total return ─────────────────────────────
    bh_return = (bt_df["bh_cap"].iloc[-1] / bt_df["bh_cap"].iloc[0]) - 1.0

    metrics = {
        "label"            : name,
        "total_return"     : total_return,
        "annualised_return": annualised_return,
        "max_drawdown"     : max_drawdown,
        "sharpe_ratio"     : sharpe,
        "win_rate"         : win_rate,
        "bh_total_return"  : bh_return,
    }

    log.info(
        "[%s] Return=%.2f%%  Ann=%.2f%%  MaxDD=%.2f%%  Sharpe=%.2f  WinRate=%.1f%%",
        name,
        total_return * 100,
        annualised_return * 100,
        max_drawdown * 100,
        sharpe,
        win_rate * 100 if not np.isnan(win_rate) else float("nan"),
    )

    return metrics
