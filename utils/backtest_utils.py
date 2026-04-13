from __future__ import annotations
import numpy as np
import pandas as pd
from config.constants import INITIAL_CAPITAL, TRANSACTION_COST
from utils.logging_utils import get_logger

log = get_logger(__name__)


def run_backtest(
    predictions: np.ndarray,
    actual_returns: np.ndarray,
    dates: pd.Series,
    initial_capital: float = INITIAL_CAPITAL,
    transaction_cost: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Simulate a long-only strategy driven by binary model predictions.

    Strategy rules:
        prediction == 1 (UP)   →  Buy at today's close, sell at tomorrow's close
        prediction == 0 (DOWN) →  Stay in cash

    Transaction cost is applied each time we enter or exit a position.

    Parameters
    ----------
    predictions      : array of 0/1 — model output for each test day
    actual_returns   : array of float — realised next-day returns for each day
    dates            : pd.Series or DatetimeIndex of corresponding dates
    initial_capital  : starting portfolio value in USD
    transaction_cost : fraction of trade value charged as cost (e.g. 0.001 = 0.1%)

    Returns
    -------
    pd.DataFrame with columns:
        date, prediction, actual_return, daily_return, strategy_cap,
        bh_cap, is_invested
    """
    n = len(predictions)
    if n == 0:
        log.warning("Empty predictions array — returning empty backtest DataFrame.")
        return pd.DataFrame()

    strategy_cap = np.zeros(n)
    bh_cap       = np.zeros(n)
    daily_returns = np.zeros(n)
    is_invested   = np.zeros(n, dtype=bool)

    cap = initial_capital
    bh  = initial_capital
    prev_invested = False

    for i in range(n):
        pred   = int(predictions[i])
        ret    = float(actual_returns[i])
        invest = pred == 1

        # Transaction cost when position changes
        cost = 0.0
        if invest != prev_invested:
            cost = transaction_cost

        # Daily return
        if invest:
            day_ret = ret - cost
        else:
            day_ret = -cost   # only cost if we exit a position

        cap = cap * (1 + day_ret)
        bh  = bh  * (1 + ret)

        strategy_cap[i]  = cap
        bh_cap[i]        = bh
        daily_returns[i] = day_ret
        is_invested[i]   = invest
        prev_invested    = invest

    bt_df = pd.DataFrame({
        "date"          : list(dates),
        "prediction"    : predictions.astype(int),
        "actual_return" : actual_returns,
        "daily_return"  : daily_returns,
        "strategy_cap"  : strategy_cap,
        "bh_cap"        : bh_cap,
        "is_invested"   : is_invested,
    })

    final_strat = strategy_cap[-1]
    final_bh    = bh_cap[-1]
    log.info(
        "Backtest complete — Strategy: $%.0f (%.1f%%)  |  B&H: $%.0f (%.1f%%)",
        final_strat, (final_strat / initial_capital - 1) * 100,
        final_bh,    (final_bh    / initial_capital - 1) * 100,
    )

    return bt_df
