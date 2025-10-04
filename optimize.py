import optuna
import pandas as pd
import numpy as np

from backtesting import backtest
from indicators import get_signals, add_indicators
from metrics import calmar_ratio


def optimize(trial: optuna.Trial, train_data: pd.DataFrame) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    """

    data = train_data.copy()

    rsi_window = trial.suggest_int("rsi_window", 5, 30)
    sma_window = trial.suggest_int("sma_window", 5, 50)
    bb_window = trial.suggest_int("bb_window", 10, 40)
    bb_dev = trial.suggest_float("bb_dev", 1.0, 3.0)

    rsi_buy = trial.suggest_int("rsi_buy", 10, 40)
    rsi_sell = trial.suggest_int("rsi_sell", 60, 90)

    # --- Trade hyperparameters ---
    sl = trial.suggest_float("SL", 0.01, 0.2,)
    tp = trial.suggest_float("TP", 0.01, 0.2)
    n_shares = trial.suggest_float("n_shares", 0.1, 10)


    # --- Add indicators with params ---
    data = add_indicators(
        data,
        rsi_window=rsi_window,
        sma_window=sma_window,
        bb_window=bb_window,
        bb_dev=bb_dev
    )
    data = get_signals(
        data,
        rsi_buy=rsi_buy,
        rsi_sell=rsi_sell
    )

    # --- Cross-validation ---
    n_splits = 7
    calmars = []
    len_data = len(data)

    for i in range(n_splits):
        size = len_data // n_splits
        start_idx = i * size
        end_idx = (i + 1) * size

        chunk = data.iloc[start_idx:end_idx, :]
        port_vals, _ = backtest(chunk, sl, tp, n_shares)

        port_series = pd.Series(port_vals)
        returns = port_series.pct_change().dropna()

        calmar = calmar_ratio(returns, periods_per_year=8760) 
        calmars.append(calmar)

    mean_calmar = np.mean(calmars)

    # If mean_calmar is NaN, assign very low value
    if np.isnan(mean_calmar):
        return -1e6

    return mean_calmar