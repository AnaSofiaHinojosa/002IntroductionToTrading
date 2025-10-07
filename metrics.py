import pandas as pd
import numpy as np

# --- Individual metrics ---

# Sharpe Ratio


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    """
    Calculate the annualized Sharpe Ratio.

    Args:
        returns (pd.Series): Periodic returns.
        risk_free_rate (float): Annual risk-free rate.
        periods_per_year (int): Number of periods per year.

    Returns:
        float: Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = np.mean(excess_returns) * periods_per_year
    ann_vol = np.std(excess_returns, ddof=1) * \
        np.sqrt(periods_per_year)  # ddof=1 for sample std
    return ann_return / ann_vol if ann_vol != 0 else np.nan

# Sortino ratio


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    """
    Calculate the annualized Sortino Ratio.

    Args:
        returns (pd.Series): Periodic returns.
        risk_free_rate (float): Annual risk-free rate.
        periods_per_year (int): Number of periods per year.

    Returns:
        float: Sortino Ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    ann_return = np.mean(excess_returns) * periods_per_year
    downside = returns[returns < 0]
    downside_vol = np.std(downside, ddof=1) * np.sqrt(periods_per_year)
    return ann_return / downside_vol if downside_vol != 0 else np.nan

# Calmar ratio


def calmar_ratio(returns: pd.Series, periods_per_year: int = 8760) -> float:
    """
    Calculate the Calmar Ratio.

    Args:
        returns (pd.Series): Periodic returns.
        periods_per_year (int): Number of periods per year.

    Returns:
        float: Calmar Ratio.
    """
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdowns = (cum_returns - peak) / peak
    max_dd = drawdowns.min()
    ann_return = np.mean(returns) * periods_per_year
    return ann_return / abs(max_dd) if max_dd != 0 else np.nan

# Maximum Drawdown


def max_drawdown(returns: pd.Series) -> float:
    """
    Calculate the maximum drawdown.

    Args:
        returns (pd.Series): Periodic returns.

    Returns:
        float: Maximum drawdown.
    """
    cum_returns = (1 + returns).cumprod()
    peak = cum_returns.cummax()
    drawdowns = (cum_returns - peak) / peak
    return drawdowns.min()

# Win Rate


def win_rate(returns: pd.Series) -> float:
    """
    Calculate the win rate (fraction of positive returns).

    Args:
        returns (pd.Series): Periodic returns.

    Returns:
        float: Win rate.
    """
    return (returns > 0).mean()

# Metrics summary


def performance_summary(prices: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> dict:
    """
    Compute a summary of performance metrics from a price series.

    Args:
        prices (pd.Series): Time series of prices.
        risk_free_rate (float): Annual risk-free rate.
        periods_per_year (int): Number of periods per year.

    Returns:
        dict: Dictionary of performance metrics.
    """
    returns = prices.pct_change().dropna()
    metrics = {
        "Sharpe Ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "Sortino Ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "Calmar Ratio": calmar_ratio(returns, periods_per_year),
        "Maximum Drawdown": max_drawdown(returns),
        "Win Rate": win_rate(returns)
    }
    return metrics
