import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def plot_port_value_train(port_hist: list[float], dates: pd.Series) -> None:
    """
    Plot the portfolio value over time for the training set.

    Args:
        port_hist (list[float]): Portfolio values at each time step.
        dates (pd.Series): Corresponding datetime values.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(dates, port_hist, color='maroon')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(linestyle=':', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_port_value_test_val(test_hist: list[float], test_dates: pd.Series,
                             val_hist: list[float], val_dates: pd.Series) -> None:
    """
    Plot portfolio value over Test and Validation sets as a continuous series.

    The Validation curve is shifted to start from the final value of the Test curve
    to ensure visual continuity.

    Args:
        test_hist (list[float]): Portfolio values during the test period.
        test_dates (pd.Series): Corresponding datetime values for the test period.
        val_hist (list[float]): Portfolio values during the validation period.
        val_dates (pd.Series): Corresponding datetime values for the validation period.
    """
    plt.figure(figsize=(12, 5))

    # Plot Test
    plt.plot(test_dates, test_hist, color='maroon', label='Test', alpha=0.5)

    # Shift Validation to start from last Test value
    shift = test_hist[-1] - val_hist[0]
    val_hist_adjusted = [v + shift for v in val_hist]
    plt.plot(val_dates, val_hist_adjusted,
             color='palevioletred', label='Validation')

    plt.title("Portfolio Value: Test + Validation")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_return_distribution(port_series: pd.Series, title="Portfolio Return Distribution", bins=30) -> None:
    """
    Plot the distribution of portfolio returns as a histogram.

    Args:
        port_series (pd.Series): Time series of portfolio values (index must be datetime-like).
        title (str): Title of the plot.
        bins (int): Number of histogram bins.
    """

    # --- Ensure datetime index and resample monthly ---
    port_series = port_series.sort_index()

    # Resample to end-of-month values
    monthly_values = port_series.resample('ME').last()

    # Compute monthly returns
    monthly_returns = monthly_values.pct_change().dropna()

    # Plot
    plt.figure(figsize=(10, 5))
    sns.histplot(monthly_returns, bins=bins, kde=False, color='palevioletred',
                 stat='density')

    plt.title(title)
    plt.xlabel("Monthly Returns")
    plt.ylabel("Density")
    plt.grid(linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()


def plot_rolling_volatility(port_series: pd.Series, window: int = 60) -> None:
    """
    Plots rolling volatility of monthly returns.

    Parameters:
    - returns: pd.Series of periodic returns (index = datetime)
    - window: rolling window size (default=60 periods)
    """

    # port_series = port_series.sort_index()
    # port_series = port_series.resample('M').last()
    returns = port_series.pct_change().dropna()
    rolling_vol = returns.rolling(window).std()

    plt.figure(figsize=(10, 5))
    plt.plot(
        rolling_vol, label=f'{window}-period Rolling Volatility', color='lightcoral')
    plt.title(f'Rolling Volatility ({window}-period)')
    plt.xlabel('Date')
    plt.ylabel('Volatility (Std. Dev.)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()


def plot_signals(df: pd.DataFrame, buy_signals: pd.Series, sell_signals: pd.Series) -> None:
    """
    Overlays buy/sell signals on price chart.

    Parameters:
    - price: pd.Series of asset/portfolio prices (index = datetime)
    - buy_signals: list or pd.Series of booleans (True where buy occurs)
    - sell_signals: list or pd.Series of booleans (True where sell occurs)
    """
    plt.figure(figsize=(15, 5))

    plt.plot(df.index, df['Close'], label='Price', color='black', linewidth=1)

    # Overlay buy/sell markers
    plt.scatter(df.index[buy_signals], df['Close'][buy_signals],
                label='Buy', marker='^', color='darkseagreen', s=80)
    plt.scatter(df.index[sell_signals], df['Close'][sell_signals],
                label='Sell', marker='v', color='indianred', s=80)

    plt.title('Buy/Sell Points on Price Chart')
    plt.xlabel('Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.grid(linestyle=':', alpha=0.5)
    plt.show()
