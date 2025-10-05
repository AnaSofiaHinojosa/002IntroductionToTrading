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
    plt.plot(val_dates, val_hist_adjusted, color='palevioletred', label='Validation')

    plt.title("Portfolio Value: Test + Validation")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(linestyle=':', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_return_distribution(port_series: pd.Series, title="Return Distribution", bins=50, overlay_normal=True):
    """
    Plot the distribution of portfolio returns as a histogram.

    Optionally overlays a normal probability density function (PDF) for comparison,
    and displays skewness and kurtosis in the legend.

    Args:
        port_series (pd.Series): Time series of portfolio values.
        title (str): Title of the plot.
        bins (int): Number of histogram bins.
        overlay_normal (bool): Whether to overlay a normal distribution curve.
    """
    returns = port_series.pct_change().dropna()
    mu, sigma = returns.mean(), returns.std()
    
    plt.figure(figsize=(10,5))
    sns.histplot(returns, bins=bins, kde=False, color='palevioletred', edgecolor='black', stat='density')
    
    if overlay_normal:
        x = np.linspace(returns.min(), returns.max(), 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), color='maroon', lw=2, label='Normal PDF')
    
    plt.title(title)
    plt.xlabel("Returns")
    plt.ylabel("Density")
    plt.grid(linestyle=':', alpha=0.5)
    
    skewness = stats.skew(returns)
    kurt = stats.kurtosis(returns)
    plt.legend(title=f"Skew: {skewness:.2f}, Kurtosis: {kurt:.2f}")
    
    plt.tight_layout()
    plt.show()


def plot_returns_heatmap(port_series: pd.Series, freq='M', title="Returns Heatmap"):
    """
    Plot a heatmap of average returns grouped by month or quarter and year.

    Positive returns are shown in green, negative in red. Useful for visualizing
    seasonal or cyclical patterns in performance.

    Args:
        port_series (pd.Series): Time series of portfolio values.
        freq (str): Frequency for grouping ('M' for month, 'Q' for quarter).
        title (str): Title of the heatmap.
    """
    # Ensure DatetimeIndex
    port_series = port_series.copy()
    if not isinstance(port_series.index, pd.DatetimeIndex):
        port_series.index = pd.to_datetime(port_series.index)
    
    returns = port_series.pct_change().dropna()
    
    if freq == 'M':
        grouped = returns.groupby([returns.index.year, returns.index.month]).mean().unstack()
        xlabel = "Month"
    elif freq == 'Q':
        grouped = returns.groupby([returns.index.year, returns.index.quarter]).mean().unstack()
        xlabel = "Quarter"
    
    plt.figure(figsize=(12,6))
    sns.heatmap(grouped, annot=True, fmt=".4f", center=0, cmap="RdYlGn", linewidths=0.5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Year")
    plt.tight_layout()
    plt.show()
