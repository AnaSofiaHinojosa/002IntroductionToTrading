import matplotlib.pyplot as plt
import pandas as pd

def plot_port_value_train(port_hist: list[float], dates: pd.Series) -> None:
    """
    Plot the portfolio value over time.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(dates, port_hist, color='maroon')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(linestyle=':', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_port_value_test_val(test_hist: list[float], test_dates: pd.Series,
                             val_hist: list[float], val_dates: pd.Series) -> None:
    """
    Plot portfolio value over Test + Validation as one continuous series.
    Validation starts from the last value of Test.
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
