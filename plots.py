import matplotlib.pyplot as plt
import pandas as pd

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
    plt.grid(linestyle=':', alpha=0.5)
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
