import matplotlib.pyplot as plt
import pandas as pd

def plot_port_value(port_hist: list[float], dates: pd.Series) -> None:
    """
    Plot the portfolio value over time.
    """

    plt.figure(figsize=(10, 5))
    plt.plot(dates, port_hist, color='palevioletred')
    plt.title("Portfolio Value Over Time")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(linestyle=':', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()