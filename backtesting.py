from dataclasses import dataclass
import pandas as pd

@dataclass
class Position:
    """
    Represents a trading position.

    Attributes:
        price (float): Entry price of the position.
        sl (float): Stop-loss price.
        tp (float): Take-profit price.
        n_shares (int): Number of shares/contracts held.
    """
    price: float
    sl: float
    tp: float
    n_shares: int


def backtest(data: pd.DataFrame, SL: float, TP: float, n_shares: int) -> tuple[list[float], float]:
    """
    Simulate a trading strategy over historical data.

    Args:
        data (pd.DataFrame): DataFrame containing market data and buy/sell signals.
        SL (float): Stop-loss threshold as a percentage (e.g., 0.1 for 10%).
        TP (float): Take-profit threshold as a percentage (e.g., 0.1 for 10%).
        n_shares (int): Number of shares/contracts to trade per signal.

    Returns:
        tuple[list[float], float]: 
            - List of portfolio values over time.
            - Final cash balance after all trades.
    """
    data = data.copy()
    COM = 0.125 / 100  # Commission rate
    cash = 1_000_000   # Starting capital
    active_long = []   # List of open long positions
    active_short = []  # List of open short positions
    port_hist = []     # Portfolio value history

    for _, row in data.iterrows():
        # Close long positions if stop-loss or take-profit is hit
        for pos in active_long.copy():
            if (pos.sl > row.Close) or (pos.tp < row.Close):
                cash += pos.n_shares * row.Close * (1 - COM)
                active_long.remove(pos)

        # Close short positions if stop-loss or take-profit is hit
        for pos in active_short.copy():
            if (pos.tp > row.Close) or (pos.sl < row.Close):
                cash += (pos.price * pos.n_shares) + \
                        (pos.price - row.Close) * pos.n_shares * (1 - COM)
                active_short.remove(pos)

        # Open new long position if buy signal is triggered
        if row.buy_signal:
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long.append(
                    Position(
                        price=row.Close,
                        sl=row.Close * (1 - SL),
                        tp=row.Close * (1 + TP),
                        n_shares=n_shares
                    )
                )

        # Open new short position if sell signal is triggered
        if row.sell_signal:
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_short.append(
                    Position(
                        price=row.Close,
                        sl=row.Close * (1 + SL),
                        tp=row.Close * (1 - TP),
                        n_shares=n_shares
                    )
                )

        # Calculate current portfolio value
        port_value = cash
        for pos in active_long:
            port_value += pos.n_shares * row.Close
        for pos in active_short:
            port_value += (pos.price * pos.n_shares) + \
                          (pos.price - row.Close) * pos.n_shares

        port_hist.append(port_value)

    return port_hist, cash
