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


def backtest(data: pd.DataFrame, SL: float, TP: float, n_shares: float) -> tuple[list[float], float]:
    """
    Simulate a trading strategy over historical data.

    Args:
        data (pd.DataFrame): DataFrame containing market data and buy/sell signals.
                             Expected columns: 'Close', 'buy_signal', 'sell_signal'.
        SL (float): Stop-loss threshold as a percentage (e.g., 0.1 for 10%).
        TP (float): Take-profit threshold as a percentage (e.g., 0.1 for 10%).
        n_shares (int): Number of shares/contracts to trade per signal.

    Returns:
        tuple[list[float], float]: 
            - List of portfolio values over time.
            - Final cash balance after all trades.
    """
    data = data.copy()
    COM = 0.125 / 100  # Commission rate (0.125%)
    cash = 1_000_000   # Starting capital
    active_long = []   # List of open long positions
    active_short = []  # List of open short positions
    port_hist = []     # Portfolio value history

    for _, row in data.iterrows():
        # Evaluate and close long positions if SL or TP is triggered
        for pos in active_long.copy():
            if (pos.sl > row.Close) or (pos.tp < row.Close):
                # Sell at current price minus commission
                cash += pos.n_shares * row.Close * (1 - COM)
                active_long.remove(pos)

        # Evaluate and close short positions if SL or TP is triggered
        for pos in active_short.copy():
            if (pos.tp > row.Close) or (pos.sl < row.Close):
                # Buy back at current price, adjusting for commission
                cash += (pos.price * pos.n_shares) + \
                        (pos.price - row.Close) * pos.n_shares * (1 - COM)
                active_short.remove(pos)

        # Open long position if buy signal is present
        if row.buy_signal:
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_long.append(
                    Position(
                        price=row.Close,
                        sl=row.Close * (1 - SL),  # Set stop-loss below entry
                        tp=row.Close * (1 + TP),  # Set take-profit above entry
                        n_shares=n_shares
                    )
                )

        # Open short position if sell signal is present
        if row.sell_signal:
            cost = row.Close * n_shares * (1 + COM)
            if cash > cost:
                cash -= cost
                active_short.append(
                    Position(
                        price=row.Close,
                        sl=row.Close * (1 + SL),  # Set stop-loss above entry
                        tp=row.Close * (1 - TP),  # Set take-profit below entry
                        n_shares=n_shares
                    )
                )

        # Calculate current portfolio value including open positions
        port_value = cash
        for pos in active_long:
            port_value += pos.n_shares * row.Close  # Market value of long positions
        for pos in active_short:
            port_value += (pos.price * pos.n_shares) + \
                          (pos.price - row.Close) * pos.n_shares 

        port_hist.append(port_value)

    # Return portfolio history and final value
    return port_hist, port_hist[-1] if port_hist else cash
