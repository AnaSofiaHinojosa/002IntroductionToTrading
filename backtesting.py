from dataclasses import dataclass


@dataclass
class Position:
    #time: str
    price: float
    sl: float
    tp: float
    n_shares: int
    #type: str

def backtest(data, SL, TP, n_shares) -> tuple[list[float], list[float]]:
    """
    Backtest a trading strategy.
    """

    data = data.copy()

    COM = 0.125 / 100
    cash = 1_000_000
    active_long = []
    active_short = []
    port_value = 0
    port_hist = []

    for i, row in data.iterrows():
        # close positions
        for pos in active_long.copy():
            if (pos.sl > row.Close) or (pos.tp < row.Close):
                cash += pos.n_shares * row.Close * (1 - COM)
                active_long.remove(pos)

        for pos in active_short.copy():
            if (pos.tp > row.Close) or (pos.sl < row.Close):
                cash += (pos.price * pos.n_shares) + (pos.price - row.Close) * pos.n_shares * (1 - COM)
                active_short.remove(pos)

        # open new positions
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

        # portfolio value
        port_value = cash
        for pos in active_long:
            port_value += pos.n_shares * row.Close
        for pos in active_short:
            port_value += (pos.price * pos.n_shares) + (pos.price - row.Close) * pos.n_shares

        port_hist.append(port_value)

    return port_hist
