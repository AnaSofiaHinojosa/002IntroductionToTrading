from data import load_data
from indicators import add_indicators, get_signals
from backtesting import backtest
from utils import plot_port_value

if __name__ == "__main__": 
    df = load_data("Binance_BTCUSDT_1h.csv")
    data = df.copy()
    data = add_indicators(data)
    data = get_signals(data)
    port_hist = backtest(data, 0.1, 0.1, 200)
    plot_port_value(port_hist, data.Date)