import ta
import pandas as pd

def add_indicators(
    data: pd.DataFrame,
    rsi_window: int = 14,
    sma_window: int = 20,
    bb_window: int = 20,
    bb_dev: float = 2.0
) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame with tunable parameters.
    """

    rsi_indicator = ta.momentum.RSIIndicator(close=data.Close, window=rsi_window)
    sma_indicator = ta.trend.SMAIndicator(close=data.Close, window=sma_window)
    bb_indicator = ta.volatility.BollingerBands(
        close=data['Close'], window=bb_window, window_dev=bb_dev
    )

    data['RSI'] = rsi_indicator.rsi()
    data['SMA'] = sma_indicator.sma_indicator()
    data['BB_Upper'] = bb_indicator.bollinger_hband()
    data['BB_Lower'] = bb_indicator.bollinger_lband()

    data = data.dropna()
    return data


def get_signals(
    data: pd.DataFrame,
    rsi_buy: int = 30,
    rsi_sell: int = 70,
    sma_window: int = 20,
    bb_window: int = 20,
    bb_dev: float = 2.0
) -> pd.DataFrame:
    """
    Generate buy/sell signals using RSI, SMA, and Bollinger Band thresholds.

    Args:
        data (pd.DataFrame): Market data with 'Close' and 'RSI' columns.
        rsi_buy (int): RSI threshold below which to trigger a buy signal.
        rsi_sell (int): RSI threshold above which to trigger a sell signal.
        sma_window (int): Window size for Simple Moving Average.
        bb_window (int): Window size for Bollinger Bands.
        bb_dev (float): Number of standard deviations for Bollinger Bands.

    Returns:
        pd.DataFrame: DataFrame with buy/sell signals added.
    """
    data = data.copy()
    
    # RSI signals
    data['buy_signal_rsi'] = data['RSI'] < rsi_buy
    data['sell_signal_rsi'] = data['RSI'] > rsi_sell

    # SMA signals
    data['SMA'] = data['Close'].rolling(window=sma_window).mean()
    data['buy_signal_sma'] = data['Close'] > data['SMA']
    data['sell_signal_sma'] = data['Close'] < data['SMA']

    # Bollinger Bands signals
    bb_ma = data['Close'].rolling(window=bb_window).mean()
    bb_std = data['Close'].rolling(window=bb_window).std()
    data['BB_Upper'] = bb_ma + bb_dev * bb_std
    data['BB_Lower'] = bb_ma - bb_dev * bb_std
    data['buy_signal_bb'] = data['Close'] < data['BB_Lower']
    data['sell_signal_bb'] = data['Close'] > data['BB_Upper']

    # Final signals: require at least 2 of 3 indicators to agree
    data['buy_signal'] = (
        data['buy_signal_rsi'].astype(int) +
        data['buy_signal_sma'].astype(int) +
        data['buy_signal_bb'].astype(int)
    ) >= 2

    data['sell_signal'] = (
        data['sell_signal_rsi'].astype(int) +
        data['sell_signal_sma'].astype(int) +
        data['sell_signal_bb'].astype(int)
    ) >= 2

    return data
