import ta
import pandas as pd

def add_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators and buy/sell signals to the DataFrame.
    """

    rsi_indicator = ta.momentum.RSIIndicator(close=data.Close, window=10)  
    sma_indicator = ta.trend.SMAIndicator(close=data.Close, window=5)      
    bb_indicator = ta.volatility.BollingerBands(
        close=data['Close'], window=15, window_dev=1.5)             

    data['RSI'] = rsi_indicator.rsi()
    data['SMA'] = sma_indicator.sma_indicator()
    data['BB_Upper'] = bb_indicator.bollinger_hband()
    data['BB_Lower'] = bb_indicator.bollinger_lband()


    data['buy_signal_rsi'] = data['RSI'] < 30      
    data['sell_signal_rsi'] = data['RSI'] > 70
    data['buy_signal_sma'] = data['Close'] > data['SMA']
    data['sell_signal_sma'] = data['Close'] < data['SMA']
    data['buy_signal_bb'] = data['Close'] < data['BB_Lower']
    data['sell_signal_bb'] = data['Close'] > data['BB_Upper']

    data = data.dropna()

    return data

def get_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate buy/sell signals based on technical indicators
    in the case that two or more conditions are met.
    """
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


