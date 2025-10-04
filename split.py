import pandas as pd


def data_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into training, and testing and validation sets (60% train, 20% test, 20% val).
    """
    n = len(data)
    train_end = int(0.60 * n)
    test_end = int(0.80 * n)

    train_data = data.iloc[:train_end, :]
    test_data = data.iloc[train_end:test_end, :]
    val_data = data.iloc[test_end:, :]
    return train_data, test_data, val_data
