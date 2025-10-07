import pandas as pd


def data_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into training, testing, and validation sets.

    The split is done sequentially to preserve temporal order:
        - 60% for training
        - 20% for testing
        - 20% for validation

    Args:
        data (pd.DataFrame): Time-indexed DataFrame to be split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: 
            - Training set
            - Testing set
            - Validation set
    """
    n = len(data)
    train_end = int(0.60 * n)
    test_end = int(0.80 * n)

    train_data = data.iloc[:train_end, :]
    test_data = data.iloc[train_end:test_end, :]
    val_data = data.iloc[test_end:, :]
    return train_data, test_data, val_data
