import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess historical price data from a CSV file.
    """
    df = pd.read_csv(file_path, skiprows=1)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed')
    df.set_index('Date', inplace=True)
    df = df.iloc[::-1].reset_index(drop=False)
    return df

