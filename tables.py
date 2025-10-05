import pandas as pd
import matplotlib.pyplot as plt


def returns_table(port_series: pd.Series) -> pd.DataFrame:
    if not isinstance(port_series.index, pd.DatetimeIndex):
        port_series.index = pd.to_datetime(port_series.index)

    monthly_returns = port_series.resample('ME').last().pct_change().round(4)
    quarterly_returns = port_series.resample('QE').last().pct_change().round(4)
    annual_returns = port_series.resample('YE').last().pct_change().round(4)

    full_index = monthly_returns.index
    quarterly_aligned = quarterly_returns.reindex(full_index, method='ffill')
    annual_aligned = annual_returns.reindex(full_index, method='ffill')

    returns_df = pd.DataFrame({
        "Monthly": monthly_returns,
        "Quarterly": quarterly_aligned,
        "Annual": annual_aligned
    })

    return returns_df


def show_table(df: pd.DataFrame, title: str = "Table"):
    """
    Display a DataFrame as a table using matplotlib.
    If more than 15 rows, split into two side-by-side tables.
    """
    df_display = df.copy().round(4).astype(str).replace("nan", "—")
    n_rows = len(df_display)

    if n_rows > 25:

        mid = (n_rows + 1) // 2
        df1, df2 = df_display.iloc[:mid], df_display.iloc[mid:]

        fig, axes = plt.subplots(1, 2, figsize=(18, max(6, n_rows * 0.15)))
        for ax, chunk in zip(axes, [df1, df2]):
            ax.axis("tight")
            ax.axis("off")
            table = ax.table(
                cellText=chunk.values,
                colLabels=chunk.columns,
                rowLabels=chunk.index.strftime('%Y-%m-%d'),
                cellLoc="center",
                rowLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.auto_set_column_width(col=list(range(len(chunk.columns))))
    else:
        fig, ax = plt.subplots(figsize=(14, max(6, n_rows * 0.25)))
        ax.axis("tight")
        ax.axis("off")
        table = ax.table(
            cellText=df_display.values,
            colLabels=df_display.columns,
            rowLabels=df_display.index.strftime('%Y-%m-%d'),
            cellLoc="center",
            rowLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df_display.columns))))

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.subplots_adjust(top=0.9, wspace=0.3)
    plt.show()
