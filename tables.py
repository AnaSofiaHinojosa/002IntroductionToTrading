import pandas as pd
import matplotlib.pyplot as plt


def returns_table(port_series: pd.Series) -> pd.DataFrame:
    """
    Compute monthly, quarterly, and annual return rates from a portfolio value series.

    Args:
        port_series (pd.Series): Time series of portfolio values.

    Returns:
        pd.DataFrame: DataFrame with columns for monthly, quarterly, and annual returns.
    """
    if not isinstance(port_series.index, pd.DatetimeIndex):
        port_series.index = pd.to_datetime(port_series.index)

    # Resample to get end-of-period values and compute returns

    monthly_returns = port_series.resample('ME').last().pct_change().round(4)
    quarterly_returns = port_series.resample('QE').last().pct_change().round(4)
    annual_returns = port_series.resample('YE').last().pct_change().round(4)

    full_index = monthly_returns.index
    quarterly_aligned = quarterly_returns.reindex(full_index, method='ffill')
    annual_aligned = annual_returns.reindex(full_index, method='ffill')

    # Combine into a single DataFrame

    returns_df = pd.DataFrame({
        "Monthly": monthly_returns,
        "Quarterly": quarterly_aligned,
        "Annual": annual_aligned
    })

    return returns_df


def show_table(df: pd.DataFrame, title: str = "Table"):
    """
    Display a DataFrame as a stylized table using matplotlib.

    If the DataFrame has more than 25 rows, it is split into two side-by-side tables.
    Cells are color-coded: green for positive values, red for negative values.

    Args:
        df (pd.DataFrame): DataFrame to display.
        title (str): Title to display above the table.
    """
    df_display = df.copy().round(4).astype(str).replace("nan", "—")
    n_rows = len(df_display)

    # Function to color cells based on value
    def color_cells(table, df_chunk):
        """
        Apply green/red background colors based on numeric sign,
        but skip headers and row labels.

        Args:
            table: Matplotlib table object.
            df_chunk (pd.DataFrame): Data chunk used to determine cell colors.
        """
        n_rows, n_cols = df_chunk.shape

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_text_props(weight='bold')
                continue
            if j == -1:
                continue
            try:
                val = float(df_chunk.iloc[i-1, j])
                if val > 0:
                    cell.set_facecolor("#dcefc5")
                elif val < 0:
                    cell.set_facecolor("#f7d6d6")
            except Exception:
                pass
    # Add condition for length            
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
            color_cells(table, chunk)
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
        color_cells(table, df_display)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df_display.columns))))

    plt.suptitle(title, fontsize=14, y=0.98)
    plt.subplots_adjust(top=0.9, wspace=0.3)
    plt.show()
