import optuna
import pandas as pd
from pathlib import Path

from data import load_data
from split import data_split
from indicators import add_indicators, get_signals
from backtesting import backtest
from metrics import performance_summary
from optimize import optimize
from plots import plot_port_value_train, plot_port_value_test_val, plot_return_distribution, plot_returns_heatmap
from tables import returns_table, show_table

if __name__ == "__main__":
    # ============================
    # 1. Load and Split Data
    # ============================
    df = load_data("Binance_BTCUSDT_1h.csv")
    train_data, test_data, val_data = data_split(df)

    # ============================
    # 2. Hyperparameter Optimization (Train Set)
    # ============================
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: optimize(trial, train_data),
        n_trials=50,
        n_jobs=-1,
        show_progress_bar=True
    )

    print("\nBest parameters found:")
    params = study.best_params
    for k, v in params.items():
        print(f"  {k}: {v}")

    # ============================
    # 3. Backtest on Train Set
    # ============================
    print("\n--- Running on Train set ---")
    train_data_proc = add_indicators(
        train_data.copy(),
        rsi_window=params["rsi_window"],
        sma_window=params["sma_window"],
        bb_window=params["bb_window"],
        bb_dev=params["bb_dev"]
    )
    train_data_proc = get_signals(
        train_data_proc,
        rsi_buy=params["rsi_buy"],
        rsi_sell=params["rsi_sell"],
        sma_window=params["sma_window"],
        bb_window=params["bb_window"],
        bb_dev=params["bb_dev"]
    )
    port_hist_train, final_cash_train = backtest(
        train_data_proc,
        SL=params["SL"],
        TP=params["TP"],
        n_shares=params["n_shares"]
    )
    metrics_train = performance_summary(pd.Series(port_hist_train, index=train_data_proc.index), periods_per_year=8760)
    plot_port_value_train(port_hist_train, train_data_proc.Date)

    print("\nPerformance Summary (Train):")
    for key, value in metrics_train.items():
        print(f"{key}: {value:.4f}")
    print(f"Final Cash: {final_cash_train:.2f}")

    # ============================
    # 4. Backtest on Test Set
    # ============================
    print("\n--- Running on Test set ---")
    test_data_proc = add_indicators(
        test_data.copy(),
        rsi_window=params["rsi_window"],
        sma_window=params["sma_window"],
        bb_window=params["bb_window"],
        bb_dev=params["bb_dev"]
    )
    test_data_proc = get_signals(
        test_data_proc,
        rsi_buy=params["rsi_buy"],
        rsi_sell=params["rsi_sell"],
        sma_window=params["sma_window"],
        bb_window=params["bb_window"],
        bb_dev=params["bb_dev"]
    )
    port_hist_test, final_cash_test = backtest(
        test_data_proc,
        SL=params["SL"],
        TP=params["TP"],
        n_shares=params["n_shares"]
    )
    metrics_test = performance_summary(pd.Series(port_hist_test, index=test_data_proc.index), periods_per_year=8760)
    print("\nPerformance Summary (Test):")
    for key, value in metrics_test.items():
        print(f"{key}: {value:.4f}")
    print(f"Final Cash: {final_cash_test:.2f}")

    # ============================
    # 5. Backtest on Validation Set
    # ============================
    print("\n--- Running on Validation set ---")
    val_data_proc = add_indicators(
        val_data.copy(),
        rsi_window=params["rsi_window"],
        sma_window=params["sma_window"],
        bb_window=params["bb_window"],
        bb_dev=params["bb_dev"]
    )
    val_data_proc = get_signals(
        val_data_proc,
        rsi_buy=params["rsi_buy"],
        rsi_sell=params["rsi_sell"],
        sma_window=params["sma_window"],
        bb_window=params["bb_window"],
        bb_dev=params["bb_dev"]
    )
    port_hist_val, final_cash_val = backtest(
        val_data_proc,
        SL=params["SL"],
        TP=params["TP"],
        n_shares=params["n_shares"]
    )
    metrics_val = performance_summary(pd.Series(port_hist_val, index=val_data_proc.index), periods_per_year=8760)
    print("\nPerformance Summary (Validation):")
    for key, value in metrics_val.items():
        print(f"{key}: {value:.4f}")
    print(f"Final Cash: {final_cash_val:.2f}")

    # ============================
    # 6. Plot Combined Test + Validation
    # ============================
    plot_port_value_test_val(
        test_hist=port_hist_test,
        test_dates=test_data_proc.Date,
        val_hist=port_hist_val,
        val_dates=val_data_proc.Date
    )

    # ============================
    # 7. Generate Returns Tables
    # ============================
    port_series_train = pd.Series(
        port_hist_train, 
        index=pd.to_datetime(train_data_proc['Date'])
    )
    returns_table_train = returns_table(port_series_train)

    port_series_test = pd.Series(
        port_hist_test, 
        index=pd.to_datetime(test_data_proc['Date'])
    )
    returns_table_test = returns_table(port_series_test)

    port_series_val = pd.Series(
        port_hist_val, 
        index=pd.to_datetime(val_data_proc['Date'])
    )
    returns_table_val = returns_table(port_series_val)

    print(port_series_val.index.min(), port_series_val.index.max())

    # ============================
    # 8. Display Tables
    # ============================
    show_table(returns_table_train, "Train Set Returns Table")
    show_table(returns_table_test, "Test Set Returns Table")
    show_table(returns_table_val, "Validation Set Returns Table")

    # ============================
    # 9. Extra Charts / Visualizations
    # ============================

    # Histogram of returns (train)
    plot_return_distribution(port_series_train, title="Train Set Returns Distribution")

    # Monthly / Quarterly / Annual Returns Heatmaps
    plot_returns_heatmap(port_series_train, freq='M', title="Train Set Monthly Returns Heatmap")
    plot_returns_heatmap(port_series_train, freq='Q', title="Train Set Quarterly Returns Heatmap")

    plot_returns_heatmap(port_series_test, freq='M', title="Test Set Monthly Returns Heatmap")
    plot_returns_heatmap(port_series_test, freq='Q', title="Test Set Quarterly Returns Heatmap")

    plot_returns_heatmap(port_series_val, freq='M', title="Validation Set Monthly Returns Heatmap")
    plot_returns_heatmap(port_series_val, freq='Q', title="Validation Set Quarterly Returns Heatmap")

    
        