import matplotlib.pyplot as plt
import pandas as pd
from pmdarima import auto_arima
from prophet import Prophet

from time_series_dl.data.dataset import (
    load_exchange_dataset,
    split_dataset,
    build_forecasting_dataset,
)


def run_arima_forecast(train_series: pd.Series, lead_time: int) -> pd.Series:
    model = auto_arima(
        train_series,
        seasonal=False,
        stepwise=True,
        suppress_warnings=True,
        d=1,
        max_p=3,
        max_q=2,
    )

    forecast = model.predict(n_periods=lead_time)
    return pd.Series(forecast)


def run_prophet_forecast(train_df: pd.DataFrame, lead_time: int) -> pd.Series:
    prophet_df = train_df.rename(columns={"date": "ds", "USD_CLOSE": "y"})[
        ["ds", "y"]
    ]

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(prophet_df)

    future = model.make_future_dataframe(periods=lead_time, freq="D")
    forecast = model.predict(future)

    yhat = forecast["yhat"].iloc[-lead_time:].reset_index(drop=True)
    return pd.Series(yhat)


def plot_example_forecast_window(
    df: pd.DataFrame,
    dataset,
    start_date: str,
    target_column: str,
    model_type: str = "arima",
    lead_time: int = 60,
    save_path: str = "example_forecast_window.png",
):
    start_date = pd.to_datetime(start_date)

    # === find matching dataset window (only for y_dates + ground truth) ===
    selected = None
    for i in range(len(dataset)):
        x, y, x_dates, y_dates = dataset[i]
        if y_dates.iloc[0] == start_date:
            selected = (x, y, x_dates, y_dates)
            break

    if selected is None:
        raise ValueError("start_date not found")

    x, y, x_dates, y_dates = selected

    # === proper training set: ALL data before forecast origin ===
    train_df = df[df["date"] < start_date]

    if len(train_df) == 0:
        raise ValueError("No training data before selected date")

    # === run model ===
    if model_type == "arima":
        forecast_values = run_arima_forecast(
            train_df[target_column],
            lead_time,
        )

    elif model_type == "prophet":
        forecast_values = run_prophet_forecast(
            train_df[["date", target_column]],
            lead_time,
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    forecast_series = pd.Series(forecast_values.values, index=y_dates)

    # === plotting ===
    plt.figure(figsize=(12, 5))

    plt.plot(x_dates, x[target_column], label="Recent history", linewidth=2)
    plt.plot(y_dates, y[target_column], label="True future", linewidth=2)
    plt.plot(
        forecast_series.index,
        forecast_series.values,
        linestyle="--",
        linewidth=2,
        label=f"{model_type.upper()} forecast",
    )

    plt.axvline(
        x=x_dates.iloc[-1],
        linestyle="--",
        color="gray",
        alpha=0.8,
        label="Forecast origin",
    )

    plt.xlabel("Date")
    plt.ylabel(target_column)
    plt.title(f"Forecast Example ({model_type.upper()}, horizon={lead_time})")

    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    df = load_exchange_dataset("boc_exchange/dataset.csv")

    # _, _, test_df = split_dataset(df)

    dataset = build_forecasting_dataset(
        df,
        lag_time=90,
        lead_time=180,
        target_column="USD_CLOSE",
    )
    model = "arima"
    plot_example_forecast_window(
        df=df,  # <-- full dataset now used
        dataset=dataset,
        lead_time=180,
        start_date="2016-06-01",
        target_column="USD_CLOSE",
        model_type=model,  # or "prophet"
        save_path=f"figures/example_forecast_window_{model}.png",
    )


if __name__ == "__main__":
    main()