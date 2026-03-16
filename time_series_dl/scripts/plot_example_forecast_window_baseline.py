import matplotlib.pyplot as plt
import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

from time_series_dl.data.dataset import load_exchange_dataset, split_dataset, build_forecasting_dataset


def plot_example_forecast_window(
    dataset,
    start_date: str,
    target_column: str,
    lag_time: int = 90,
    lead_time: int = 60,
    save_path: str = "example_forecast_window.png",
):
    """
    Generate an example forecast window figure using a persistence forecaster.

    Parameters
    ----------
    dataset : ForecastingDataset
        Dataset providing sliding windows.

    start_date : str
        Date corresponding to forecast origin (first prediction).

    target_column : str
        Name of target column.

    lag_time : int
        Input window length.

    lead_time : int
        Forecast horizon.

    save_path : str
        Output figure path.
    """

    start_date = pd.to_datetime(start_date)

    selected = None

    for i in range(len(dataset)):
        x, y, x_dates, y_dates = dataset[i]

        if y_dates.iloc[0] == start_date:
            selected = (x, y, x_dates, y_dates)
            break

    if selected is None:
        raise ValueError("start_date not found as forecast origin")

    x, y, x_dates, y_dates = selected

    # run persistence forecast
    model = NaiveForecaster(strategy="last")

    forecast = model.fit_predict(
        x[target_column],
        fh=list(range(1, lead_time + 1)),
    )

    forecast_series = pd.Series(forecast.values, index=y_dates)

    # plotting
    plt.figure(figsize=(12, 5))

    plt.plot(
        x_dates,
        x[target_column],
        label="Historical observations",
        linewidth=2,
        color="blue",
    )

    plt.plot(
        y_dates,
        y[target_column],
        label="True future values",
        linewidth=2,
        color="black",
    )

    plt.plot(
        forecast_series.index,
        forecast_series.values,
        label="Persistence forecast",
        linestyle="--",
        linewidth=2,
        color="red",
    )

    # forecast origin
    plt.axvline(
        x=x_dates.iloc[-1],
        linestyle="--",
        color="gray",
        alpha=0.8,
        label="Forecast origin",
    )

    plt.xlabel("Date")
    plt.ylabel(target_column)

    plt.title("Example Forecast Window (lag=90, horizon=60)")

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    df = load_exchange_dataset("boc_exchange/dataset.csv")

    train_df, val_df, test_df = split_dataset(df)

    dataset = build_forecasting_dataset(
        test_df,
        lag_time=90,
        lead_time=60,
        target_column="USD_CLOSE",
    )

    plot_example_forecast_window(
        dataset=dataset,
        start_date="2016-06-01",
        target_column="USD_CLOSE",
        save_path="figures/example_forecast_window.png",
    )


if __name__ == "__main__":
    main()
