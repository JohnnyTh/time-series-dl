import json
import logging
import typing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sktime.forecasting.naive import NaiveForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

logger = logging.getLogger(__name__)


class ForecastingDataset:
    def __init__(
        self, dataset: pd.DataFrame, lag_time: int, lead_time: int, feature_columns: list[str]
    ) -> None:
        self.n_examples = len(dataset) - lag_time - lead_time + 1
        assert self.n_examples > 0, "Dataset must contain at least one example."
        assert (
            "date" in dataset.columns or "ds" in dataset.columns
        ), "Source DataFrame must contain a date/ds column."

        self.df = dataset[feature_columns]

        if "date" in dataset.columns:
            self.dates = dataset.date
        elif "ds" in dataset.columns:
            self.dates = dataset.ds

        self.lag_time = lag_time
        self.lead_time = lead_time

    def __len__(self) -> int:
        return self.n_examples

    def __getitem__(self, idx) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
        input_ = self.df.iloc[idx : idx + self.lag_time]
        output = self.df.iloc[idx + self.lag_time : idx + self.lag_time + self.lead_time]

        input_dates = self.dates[idx : idx + self.lag_time]
        output_dates = self.dates[idx + self.lag_time : idx + self.lag_time + self.lead_time]

        return input_, output, input_dates, output_dates


def load_dataset_splits(train_split_size: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset = pd.read_csv("boc_exchange/dataset.csv")
    dataset = dataset.rename(columns={dataset.columns[0]: "date"})
    dataset["date"] = pd.to_datetime(dataset["date"])

    train_df = dataset.iloc[: int(len(dataset) * train_split_size)]
    test_df = dataset.iloc[int(len(dataset) * train_split_size) :]

    return train_df, test_df


def plot_dataset(dataset: pd.DataFrame) -> None:
    ax = dataset.plot()
    plt.savefig("boc_exchange.png", dpi=150, bbox_inches="tight")
    plt.close()


def compute_error_statistics(error_metrics_dict: dict, exp_name: str) -> dict[str, pd.Series]:
    return {
        "mean": pd.DataFrame(error_metrics_dict).mean(axis=0).rename(f"{exp_name}_mean_metrics"),
        "std": pd.DataFrame(error_metrics_dict).std(axis=0).rename(f"{exp_name}_std_metrics"),
        "max": pd.DataFrame(error_metrics_dict).max(axis=0).rename(f"{exp_name}_max_metrics"),
    }


def compute_baseline_error_metrics(
    forecasts: list, test_dataset: ForecastingDataset, metrics: dict[str, typing.Callable]
) -> tuple[dict[str, list[float]], list]:
    errors = {metric_name: [] for metric_name in metrics.keys()}

    for i in range(len(forecasts)):

        fc = forecasts[i]
        x, y, x_d, y_d = test_dataset[i]

        for metric_name, metric_fn in metrics.items():
            errors[metric_name].append(metric_fn(y_true=y["USD_CLOSE"], y_pred=fc))

    return errors, forecasts


def log_metrics(metrics: dict[str, pd.Series]) -> None:
    """
    Log metrics stored as dict[str, pandas.Series].
    Each series contains metric_name -> value.
    """
    for stat_name, series in metrics.items():
        formatted = ", ".join(f"{metric}={value:.6f}" for metric, value in series.items())
        logger.info("%s: %s", stat_name, formatted)


def plot_forecasts(
    forecasts: list, test_df: pd.DataFrame, lead_time: int, name: str, target_column: str
):
    max_fcs = [{"date": fc.index[-1:][0], "yhat": fc[-1:][0]} for fc in forecasts]
    max_fcs = pd.DataFrame(max_fcs)

    plt.figure(figsize=(16, 4))
    plt.plot(test_df.date, test_df[target_column], color="blue", label="ground truth")
    plt.plot(max_fcs.date, max_fcs.yhat, color="red", label="forecast")
    plt.title(f"Forecasts at max lead time ({lead_time} samples) - {name}")
    plt.legend(loc="upper right")

    # Plot ground truth
    # plt.figure(figsize=(12, 3))
    # ground_truth = test_df[["date", target_column]]
    # plt.plot(ground_truth.date, ground_truth[target_column], label="ground truth")

    # Plot example single forecast
    # plt.plot(forecasts[-1], label="forecast")
    # plt.legend()

    ax = plt.gca()
    locator = mdates.AutoDateLocator(minticks=3, maxticks=10)
    formatter = mdates.ConciseDateFormatter(locator)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"forecasts_{name}.png", dpi=200)
    plt.close()


def main() -> None:
    lag_time = 90
    lead_time = 60

    target_column = "USD_CLOSE"

    train_df, test_df = load_dataset_splits(train_split_size=0.8)

    feature_columns = [col for col in test_df if col.endswith("_CLOSE")]
    test_dataset = ForecastingDataset(test_df, lag_time, lead_time, feature_columns)

    metrics = {
        "mse": mean_squared_error,
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
    }

    # =========== train naive forecaster
    baseline_model_persistence = NaiveForecaster(strategy="last")
    baseline_model_mean = NaiveForecaster(strategy="mean", window_length=lag_time)

    forecasts_persistence = []
    forecasts_mean = []
    for i in range(len(test_dataset)):
        x, y, x_dates, y_dates = test_dataset[i]

        persistence_fc = baseline_model_persistence.fit_predict(
            x[target_column], fh=list(range(1, lead_time + 1))
        )
        persistence_fc = pd.Series(persistence_fc.values, index=y_dates)
        forecasts_persistence.append(persistence_fc)

        mean_fc = baseline_model_mean.fit_predict(
            x[target_column], fh=list(range(1, lead_time+1))
        )
        mean_fc = pd.Series(mean_fc.values, index=y_dates)
        forecasts_mean.append(mean_fc)

        if i % 25 == 0:
            logger.info(f"Predict {i}/{len(test_dataset)}")

    persistence_errors, _ = compute_baseline_error_metrics(
        forecasts_persistence, test_dataset, metrics
    )
    mean_errors, _ = compute_baseline_error_metrics(forecasts_mean, test_dataset, metrics)

    persistence_stats = compute_error_statistics(persistence_errors, "persistence")
    mean_window_stats = compute_error_statistics(mean_errors, "mean_window")

    logger.info(f"NaiveForecaster with strategy=last metrics:")
    log_metrics(persistence_stats)
    plot_forecasts(
        forecasts_persistence, test_df, lead_time, "Persistence", target_column=target_column
    )

    logger.info(f"NaiveForecaster with strategy=mean metrics")
    log_metrics(mean_window_stats)
    plot_forecasts(forecasts_mean, test_df, lead_time, "Mean Window", target_column=target_column)


if __name__ == "__main__":
    main()
