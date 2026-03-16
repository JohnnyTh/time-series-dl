# === File: main_non_dl_models.py ===
import logging
from pathlib import Path

import pandas as pd
from prophet import Prophet
from pmdarima import auto_arima

from time_series_dl.data.dataset import (
    load_exchange_dataset,
    split_dataset,
    build_forecasting_dataset,
    ForecastingDataset,
)
from time_series_dl.evaluation.statistics import (
    compute_metrics,
    compute_statistics,
    compute_horizon_metrics,
)
from time_series_dl.metrics.metrics import get_metrics
from time_series_dl.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def prepare_prophet_dataframe(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Convert to Prophet-compatible dataframe.
    """
    prophet_df = df[["date", target_column]].rename(columns={"date": "ds", target_column: "y"})
    return prophet_df


def run_prophet_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, lead_time: int):
    """
    Train Prophet on training set and forecast test set.
    Returns pd.Series aligned with test_df dates.
    """
    prophet_train = prepare_prophet_dataframe(train_df, "USD_CLOSE")

    model = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(prophet_train)

    future = pd.DataFrame({"ds": test_df["date"].iloc[:lead_time]})
    forecast = model.predict(future)

    yhat = pd.Series(forecast["yhat"].values, index=future["ds"])
    return yhat


def run_arima_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame, lead_time: int):
    """
    Train auto_arima and forecast test set.
    """
    train_series = train_df["USD_CLOSE"]

    model = auto_arima(
        train_series,
        seasonal=False,  # Bank of Canada exchange rate might be treated as non-seasonal
        stepwise=True,
        suppress_warnings=True,
        d=1,
        p=3,
        q=2,
    )
    # logger.info(f"ARIMA model summary:\n{model.summary()}")

    forecast = model.predict(n_periods=lead_time)
    yhat = pd.Series(forecast.values, index=test_df["date"].iloc[:lead_time])
    return yhat


def collect_forecasts(
    dataset: ForecastingDataset, train_df: pd.DataFrame, test_df: pd.DataFrame, lead_time: int
):
    """
    Run Prophet and ARIMA forecasts using a rolling/expanding window.
    Each forecast starts from the expanding window of train + previous test points.
    Forecasts are aligned with dataset indices for proper evaluation.
    """
    forecasts = {"prophet": [], "arima": []}
    n_windows = len(dataset)

    step_size = 10  # or 10
    for i in range(0, len(dataset), step_size):
        x, y, x_dates, y_dates = dataset[i]

        # Build expanding training window: train_df + all previous test examples
        train_expanded = pd.concat([train_df, test_df.iloc[:i]])

        # Forecast next 'lead_time' points
        prophet_fc = run_prophet_forecast(
            train_expanded, test_df.iloc[i : i + lead_time], lead_time
        )
        arima_fc = run_arima_forecast(train_expanded, test_df.iloc[i : i + lead_time], lead_time)

        # Align forecasts with the current window dates
        forecasts["prophet"].append(pd.Series(prophet_fc.values, index=y_dates))
        forecasts["arima"].append(pd.Series(arima_fc.values, index=y_dates))

        logger.info(f"Processed {i}/{n_windows} rolling forecast windows")

    return forecasts


def main():
    lag_time = 90
    lead_time = 60
    target_column = "USD_CLOSE"

    # Load and split dataset
    df = load_exchange_dataset("boc_exchange/dataset.csv")
    train_df, val_df, test_df = split_dataset(df)
    train_df = pd.concat([train_df, val_df])

    feature_columns = [col for col in df.columns if col != "date" and col != target_column] + [
        target_column
    ]
    test_dataset = build_forecasting_dataset(test_df, lag_time, lead_time, target_column)

    # Metrics
    metrics = get_metrics()
    results_dir = Path("results/non_dl_models")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Collect forecasts
    forecasts = collect_forecasts(test_dataset, train_df, test_df, lead_time)

    results = {}

    for model_name, model_forecasts in forecasts.items():
        logger.info("Processing model: %s", model_name)

        metric_values = compute_metrics(model_forecasts, test_dataset, metrics, target_column)
        statistics = compute_statistics(metric_values)
        horizon_metrics = compute_horizon_metrics(
            model_forecasts, test_dataset, metrics, target_column
        )

        results[model_name] = {
            "statistics": statistics,
            "horizon_metrics": horizon_metrics,
        }

        save_json(results[model_name], results_dir / f"{model_name}_metrics.json")

    save_json(results, results_dir / "all_non_dl_results.json")
    logger.info("All non-DL model results saved.")


if __name__ == "__main__":
    main()
