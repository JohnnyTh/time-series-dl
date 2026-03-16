import logging

from time_series_dl.metrics.metrics import get_metrics
from time_series_dl.baselines.naive_models import build_baseline_models
from time_series_dl.evaluation.forecasting_runner import run_rolling_forecast
from time_series_dl.evaluation.statistics import (
    compute_metrics,
    compute_statistics,
    compute_horizon_metrics,
)
from time_series_dl.utils.io import save_json

logger = logging.getLogger(__name__)

def run_baseline_experiment(dataset, target_column, lag_time, lead_time, results_dir):
    metrics = get_metrics()
    models = build_baseline_models(lag_time)

    results = {}

    for model_name, model in models.items():
        logger.info("Running model: %s", model_name)

        forecasts = run_rolling_forecast(model, dataset, target_column, lead_time)

        metric_values = compute_metrics(forecasts, dataset, metrics, target_column)

        statistics = compute_statistics(metric_values)

        horizon_metrics = compute_horizon_metrics(forecasts, dataset, metrics, target_column)

        results[model_name] = {
            "statistics": statistics,
            "horizon_metrics": horizon_metrics,
        }

        save_json(
            results[model_name],
            f"{results_dir}/{model_name}_metrics.json",
        )

    save_json(results, f"{results_dir}/all_baseline_results.json")

    return results
