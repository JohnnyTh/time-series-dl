import numpy as np
import pandas as pd


def compute_metrics(forecasts: list[pd.Series], dataset: pd.DataFrame, metrics, target_column):
    results = {k: [] for k in metrics}

    for i in range(len(forecasts)):

        fc = forecasts[i]
        _, y, _, _ = dataset[i]

        y_true = y[target_column]

        for name, fn in metrics.items():
            results[name].append(fn(y_true, fc))

    return results


def compute_statistics(metrics_dict):

    df = pd.DataFrame(metrics_dict)

    return {
        "mean": df.mean().to_dict(),
        "std": df.std().to_dict(),
        "median": df.median().to_dict(),
        "max": df.max().to_dict(),
        "min": df.min().to_dict(),
    }


def compute_horizon_metrics(forecasts, dataset, metrics, target_column):
    horizon_results = {}

    lead_time = len(forecasts[0])

    for h in range(lead_time):

        horizon_results[h + 1] = {}

        for name in metrics:
            horizon_results[h + 1][name] = []

        for i in range(len(forecasts)):

            fc = forecasts[i]
            _, y, _, _ = dataset[i]

            y_true = y[target_column].iloc[h]
            y_pred = fc.iloc[h]

            for name, fn in metrics.items():
                horizon_results[h + 1][name].append(fn([y_true], [y_pred]))

        for name in metrics:
            horizon_results[h + 1][name] = float(np.mean(horizon_results[h + 1][name]))

    return horizon_results
