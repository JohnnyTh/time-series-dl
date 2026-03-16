import json
import pathlib

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def bucket_horizon_metrics(
    data: Dict,
    buckets: List[Tuple[int, int]] = [(1, 5), (6, 10), (11, 20), (21, 40), (41, 60)],
) -> Dict:
    """
    Convert horizon metrics into aggregated buckets.

    Parameters
    ----------
    data : dict
        Input json structure containing model -> horizon_metrics.

    buckets : list[(start, end)]
        Horizon bucket ranges.

    Returns
    -------
    dict
        Aggregated horizon bucket metrics.
    """

    result = {}

    for model_name, model_data in data.items():

        horizon_metrics = model_data["horizon_metrics"]
        bucket_results = {}

        for start, end in buckets:

            bucket_name = f"{start}-{end}"

            bucket_values = {
                "mse": [],
                "rmse": [],
                "mae": [],
                "mape": [],
            }

            for h in range(start, end + 1):

                h_str = str(h)
                if h_str not in horizon_metrics:
                    continue

                metrics = horizon_metrics[h_str]

                for metric in bucket_values:
                    bucket_values[metric].append(metrics[metric])

            bucket_avg = {
                metric: round(sum(values) / len(values), 4)
                for metric, values in bucket_values.items()
                if values
            }

            bucket_results[bucket_name] = bucket_avg

        result[model_name] = {"horizon_buckets": bucket_results}

    return result


def plot_forecast_error_vs_horizon(
    json_path: str,
    metric: str = "mae",
    save_path: str = "forecast_error_vs_horizon.png",
):
    """
    Plot forecast error vs horizon for baseline models.

    Parameters
    ----------
    json_path : str
        Path to results json file.

    metric : str
        Error metric to plot ("mae", "rmse", "mse", "mape").

    save_path : str
        Path to output figure.
    """

    with open(json_path, "r") as f:
        data = json.load(f)

    models = {
        "Persistence": data["persistence"]["horizon_metrics"],
        "Mean Window": data["mean_window"]["horizon_metrics"],
        "Drift": data["drift"]["horizon_metrics"],
    }

    plt.figure(figsize=(12, 5))

    for model_name, horizon_metrics in models.items():
        horizons = sorted(int(h) for h in horizon_metrics.keys())

        errors = [horizon_metrics[str(h)][metric] for h in horizons]

        plt.plot(
            horizons,
            errors,
            linewidth=2.5,
            label=model_name,
        )

    plt.xlabel("Forecast Horizon (days ahead)", fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)

    plt.title("Forecast Error vs Prediction Horizon", fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.6)

    plt.legend(frameon=True)

    plt.xlim(1, max(horizons))

    plt.tight_layout()

    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    json_path = "results/baselines/all_baseline_results.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    bucketed = bucket_horizon_metrics(data)
    print(bucketed)

    plot_forecast_error_vs_horizon(json_path)


if __name__ == "__main__":
    main()
