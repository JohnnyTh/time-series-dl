import logging

from time_series_dl.data.dataset import (
    load_exchange_dataset,
    split_dataset,
    build_forecasting_dataset,
)
from time_series_dl.experiments.baseline_experiment import run_baseline_experiment
from time_series_dl.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

logger = logging.getLogger(__name__)


def main() -> None:
    df = load_exchange_dataset("boc_exchange/dataset.csv")

    train_df, val_df, test_df = split_dataset(df)

    test_dataset = build_forecasting_dataset(
        test_df,
        lag_time=90,
        lead_time=60,
        target_column="USD_CLOSE",
    )

    results = run_baseline_experiment(
        dataset=test_dataset,
        target_column="USD_CLOSE",
        lag_time=90,
        lead_time=60,
        results_dir="results/baselines",
    )
    save_json(results, "results_baseline.json")


if __name__ == "__main__":
    main()
