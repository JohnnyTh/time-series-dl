import logging
from pathlib import Path

import pandas as pd

from time_series_dl.data.dataset import (
    load_exchange_dataset,
    split_dataset,
    build_forecasting_dataset,
)
from time_series_dl.evaluation.statistics import (
    compute_metrics,
    compute_statistics,
    compute_horizon_metrics,
)
from time_series_dl.metrics.metrics import get_metrics

import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    NBeats,
    TemporalFusionTransformer,
    DeepAR,
)
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE, RMSE, MAPE
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.encoders import GroupNormalizer
from pytorch_forecasting import Baseline
from torch.utils.data import DataLoader

from time_series_dl.utils.io import save_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


def main():
    # --- 1. Load and preprocess data ---
    df = load_exchange_dataset("boc_exchange/dataset.csv")
    train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.1)

    # Add time features for TFT / DeepAR
    for dataset in [train_df, val_df, test_df]:
        dataset["month"] = dataset["date"].dt.month.astype("int")
        dataset["day_of_week"] = dataset["date"].dt.dayofweek.astype("int")
        dataset["time_idx"] = (dataset["date"] - dataset["date"].min()).dt.days

    TARGET = "USD_CLOSE"
    GROUP_IDS = ["dummy"]  # Single time series in this case

    # --- 2. Build pytorch-forecasting dataset ---
    max_encoder_length = 90
    max_prediction_length = 60

    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target=TARGET,
        group_ids=GROUP_IDS,
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[TARGET],
        time_varying_known_reals=["time_idx", "month", "day_of_week"],
        add_relative_time_idx=True,
        add_target_scales=True,
    )

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, val_df, predict=True, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, test_df, predict=True, stop_randomization=True
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # --- 3. Select model ---
    MODEL_NAME = "TFT"  # choose from: "NBEATS", "TFT", "DeepAR"

    if MODEL_NAME == "NBEATS":
        model_class = NBeats
    elif MODEL_NAME == "TFT":
        model_class = TemporalFusionTransformer
    elif MODEL_NAME == "DeepAR":
        model_class = DeepAR
    else:
        raise ValueError(f"Unknown model {MODEL_NAME}")

    model = model_class.from_dataset(
        train_dataset,
        learning_rate=1e-3,
        log_interval=10,
        log_val_interval=1,
        hidden_size=32,
        attention_head_size=4 if MODEL_NAME == "TFT" else None,
        dropout=0.1,
        loss=MAE(),
        optimizer="adam",
    )

    # --- 4. Train model ---
    trainer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logger.info(f"Training {MODEL_NAME} model on {len(train_dataset)} samples")

    trainer = Trainer(
        max_epochs=20,
        gpus=1 if torch.cuda.is_available() else 0,
        gradient_clip_val=0.1,
        limit_train_batches=1.0,  # fraction or integer
    )

    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

    # --- 5. Inference ---
    raw_predictions, x = model.predict(test_loader, mode="raw", return_x=True)
    predictions = model.predict(test_loader)

    # Convert predictions to series aligned with dates
    pred_series = pd.Series(predictions.numpy(), index=test_df["date"].iloc[: len(predictions)])

    # --- 6. Evaluation ---
    from time_series_dl.evaluation.statistics import (
        compute_metrics,
        compute_statistics,
        compute_horizon_metrics,
    )
    from time_series_dl.metrics.metrics import get_metrics

    metrics = get_metrics()
    forecasts = [pred_series]  # wrap as list to reuse your functions

    metric_values = compute_metrics(forecasts, test_dataset, metrics, TARGET)
    statistics = compute_statistics(metric_values)
    horizon_metrics = compute_horizon_metrics(forecasts, test_dataset, metrics, TARGET)

    results_dir = Path("results/dl_models")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "statistics": statistics,
        "horizon_metrics": horizon_metrics,
    }

    save_json(results, results_dir / f"{MODEL_NAME}_results.json")
    logger.info(f"Training and evaluation of {MODEL_NAME} completed. Results saved.")


if __name__ == "__main__":
    main()
