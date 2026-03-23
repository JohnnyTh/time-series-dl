import json
import logging
import typing
from pathlib import Path

import numpy as np
import pandas as pd

from time_series_dl.data.dataset import (
    load_exchange_dataset,
    split_dataset,
    build_forecasting_dataset,
)

import pytorch_lightning as pl
from lightning.pytorch import Trainer

import torch
from pytorch_forecasting import (
    TimeSeriesDataSet,
    NBeats,
    TemporalFusionTransformer,
    DeepAR,
)
from pytorch_forecasting.metrics import MAE

from time_series_dl.utils.io import save_json


from time_series_dl.evaluation.statistics import compute_statistics
from time_series_dl.metrics.metrics import get_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")
logger = logging.getLogger(__name__)


class LossHistoryCallback(pl.Callback):
    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.history = {
            "train_loss": [],
            "val_loss": [],
        }

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        train_loss = metrics.get("train_loss")
        if train_loss is not None:
            self.history["train_loss"].append(float(train_loss.cpu()))

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        val_loss = metrics.get("val_loss")
        if val_loss is not None:
            self.history["val_loss"].append(float(val_loss.cpu()))

    def on_fit_end(self, trainer, pl_module):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.save_path, "w") as f:
            json.dump(self.history, f, indent=4)


def convert_predictions_to_forecasts(
    predictions: typing.Dict[str, torch.Tensor], x: typing.Dict[str, torch.Tensor], df: pd.DataFrame
) -> list[pd.Series]:
    forecasts = []

    pred_values = predictions["prediction"].detach().cpu().numpy()
    time_idx = x["decoder_time_idx"].detach().cpu().numpy()
    date_lookup = df.set_index("time_idx")["date"]

    for i in range(pred_values.shape[0]):

        horizon = pred_values[i].squeeze(-1)
        idx = time_idx[i]

        dates = date_lookup.reindex(idx)

        forecasts.append(pd.Series(horizon, index=dates.values))

    return forecasts


def build_dataset(
    df: pd.DataFrame,
    model_name: str,
    lag_time: int,
    lead_time: int,
    target: str,
    group_ids: list[str],
) -> TimeSeriesDataSet:
    if model_name == "NBEATS":
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=target,
            group_ids=group_ids,
            max_encoder_length=lag_time,
            max_prediction_length=lead_time,
            time_varying_unknown_reals=[target],
            add_relative_time_idx=False,
            add_target_scales=False,
        )

    elif model_name in ["TFT", "DEEPAR"]:
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target=target,
            group_ids=group_ids,
            max_encoder_length=lag_time,
            max_prediction_length=lead_time,
            time_varying_unknown_reals=[target],
            time_varying_known_reals=["time_idx", "month", "day_of_week"],
            add_relative_time_idx=True,
            add_target_scales=True,
        )

    else:
        raise ValueError(model_name)


def convert_predictions_to_forecasts_and_truth(
    prediction,
    df: pd.DataFrame,
):
    forecasts = []
    targets = []

    pred_values = prediction.output["prediction"].detach().cpu().numpy()
    decoder_target = prediction.x["decoder_target"].detach().cpu().numpy()
    time_idx = prediction.x["decoder_time_idx"].detach().cpu().numpy()

    date_lookup = df.set_index("time_idx")["date"]

    for i in range(pred_values.shape[0]):

        y_pred = pred_values[i].squeeze(-1)
        y_true = decoder_target[i]
        idx = time_idx[i]

        dates = date_lookup.reindex(idx)

        forecasts.append(pd.Series(y_pred, index=dates.values))
        targets.append(pd.Series(y_true, index=dates.values))

    return forecasts, targets


def compute_metrics_direct(forecasts, targets, metrics):
    results = {k: [] for k in metrics}

    for y_pred, y_true in zip(forecasts, targets):
        for name, fn in metrics.items():
            results[name].append(fn(y_true, y_pred))

    return results


def compute_horizon_metrics_direct(forecasts, targets, metrics):
    lead_time = len(forecasts[0])
    horizon_results = {}

    for h in range(lead_time):
        horizon_results[h + 1] = {}

        for name in metrics:
            values = []

            for y_pred, y_true in zip(forecasts, targets):
                values.append(fn := metrics[name]([y_true.iloc[h]], [y_pred.iloc[h]]))

            horizon_results[h + 1][name] = float(np.mean(values))

    return horizon_results


def main() -> None:
    pl.seed_everything(42)

    model_name = "NBEATS"

    group_ids = ["series"]
    target = "USD_CLOSE"
    lag_time = 90
    lead_time = 60

    # --- 1. Load and preprocess data ---
    df = load_exchange_dataset("boc_exchange/dataset.csv")
    df["series"] = 0
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days

    train_df, val_df, test_df = split_dataset(df, train_ratio=0.7, val_ratio=0.1)

    # Add time features for TFT / DeepAR
    for dataset in [train_df, val_df, test_df]:
        dataset["month"] = dataset["date"].dt.month.astype("int")
        dataset["day_of_week"] = dataset["date"].dt.dayofweek.astype("int")

    # --- 2. Build pytorch-forecasting dataset ---
    train_dataset = build_dataset(train_df, model_name, lag_time, lead_time, target, group_ids)

    val_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, val_df, predict=False, stop_randomization=True
    )
    test_dataset = TimeSeriesDataSet.from_dataset(
        train_dataset, test_df, predict=False, stop_randomization=True
    )

    train_loader = train_dataset.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)
    test_loader = test_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    # --- 3. Select model ---
    if model_name == "NBEATS":
        model_class = NBeats
        extra_args = {}
    elif model_name == "TFT":
        model_class = TemporalFusionTransformer
        extra_args = {"attention_head_size": 4}
    elif model_name == "DEEPAR":
        model_class = DeepAR
        extra_args = {}
    else:
        raise ValueError(f"Unknown model {model_name}")

    if model_name == "NBEATS":
        model = NBeats.from_dataset(
            train_dataset,
            learning_rate=1e-3,
            loss=MAE(),
            # --- architecture ---
            stack_types=["generic"],
            num_blocks=[3],
            num_block_layers=[4],
            widths=[256],
            # --- regularization ---
            dropout=0.1,
            weight_decay=1e-4,
            # --- training behavior ---
            backcast_loss_ratio=0.0,
            reduce_on_plateau_patience=5,
            log_interval=10,
            log_val_interval=1,
        )
    else:
        model = model_class.from_dataset(
            train_dataset,
            learning_rate=1e-3,
            hidden_size=32,
            dropout=0.1,
            loss=MAE(),
            log_interval=10,
            log_val_interval=1,
            **extra_args,
        )

    # --- 4. Train model ---
    logger.info(f"Training {model_name} model on {len(train_dataset)} samples")

    results_dir = Path("results/dl_models")
    loss_path = results_dir / f"{model_name}_loss_history.json"

    loss_callback = LossHistoryCallback(loss_path)

    trainer = Trainer(
        max_epochs=30,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=0.1,
        callbacks=[loss_callback],
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # --- 5. Inference ---
    prediction = model.predict(test_loader, mode="raw", return_x=True)

    forecasts, targets = convert_predictions_to_forecasts_and_truth(prediction, test_df)

    # ----------------------------
    # 6. Evaluation
    # ----------------------------
    metrics = get_metrics()

    metric_values = compute_metrics_direct(forecasts, targets, metrics)
    statistics = compute_statistics(metric_values)
    horizon_metrics = compute_horizon_metrics_direct(forecasts, targets, metrics)

    # ----------------------------
    # 7. Save results
    # ----------------------------
    results_dir = Path("results/dl_models")
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "statistics": statistics,
        "horizon_metrics": horizon_metrics,
    }

    save_json(results, results_dir / f"{model_name}_results.json")

    logger.info("Finished. Results saved.")


if __name__ == "__main__":
    main()
