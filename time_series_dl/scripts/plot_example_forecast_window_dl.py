import torch
import matplotlib.pyplot as plt
import pandas as pd

from pytorch_forecasting import (
    NBeats,
    TemporalFusionTransformer,
    DeepAR,
    TimeSeriesDataSet,
    GroupNormalizer,
)

from time_series_dl.data.dataset import load_exchange_dataset


# ----------------------------
# Model factory
# ----------------------------
def load_model(model_name: str, checkpoint_path: str):
    if model_name == "NBEATS":
        return NBeats.load_from_checkpoint(checkpoint_path)
    elif model_name == "TFT":
        return TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    elif model_name == "DEEP_AR":
        return DeepAR.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model {model_name}")


# ----------------------------
# Dataset builder (aligned with training)
# ----------------------------
def build_full_dataset(df: pd.DataFrame, model_name: str) -> TimeSeriesDataSet:
    if model_name == "NBEATS":
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="USD_CLOSE",
            group_ids=["series"],
            min_encoder_length=90,
            max_encoder_length=90,
            min_prediction_length=60,
            max_prediction_length=60,
            time_varying_unknown_reals=["USD_CLOSE"],
            add_relative_time_idx=False,
            add_target_scales=False,
        )
    elif model_name == "TFT":
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="USD_CLOSE",
            group_ids=["series"],
            min_encoder_length=90,
            max_encoder_length=90,
            min_prediction_length=60,
            max_prediction_length=60,
            time_varying_unknown_reals=["USD_CLOSE"],
            time_varying_known_reals=["time_idx", "month", "day_of_week"],
            static_categoricals=["series"],
            add_relative_time_idx=True,
            add_target_scales=True,
        )
    elif model_name == "DEEP_AR":
        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="USD_CLOSE",
            group_ids=["series"],
            min_encoder_length=90,
            max_encoder_length=90,
            min_prediction_length=60,
            max_prediction_length=60,
            time_varying_unknown_reals=["USD_CLOSE"],
            time_varying_known_reals=["month", "day_of_week"],
            add_relative_time_idx=True,
            add_target_scales=True,
            static_categoricals=["series"],  # must have at least one
            target_normalizer=GroupNormalizer(groups=["series"]),
        )
    else:
        raise ValueError(model_name)


# ----------------------------
# Single window extraction
# ----------------------------
def build_single_window(full_dataset, df, forecast_origin_idx):
    enc_len = full_dataset.max_encoder_length
    dec_len = full_dataset.max_prediction_length

    start_idx = forecast_origin_idx - enc_len
    end_idx = forecast_origin_idx + dec_len

    if start_idx < 0 or end_idx > len(df):
        raise ValueError("Invalid forecast origin")

    window_df = df.iloc[start_idx:end_idx].copy()

    return TimeSeriesDataSet.from_dataset(
        full_dataset,
        window_df,
        predict=True,
        stop_randomization=True,
    )


# ----------------------------
# Plot function (generic)
# ----------------------------
def plot_example_forecast_window(
    model_name: str,
    full_dataset: TimeSeriesDataSet,
    checkpoint_path: str,
    df: pd.DataFrame,
    target_column: str,
    start_date: str,
    save_path: str,
):
    start_date = pd.to_datetime(start_date)

    forecast_origin_idx = df.index[df["date"] == start_date][0]

    enc_len = full_dataset.max_encoder_length
    dec_len = full_dataset.max_prediction_length

    encoder_df = df.iloc[forecast_origin_idx - enc_len:forecast_origin_idx]
    decoder_df = df.iloc[forecast_origin_idx:forecast_origin_idx + dec_len]

    # dataset
    dataset = build_single_window(full_dataset, df, forecast_origin_idx)
    dataloader = dataset.to_dataloader(train=False, batch_size=1)

    # model
    model = load_model(model_name, checkpoint_path)
    model = model.cpu()
    model.eval()
    model.freeze()

    # inference
    batch = next(iter(dataloader))

    with torch.no_grad():
        prediction = model(batch[0])
        forecast = prediction["prediction"].squeeze().cpu().numpy()

    if model_name == "DEEP_AR":
        forecast = forecast.mean(-1)
    # ----------------------------
    # Plot
    # ----------------------------
    plt.figure(figsize=(12, 5))

    plt.plot(
        encoder_df["date"],
        encoder_df[target_column],
        label="Historical observations",
        linewidth=2,
        color="blue",
    )

    plt.plot(
        decoder_df["date"],
        decoder_df[target_column],
        label="True future values",
        linewidth=2,
        color="black",
    )

    plt.plot(
        decoder_df["date"],
        forecast,
        linestyle="--",
        linewidth=2,
        label=f"{model_name} forecast",
        color="red"
    )

    plt.axvline(
        x=encoder_df["date"].iloc[-1],
        linestyle="--",
        color="gray",
        alpha=0.8,
        label="Forecast origin",
    )

    plt.xlabel("Date")
    plt.ylabel(target_column)
    plt.title(f"Example Forecast Window with {model_name}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    plt.close()


# ----------------------------
# Main
# ----------------------------
def main():
    model_name = "DEEP_AR"  # change: NBEATS / TFT / DEEPAR

    checkpoint_map = {
        "NBEATS": "checkpoints/NBEATS-epoch=20-val_loss=0.0243.ckpt",
        "TFT": "checkpoints/TFT-epoch=06-val_loss=0.0267-v1.ckpt",
        "DEEP_AR": "checkpoints/DEEP_AR-epoch=29-val_loss=-3.3002.ckpt",
    }

    df = load_exchange_dataset("boc_exchange/dataset.csv")

    # CRITICAL FIX: categorical
    df["series"] = pd.Series("series_0", index=df.index, dtype="category")

    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["month"] = df["date"].dt.month.astype("int")
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int")

    full_dataset = build_full_dataset(df, model_name)

    plot_example_forecast_window(
        model_name=model_name,
        full_dataset=full_dataset,
        checkpoint_path=checkpoint_map[model_name],
        df=df,
        start_date="2016-06-01",
        target_column="USD_CLOSE",
        save_path=f"figures/example_forecast_window_{model_name}.png",
    )


if __name__ == "__main__":
    main()
