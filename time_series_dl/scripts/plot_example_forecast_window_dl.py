import torch
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_forecasting import NBeats
from pytorch_forecasting.data import TimeSeriesDataSet

from time_series_dl.data.dataset import load_exchange_dataset


def build_single_example_dataset(
    full_dataset: TimeSeriesDataSet, example_df: pd.DataFrame
) -> TimeSeriesDataSet:
    """
    Construct a TimeSeriesDataSet for a single example using the original dataset as reference.
    """
    return TimeSeriesDataSet.from_dataset(
        full_dataset, example_df, predict=True, stop_randomization=True
    )


def build_single_window(full_dataset, df, forecast_origin_idx) -> TimeSeriesDataSet:
    """
    Returns a TimeSeriesDataSet containing a single prediction window.

    full_dataset : TimeSeriesDataSet
        Original training dataset to extract parameters from.
    df : pd.DataFrame
        Full dataframe containing the target and features.
    forecast_origin_idx : int
        Index in df corresponding to forecast origin (first prediction step).
    """

    enc_len = full_dataset.max_encoder_length
    dec_len = full_dataset.max_prediction_length

    start_idx = forecast_origin_idx - enc_len
    end_idx = forecast_origin_idx + dec_len

    if start_idx < 0 or end_idx > len(df):
        raise ValueError("Not enough rows before/after forecast origin to create full window")

    single_window_df = df.iloc[start_idx:end_idx].copy()

    single_window_dataset = TimeSeriesDataSet.from_dataset(
        full_dataset,
        single_window_df,
        predict=True,
        stop_randomization=True,
    )

    return single_window_dataset


def plot_example_forecast_window_nbeats(
    full_dataset: TimeSeriesDataSet,
    checkpoint_path: str,
    example_df: pd.DataFrame,
    target_column: str,
    start_date: str,
    save_path: str = "example_forecast_window_nbeats.png",
):
    """
    Generate a forecast figure for a single example using a trained N-BEATS checkpoint.
    """

    start_date = pd.to_datetime(start_date)

    # select the row in example_df corresponding to the forecast origin
    forecast_origin_idx = example_df.index[example_df["date"] == start_date][0]
    encoder_start_idx = max(0, forecast_origin_idx - full_dataset.max_encoder_length)
    encoder_df = example_df.iloc[encoder_start_idx:forecast_origin_idx]
    decoder_df = example_df.iloc[
        forecast_origin_idx : forecast_origin_idx + full_dataset.max_prediction_length
    ]

    # build TimeSeriesDataSet for this single example
    single_example_dataset = build_single_window(full_dataset, example_df, forecast_origin_idx)
    dataloader = single_example_dataset.to_dataloader(batch_size=1, shuffle=False)

    # load model
    model = NBeats.load_from_checkpoint(checkpoint_path)
    model = model.cpu()
    model.eval()
    model.freeze()

    # prediction
    batch = next(iter(dataloader))
    with torch.no_grad():
        prediction = model(batch[0])
        forecast = prediction["prediction"].squeeze().cpu().numpy()

    # plot
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
        label="N-BEATS forecast",
        linestyle="--",
        linewidth=2,
        color="red",
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
    plt.title("Example Forecast Window with N-BEATS")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main() -> None:
    df = load_exchange_dataset("boc_exchange/dataset.csv")
    df["series"] = 0
    df["time_idx"] = (df["date"] - df["date"].min()).dt.days
    df["month"] = df["date"].dt.month.astype("int")
    df["day_of_week"] = df["date"].dt.dayofweek.astype("int")
    group_ids = ["series"]

    # construct the full TimeSeriesDataSet
    full_dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target="USD_CLOSE",
        group_ids=group_ids,
        min_encoder_length=90,
        max_encoder_length=90,
        min_prediction_length=60,
        max_prediction_length=60,
        static_categoricals=[],
        static_reals=[],
        time_varying_known_categoricals=[],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=["USD_CLOSE"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    checkpoint_path = "checkpoints/NBEATS-epoch=20-val_loss=0.0243.ckpt"

    plot_example_forecast_window_nbeats(
        example_df=df,
        full_dataset=full_dataset,
        checkpoint_path=checkpoint_path,
        start_date="2016-06-01",
        target_column="USD_CLOSE",
        save_path="figures/example_forecast_window_nbeats.png",
    )


if __name__ == "__main__":
    main()
