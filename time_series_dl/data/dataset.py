import pandas as pd


class ForecastingDataset:
    """
    Sliding window dataset for time-series forecasting.

    Produces (input_window, forecast_window).
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        lag_time: int,
        lead_time: int,
        feature_columns: list[str],
        target_column: str,
    ) -> None:

        assert "date" in dataset.columns or "ds" in dataset.columns

        self.df = dataset[feature_columns]
        self.target_column = target_column

        if "date" in dataset.columns:
            self.dates = dataset["date"]
        else:
            self.dates = dataset["ds"]

        self.lag_time = lag_time
        self.lead_time = lead_time

        self.n_examples = len(dataset) - lag_time - lead_time + 1

        if self.n_examples <= 0:
            raise ValueError("Dataset too small for chosen lag_time and lead_time")

    def __len__(self):
        return self.n_examples

    def __getitem__(self, idx):

        x = self.df.iloc[idx : idx + self.lag_time]
        y = self.df.iloc[idx + self.lag_time : idx + self.lag_time + self.lead_time]

        x_dates = self.dates.iloc[idx : idx + self.lag_time]
        y_dates = self.dates.iloc[idx + self.lag_time : idx + self.lag_time + self.lead_time]

        return x, y, x_dates, y_dates


def load_exchange_dataset(path: str) -> pd.DataFrame:
    """
    Load Bank of Canada exchange rate dataset.
    """

    df = pd.read_csv(path)

    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
):
    """
    Chronological train/val/test split.
    """

    n = len(df)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    return train_df, val_df, test_df


def detect_feature_columns(df: pd.DataFrame, target_column: str):
    return [col for col in df.columns if col != "date" and col != target_column] + [target_column]


def build_forecasting_dataset(
    df: pd.DataFrame,
    lag_time: int,
    lead_time: int,
    target_column: str,
):
    feature_columns = detect_feature_columns(df, target_column)

    return ForecastingDataset(
        dataset=df,
        lag_time=lag_time,
        lead_time=lead_time,
        feature_columns=feature_columns,
        target_column=target_column,
    )
