import typing

from sktime.forecasting.naive import NaiveForecaster


def build_baseline_models(lag_time: int) -> typing.Dict[str, NaiveForecaster]:
    return {
        "persistence": NaiveForecaster(strategy="last"),
        "mean_window": NaiveForecaster(strategy="mean", window_length=lag_time),
        "drift": NaiveForecaster(strategy="drift"),
    }
