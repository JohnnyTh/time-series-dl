import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mase(y_true, y_pred, y_train, m=1):
    naive_errors = np.abs(y_train[m:] - y_train[:-m])
    mae_naive = np.mean(naive_errors)

    mae_forecast = mean_absolute_error(y_true, y_pred)
    return mae_forecast / mae_naive


def get_metrics():
    return {
        "mse": mean_squared_error,
        "rmse": rmse,
        "mae": mean_absolute_error,
        "mape": mean_absolute_percentage_error,
    }
