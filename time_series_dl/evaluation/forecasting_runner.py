import logging

import pandas as pd
from sktime.forecasting.naive import NaiveForecaster

logger = logging.getLogger(__name__)

def run_rolling_forecast(
    model: NaiveForecaster, dataset: pd.DataFrame, target_column: str, lead_time: int
) -> list[pd.Series]:
    forecasts = []

    fh = list(range(1, lead_time + 1))

    for i in range(len(dataset)):
        if i % 25 == 0:
            logger.info(f"Process {i+1}/{len(dataset)}")

        x, y, x_dates, y_dates = dataset[i]

        fc = model.fit_predict(x[target_column], fh=fh)
        fc = pd.Series(fc.values, index=y_dates)

        forecasts.append(fc)

    return forecasts
