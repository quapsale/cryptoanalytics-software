import bentoml
import numpy as np
import pandas as pd
from bentoml._internal.io_descriptors import JSON, NumpyNdarray
from pydantic import BaseModel
import torch
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import Holt
import warnings

warnings.filterwarnings('ignore')

runner = bentoml.pytorch_lightning.get('cryptoanalytics:latest').to_runner()


class CryptoFeatures(BaseModel):
    n_steps: int
    target: str
    features: list


input_spec = JSON(pydantic_model=CryptoFeatures)

svc = bentoml.Service('cryptoanalytics', runners=[runner])


@svc.api(input=input_spec, output=NumpyNdarray())
def cryptoanalytics(input_series: CryptoFeatures) -> np.ndarray:
    n_steps = input_series.n_steps
    target = input_series.target
    features = input_series.features
    now = datetime.now()
    test = pd.read_csv('valid.csv', sep=',')
    last_valid_date = test.iloc[-1].Date
    last_valid_date = datetime.strptime(last_valid_date, '%Y-%m-%d')
    before = (now - last_valid_date).days
    n_steps_with_before = n_steps + before
    test.set_index('Date', inplace=True)
    test_scaler = MinMaxScaler()
    test_index = pd.DatetimeIndex(test.index, freq='D')
    test_scaled = pd.DataFrame(test_scaler.fit_transform(test),
                               index=test_index, columns=test.columns)

    target_scaler = MinMaxScaler()
    target_scaler.fit_transform(test[target].values.reshape(-1, 1))
    X_min = target_scaler.min_
    X_scale = target_scaler.scale_

    pred_set = pd.DataFrame()
    for feature in features:
        model = Holt(test_scaled[feature])
        fit = model.fit(smoothing_level=0.3, smoothing_trend=0.05, optimized=False)
        pred = fit.forecast(n_steps_with_before)
        pred_set = pd.concat([pred_set, pred], axis=1)
    pred_set.columns = features
    pred_set = pred_set.tail(n_steps)
    pred_set = torch.tensor(pred_set.values, dtype=torch.float32).unsqueeze(1)
    output = runner.run(pred_set)

    def inv_scale(x):
        x -= X_min
        x /= X_scale
        return x

    postprocess_output = output.cpu().detach().numpy()
    result = np.vectorize(inv_scale, otypes=[float])(postprocess_output)
    return result
