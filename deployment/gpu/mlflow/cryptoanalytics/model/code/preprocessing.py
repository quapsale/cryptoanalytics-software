import pandas as pd
import torch
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import Holt


def preprocess(data):
    """
    Transform raw input into model input data.
    :param batch: list of raw requests, should match batch size
    :return: list of preprocessed model input data
    """
    # Take the input data and make it inference ready
    n_steps = data.get('n_steps').item()
    target = data.get('target').item()
    features = data.get('features').tolist()
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
    model_input = {'input_data': pred_set, 'X_min': X_min, 'X_scale': X_scale}
    return model_input
