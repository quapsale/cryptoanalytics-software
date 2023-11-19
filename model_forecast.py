"""
File: pretrain.py
Description: Model forecast.
File Created: 06/09/2023
Python Version: 3.9
"""

# Imports
import json
import os
import argparse
import sys
import warnings
import pandas as pd
import torch
from datetime import datetime
import numpy as np
from statsmodels.tsa.holtwinters import Holt
from sklearn.preprocessing import MinMaxScaler
from pretrain.gru import GRU
from pretrain.lstm import LSTM
import xgboost as xgb
import catboost
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Parser for CLI
parser = argparse.ArgumentParser(description='forecast crypto-coins prices with ML models')

# Horizon
parser.add_argument('-hz', '--horizon', type=int, nargs=1, help='forecasting horizon')

# Valid
parser.add_argument('-vd', '--valid', type=str, nargs=1, help='path to the csv valid dataset')

# Pretrained
parser.add_argument('-pt', '--pretrained', type=str, nargs=1, help='path to the pretrained model')

# Target
parser.add_argument('-t', '--target', type=str, nargs=1, help='target coin to predict (same as pretraining)')

# Features
parser.add_argument('-ft', '--features', type=str, nargs=1,
                    help='path for json with feature variable list (same as pretraining)')

# Model
parser.add_argument('-m', '--model', type=str, nargs=1, help='model to use for inference')

# Path
parser.add_argument('-p', '--path', type=str, nargs='?', default=os.getcwd(),
                    help='path for saving the predictions (default is current directory)')

# Filename
parser.add_argument('-f', '--filename', type=str, nargs='?',
                    help='filename for predictions (default is predictions_model_TODAY)')

# Arg parse
args = parser.parse_args()

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Warning (invariance respect to training)
print('User info: for the prediction phase, always use the same settings of '
      'the pre-training (target, features).')

# Exception (invalid path)
if not (os.path.exists(args.path)):
    print('Invalid path provided: folder does not exist!')
    sys.exit(1)

# Validate horizon
if not args.horizon:
    print('Missing argument: --horizon is required!')
    sys.exit(1)

# Validate features
if not args.features:
    print('Missing argument: --features is required!')
    sys.exit(1)

# Validate target
if not args.target:
    print('Missing argument: --target is required!')
    sys.exit(1)

# Validate valid
if not args.valid:
    print('Missing argument: --valid is required!')
    sys.exit(1)

# Validate pretrained
if not args.pretrained:
    print('Missing argument: --pretrained is required!')
    sys.exit(1)

# Validate model
if not args.model:
    print('Missing argument: --model is required!')
    sys.exit(1)

# Validate args
(target,) = args.target
(mdl,) = args.model
(horizon,) = args.horizon
(prt,) = args.pretrained
(feat,) = args.features
(valid,) = args.valid

# Validate filename
if not args.filename:
    now = datetime.now()
    today = datetime.strftime(now, '%d-%m-%Y')
    filename = 'predictions_' + mdl + '_' + today
    filename = filename.replace('-', '')

else:
    filename = args.filename

# Print args
print({'--horizon': horizon, '--target': target, '--features': feat, '--pretrained': prt,
       '--valid': valid, '--model': mdl,  '--path': args.path, '--filename': filename})

# Predict

try:
    f = open(feat)
    features = json.load(f)

    try:
        features = features['features']

        # Exception (not a list)
        if not isinstance(features, list):
            print('Invalid data format: wrong features list provided!')
            sys.exit(1)

        # Target in features
        if target in features:
            print('Invalid data format: target is in features list!')
            sys.exit(1)

        # Create prediction set
        try:
            test = pd.read_csv(valid, sep=',')
            now = datetime.now()
            last_valid_date = test.iloc[-1].Date
            last_valid_date = datetime.strptime(last_valid_date, '%Y-%m-%d')
            before = (now - last_valid_date).days
            n_steps_with_before = horizon + before
            test.set_index('Date', inplace=True)
            test_scaler = MinMaxScaler()
            test_index = pd.DatetimeIndex(test.index, freq='D')
            test_scaled = pd.DataFrame(test_scaler.fit_transform(test),
                                       index=test_index, columns=test.columns)

            target_scaler = MinMaxScaler()
            target_scaler.fit_transform(test[target].values.reshape(-1, 1))
            X_min = target_scaler.min_
            X_scale = target_scaler.scale_

            def inv_scale(x):
                x -= X_min
                x /= X_scale
                return x

            pred_set = pd.DataFrame()
            for feature in features:
                holt = Holt(test_scaled[feature])
                fit = holt.fit(smoothing_level=0.3, smoothing_trend=0.05, optimized=False)
                pred = fit.forecast(n_steps_with_before)
                pred_set = pd.concat([pred_set, pred], axis=1)
            pred_set.columns = features
            pred_set = pred_set.tail(horizon)

            # GRU
            if mdl == 'gru':
                # Predict
                pred_set = torch.tensor(pred_set.values, dtype=torch.float32).unsqueeze(1).to(device)
                model = GRU().to(device)
                model.load_state_dict(torch.load(prt))
                predictions = []
                output = model(pred_set)
                postprocess_output = output.cpu().detach().numpy()
                result = np.array(np.vectorize(inv_scale, otypes=[float])(postprocess_output))
                # Create output
                file_name = os.path.join(args.path, filename + '.txt')
                with open(file_name, 'w') as f:
                    f.write(str(result))
                    print(result)

            # LSTM
            if mdl == 'lstm':
                # Predict
                pred_set = torch.tensor(pred_set.values, dtype=torch.float32).unsqueeze(1).to(device)
                model = LSTM().to(device)
                model.load_state_dict(torch.load(prt))
                predictions = []
                output = model(pred_set)
                postprocess_output = output.cpu().detach().numpy()
                result = np.array(np.vectorize(inv_scale, otypes=[float])(postprocess_output))
                # Create output
                file_name = os.path.join(args.path, filename + '.txt')
                with open(file_name, 'w') as f:
                    f.write(str(result))
                    print(result)

            # XGBoost
            if mdl == 'xgboost':
                # Predict
                model = xgb.XGBRegressor()
                model.load_model(prt)
                predictions = model.predict(pred_set, ntree_limit=model.best_iteration)
                result = np.array(np.vectorize(inv_scale, otypes=[float])(predictions))
                # Create output
                file_name = os.path.join(args.path, filename + '.txt')
                with open(file_name, 'w') as f:
                    f.write(str(result))
                    print(result)

            # LightGBM
            if mdl == 'lightgbm':
                # Predict
                model = lgb.Booster(model_file=prt)
                predictions = model.predict(pred_set)
                result = np.array(np.vectorize(inv_scale, otypes=[float])(predictions))
                # Create output
                file_name = os.path.join(args.path, filename + '.txt')
                with open(file_name, 'w') as f:
                    f.write(str(result))
                    print(result)

            # CatBoost
            if mdl == 'catboost':
                # Predict
                model = catboost.CatBoostRegressor()
                model.load_model(prt)
                predictions = model.predict(pred_set)
                result = np.array(np.vectorize(inv_scale, otypes=[float])(predictions))
                # Create output
                file_name = os.path.join(args.path, filename + '.txt')
                with open(file_name, 'w') as f:
                    f.write(str(result))
                    print(result)

            # Not allowed
            elif mdl not in ['gru', 'lstm', 'xgboost', 'lightgbm', 'catboost']:
                print('Invalid model selected: allowed are gru, lstm, xgboost, lightgbm and catboost!')
                sys.exit(1)

        # Exception: file not found
        except FileNotFoundError:
            print('Invalid path provided: csv file does not exist!')
            sys.exit(1)

    # Exception: bad formatted json
    except KeyError:
        print('Invalid data format: wrong features list provided!')
        sys.exit(1)

# Exception: file not found
except FileNotFoundError:
    print('Invalid path provided: json file does not exist!')
    sys.exit(1)

# Exception: not a json
except ValueError:
    print('Invalid data format: features list is not a json!')
    sys.exit(1)


