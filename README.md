# CryptoAnalytics

CryptoAnalytics is a software artifact for the analysis and forecasting of financial time series and cryptocurrency price trends.

## Installation

The Python version used in this project is 3.9.10. 
It is suggested to create a clean virtual environment with the correct version using [pyenv](https://github.com/pyenv/pyenv).

Please refer to [these](https://github.com/pyenv/pyenv#installation) instructions to install pyenv on your current OS. 
To install pyenv-virtualenv as well, please follow [this](https://github.com/pyenv/pyenv-virtualenv#installation) documentation.

Once pyenv has been set up, you can install the required Python version and generate a new virtual environment as it follows.

```bash
pyenv install 3.9.10
pyenv virtualenv 3.9.10 venv
```

With the following you can activate your virtual environment.

```bash
pyenv activate venv
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the project requirements as it follows.

```bash
pip install -r requirements.txt
```

## Demo
A demo for the price prediction flow of Bitcoin (BTC), using a set of predefined args, is available on CodeOcean.

The demo uses a pre-built dataset (using the data_pull command) with the following characteristics:

1. coins: "btc", "eth", "usdt", "usdc", "xrp", "busd", "ada", "doge".

2. start: 15-08-2023.

3. end: 15-11-2023.

To launch the demo, start a reproducible run on our CodeOcean [capsule](https://codeocean.com/capsule/7158163/tree).

## Structure
This artifact is organized as it follows:

### Software Deployment
Set of utilities to deploy CryptoAnalytics in a production environment using four different frameworks: Torchserve, BentoML, Mlflow (base) and Mlflow (with MLServer).
For further instructions, please refer to the README file in the directory /deployment.

### Price Prediction Flow

1. data_pull.py: command to generate a new dataset of OHLC cryptocoin prices from [CoinMarketCap](https://coinmarketcap.com/).
2. data_split.py: command to generate train and validation sets from the original data.
3. model_pretrain.py: command to pretrain ML models for cryptocoin prices forecast.
4. model_forecast.py: command to use the pretrained ML models to forecast cryptocoin prices.

### Extra

1. correlation_analysis.py: command to analyze correlations among cryptocoin prices, can be useful for pre-selection of feature variables.

## Usage

### Price Prediction Flow

#### Data Pull

```bash
python data_pull.py -p "destination_path" -f "filename" -c "examples\coins.json" -s "01-11-2020" -e "01-11-2023"
```
Args:
* -p, --path: destination directory for dataset storage (OPTIONAL, defaults to current directory).
* -f, --filename: file name for pulled dataset (OPTIONAL, defaults to dataset_coinmarketcap_START_END).
* -c, --coins: path to .json file with list of coins to include in the dataset (OPTIONAL, defaults to top-20 coins for market cap).
* -s, --start: starting date for dataset pull, in format %d-%m-%Y (OPTIONAL, defaults to yesterday).
* -e, --end: ending date for dataset pull, in format %d-%m-%Y (OPTIONAL, defaults to today).

#### Data Split

```bash
python data_split.py -p "destination_path" -f "filename1" "filename2" -d "data_path" -v "avg_ohlc" -tr 0.8 -vd 0.2
```
Args:
* -p, --path: destination directory for train, validation and test dataset storage (OPTIONAL, defaults to current directory).
* -f, --filenames: file names for pulled dataset (OPTIONAL, defaults to [train_TODAY, valid_TODAY, test_TODAY]).
* -d, --data: path to original .csv file dataset (REQUIRED).
* -v, --variable: price variable to consider, either **avg_ohlc** or **close** (OPTIONAL, defaults to avg_ohlc).
* -tr, --train: ratio for train set split (REQUIRED).
* -vd, --valid: ratio for validation set split (REQUIRED).

#### Model Pretrain

```bash
python model_pretrain.py -p "destination_path" -f "filename" -tr "train_path" -vd "valid_path" -t "btc" -ft "examples\features.json" -m "lstm" -c "examples\config_nn.json"
```
Args:
* -p, --path: destination directory for pretrained model storage (OPTIONAL, defaults to current directory).
* -f, --filename: file name for pretrained model (OPTIONAL, defaults to model_TODAY).
* -tr, --train: path to the .csv file train dataset (REQUIRED).
* -vd, --valid: path to the .csv file validation dataset (REQUIRED).
* -t, --target: target coin to predict (REQUIRED).
* -ft, --features: path to .json file with list of coins to use as feature/predicting variables (REQUIRED).
* -m, --model: model to pretrain for inference, either **lstm**, **gru**, **xgboost**, **lightgbm** or **catboost** (REQUIRED).
* -c, --config: path to .json file with list of configs to use for pretraining (REQUIRED).

#### Model Forecast

```bash
python model_forecast.py -p "destination_path" -f "filename" -hz "horizon" -vd "valid_path" -pt "pretrained_path" -t "btc" -ft "examples\features.json" -m "lstm"
```
Args:
* -p, --path: destination directory for predictions storage (OPTIONAL, defaults to current directory).
* -f, --filename: file name for predictions (OPTIONAL, defaults to predictions_model_TODAY).
* -hz, --horizon: forecasting horizon (REQUIRED).
* -vd, --valid: path to the .csv file valid dataset (REQUIRED).
* -pt, --pretrained: path to the pretrained model (REQUIRED).
* -t, --target: target coin to predict, **same as pretraining** (REQUIRED).
* -ft, --features: path to .json file with list of coins to use as feature/predicting variables, **same as pretraining** (REQUIRED).
* -m, --model: model to use for inference, **same as pretraining** (REQUIRED).

### Extra

#### Correlation Analysis

```bash
python correlation_analysis.py -p "destination_path" -f "filename" -d "data_path" -v "avg_ohlc" -w "daily" -m "pearson"
```
Args:
* -p, --path: destination directory for correlations storage (OPTIONAL, defaults to current directory).
* -f, --filename: file name for correlations dataset (OPTIONAL, defaults to correlations_TODAY).
* -d, --data: path to .csv file dataset (REQUIRED).
* -v, --variable: price variable on which computing correlations, either **avg_ohlc** or **close** (OPTIONAL, defaults to avg_ohlc).
* -w, --window: sliding time window to use for computations, either **daily**, **monthly** or **weekly** (OPTIONAL, defaults to daily).
* -m, --method: method to compute correlations, either **pearson**, **kendall** or **spearman** (OPTIONAL, defaults to pearson).

## Examples
You can find examples of coin list (for data pull), configs (for pretraining/forecast) and feature variable list (for pretraining/forecast) in the directory /examples.

## License
[MIT](https://choosealicense.com/licenses/mit/)
