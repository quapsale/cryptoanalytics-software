# CryptoAnalytics (Software Artifact)

[![Paper DOI](https://img.shields.io/badge/SoftwareX_2024-10.1016%2Fj.softx.2024.101663-blue.svg)](https://doi.org/10.1016/j.softx.2024.101663)
[![CodeOcean Capsule](https://img.shields.io/badge/CodeOcean-Capsule_7158163-success.svg)](https://codeocean.com/capsule/7158163/tree)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📌 Overview
This repository contains the complete, reproducible software artifact for **CryptoAnalytics**—an end-to-end toolchain for aggregating, preprocessing, analyzing, and forecasting financial cryptocurrency time series. 

This artifact serves as the official open-source codebase for the peer-reviewed paper: *"CryptoAnalytics: Cryptocoins price forecasting with machine learning techniques"* (SoftwareX 2024).

---

## 🚀 Environment Setup & Installation

The core python environment is built on **Python 3.9.10**. For strict reproducibility, we recommend managing your runtime using [`pyenv`](https://github.com/pyenv/pyenv) and [`pyenv-virtualenv`](https://github.com/pyenv/pyenv-virtualenv).

```bash
# 1. Install target Python runtime and spin up a dedicated virtual environment
pyenv install 3.9.10
pyenv virtualenv 3.9.10 venv

# 2. Activate the localized virtual environment
pyenv activate venv

# 3. Install core framework requirements via pip
pip install -r requirements.txt
```

---

## 🧪 Reproducible Live Demo (CodeOcean)
A pre-configured live execution capsule for the **Bitcoin (BTC)** price prediction pipeline is fully operational on CodeOcean. This live environment runs a predefined dataset cycle matching the following boundary parameters:
* **Target Coins:** `btc`, `eth`, `usdt`, `usdc`, `xrp`, `busd`, `ada`, `doge`
* **Temporal Range:** `15-08-2023` to `15-11-2023`

👉 **[Launch and Execute the CodeOcean Capsule](https://codeocean.com/capsule/7158163/tree)**

---

## 📂 Artifact Modular Architecture

### 🎛️ 1. Software Deployment Utilities (`/deployment`)
Contains dedicated production-grade deployment recipes to encapsulate and serve your trained CryptoAnalytics forecasting pipelines across four robust model-serving layers:
* **TorchServe**
* **BentoML**
* **MLflow (Base REST Engine)**
* **MLflow combined with MLServer**
* *See the specialized deployment guide inside `/deployment/README.md` for explicit initialization parameters.*

### 📈 2. Core Price Prediction Flow Pipeline
* `data_pull.py`: Connects directly to the [CoinMarketCap API](https://coinmarketcap.com/) to fetch historical Open-High-Low-Close (OHLC) pricing logs.
* `data_split.py`: Partitions compiled raw matrices into isolated train, validation, and terminal test splits.
* `model_pretrain.py`: Trains selected machine learning estimators for trend estimation.
* `model_forecast.py`: Generates future out-of-sample multi-step horizons from optimized weights.

### 📊 3. Extra Exploratory Analysis
* `correlation_analysis.py`: Computes structural inter-token temporal correlations to guide upstream feature engineering and pre-selection.

---

## 💻 CLI Usage Reference

### Price Prediction Flow

#### A. Data Pulling
```bash
python data_pull.py -p "destination_path" -f "filename" -c "examples/coins.json" -s "01-11-2020" -e "01-11-2023"
```
| Flag | Alternative | Description | Default Status |
| :--- | :--- | :--- | :--- |
| `-p` | `--path` | Output directory destination path | Current working directory |
| `-f` | `--filename` | Output string name for compiled dataset | `dataset_coinmarketcap_START_END` |
| `-c` | `--coins` | Path to `.json` file containing listing array | Top 20 market cap assets |
| `-s` | `--start` | Ingestion sequence start date (`%d-%m-%Y`) | Yesterday |
| `-e` | `--end` | Ingestion sequence terminal date (`%d-%m-%Y`) | Today |

#### B. Data Partitioning
```bash
python data_split.py -p "destination_path" -f "filename1" "filename2" -d "data_path" -v "avg_ohlc" -tr 0.8 -vd 0.2
```
| Flag | Alternative | Description | Default Status |
| :--- | :--- | :--- | :--- |
| `-p` | `--path` | Target output folder for data splits | Current working directory |
| `-f` | `--filenames` | Structured file names for generated splits | `[train_TODAY, valid_TODAY, test_TODAY]` |
| `-d` | `--data` | Complete input source target `.csv` file | **REQUIRED** |
| `-v` | `--variable` | Price tracking variable (`avg_ohlc` \| `close`) | `avg_ohlc` |
| `-tr`| `--train` | Proportional split target ratio for training array | **REQUIRED** |
| `-vd`| `--valid` | Proportional split target ratio for validation array | **REQUIRED** |

#### C. Model Pretraining
```bash
python model_pretrain.py -p "destination_path" -f "filename" -tr "train_path" -vd "valid_path" -t "btc" -ft "examples/features.json" -m "lstm" -c "examples/config_nn.json"
```
| Flag | Alternative | Description | Default Status |
| :--- | :--- | :--- | :--- |
| `-p` | `--path` | Output model binary save location | Current working directory |
| `-f` | `--filename` | Filename tag for the exported model | `model_TODAY` |
| `-tr`| `--train` | Source training split path (`.csv`) | **REQUIRED** |
| `-vd`| `--valid` | Source validation split path (`.csv`) | **REQUIRED** |
| `-t` | `--target` | Target coin ticker signature to predict | **REQUIRED** |
| `-ft`| `--features` | Path to `.json` tracking feature predicting assets | **REQUIRED** |
| `-m` | `--model` | Estimator type (`lstm` \| `gru` \| `xgboost` \| `lightgbm` \| `catboost`) | **REQUIRED** |
| `-c` | `--config` | Parameter configuration configurations map (`.json`) | **REQUIRED** |

#### D. Model Inference Forecasting
```bash
python model_forecast.py -p "destination_path" -f "filename" -hz "horizon" -vd "valid_path" -pt "pretrained_path" -t "btc" -ft "examples/features.json" -m "lstm"
```
| Flag | Alternative | Description | Default Status |
| :--- | :--- | :--- | :--- |
| `-p` | `--path` | Target directory for generated prediction logs | Current working directory |
| `-f` | `--filename` | String filename tag for target prediction dump | `predictions_model_TODAY` |
| `-hz`| `--horizon` | Total forecasting step horizon length | **REQUIRED** |
| `-vd`| `--valid` | Source validation split path (`.csv`) | **REQUIRED** |
| `-pt`| `--pretrained`| Path pointing to pretrained active weights target | **REQUIRED** |
| `-t` | `--target` | Target coin ticker (*Must match pretraining context*) | **REQUIRED** |
| `-ft`| `--features` | Feature config target path (*Must match pretraining context*)| **REQUIRED** |
| `-m` | `--model` | Selected runner template (*Must match pretraining context*) | **REQUIRED** |

---

### Extra: Correlation Analysis Suite

```bash
python correlation_analysis.py -p "destination_path" -f "filename" -d "data_path" -v "avg_ohlc" -w "daily" -m "pearson"
```
| Flag | Alternative | Description | Default Status |
| :--- | :--- | :--- | :--- |
| `-p` | `--path` | Target output path for computed correlation states | Current working directory |
| `-f` | `--filename` | String filename for exported tracking metrics | `correlations_TODAY` |
| `-d` | `--data` | Complete historical raw tracking source file | **REQUIRED** |
| `-v` | `--variable` | Targeting price metrics (`avg_ohlc` \| `close`) | `avg_ohlc` |
| `-w` | `--window` | Analysis tracking window range (`daily` \| `weekly` \| `monthly`) | `daily` |
| `-m` | `--method` | Target matrix algorithm (`pearson` \| `kendall` \| `spearman`) | `pearson` |

*Refer to the `/examples` path directory to find boilerplate setup arrays for structuring custom asset configurations, tuning models, or defining feature tracking configurations.*

---

## ⚠️ Risk Warning & Disclaimer
Financial time series modeling and asset valuation contain inherent volatility vectors. Predictive evaluations generated using this artifact function strictly as algorithmic reference benchmarks and structural guidelines; they do not construct, imply, or constitute professional investment or trading advice.

---

## 👥 Authors
* **Pasquale De Rosa** – *University of Neuchâtel* – [pasquale.derosa@unine.ch](mailto:pasquale.derosa@unine.ch)
* **Pascal Felber** – *University of Neuchâtel* – [pascal.felber@unine.ch](mailto:pascal.felber@unine.ch)
* **Valerio Schiavoni** – *University of Neuchâtel* – [valerio.schiavoni@unine.ch](mailto:valerio.schiavoni@unine.ch)

---

## 📄 Citation
If you implement, adapt, or build upon this computational software framework or the underlying methodologies within an academic context, please cite our tracking publication in SoftwareX:

```bibtex
@article{DEROSA2024101663,
title = {CryptoAnalytics: Cryptocoins price forecasting with machine learning techniques},
journal = {SoftwareX},
volume = {26},
pages = {101663},
year = {2024},
issn = {2352-7110},
doi = {https://doi.org/10.1016/j.softx.2024.101663},
url = {https://www.sciencedirect.com/science/article/pii/S2352711024000347},
author = {Pasquale {De Rosa} and Pascal Felber and Valerio Schiavoni},
keywords = {Cryptocoins, Machine learning, Forecasting}
}
```
