# Instructions for Deployment
This document contains the necessary information to deploy CryptoAnalytics with four frameworks: TorchServe, 
BentoML, MLFlow (base) and MLFlow + MLServer. 

## Preliminaries
The Python version used in this project is 3.9.10. 
For each deployment strategy we provided a configuration for both CPU and GPU-enabled machines.
For GPU machines, please install in your environment a [CUDA-enabled version](https://download.pytorch.org/whl/cu118) of PyTorch.

## TorchServe
To deploy CryptoAnalytics on TorchServe, enter the directory /torchserve/serve inside the appropriate path (/cpu or /gpu, 
depending on the configuration used). Therefore, start the service with the following command.

```bash
torchserve --start --ncs --model-store model-store --models cryptoanalytics=cryptoanalytics.mar --ts-config config.properties
```
This will start a new TorchServe instance at the address http://0.0.0.0:8080. Here follows an example call to the server 
using curl:

```bash
curl -X POST "http://0.0.0.0:8080/predictions/cryptoanalytics" -H "Content-Type: application/json" -d '{"n_steps": 7, "target": "btc", "features": ["eth", "xrp", "ada", "doge"]}' 
```
Another test call (with Python requests) can be found in example_request.py.

## BentoML
To deploy CryptoAnalytics on BentoML, you need first to export the correct BentoML path locating the directory /bentoml 
inside the appropriate path (/cpu or /gpu, depending on the configuration used).

```bash
export BENTOML_HOME='/your/path/to/bentoml'
```

Then, start the service with the following command.

```bash
bentoml serve cryptoanalytics:latest
```
This will start a new BentoML instance at the address http://0.0.0.0:3000. Here follows an example call to the server 
using curl:

```bash
curl -X POST "http://0.0.0.0:3000/cryptoanalytics" -H "Content-Type: application/json" -d '{"n_steps": 7, "target": "btc", "features": ["eth", "xrp", "ada", "doge"]}' 
```

Another test call (with Python requests) can be found in example_request.py.

## MLFlow and MLFlow + MLServer
To deploy CryptoAnalytics on MLFlow, enter the directory /mlflow/cryptoanalytics inside the appropriate path (/cpu or /gpu, 
depending on the configuration used). Therefore, start the service with the following command.

```bash
mlflow models serve --host 0.0.0.0 --port 5000 -m model --env-manager=local
```
This will start a new MLFlow instance at the address http://0.0.0.0:5000. Similarly, you can launch MLFlow with MLServer enabled as it follows.


```bash
mlflow models serve --host 0.0.0.0 --port 5000 -m model --env-manager=local --enable-mlserver
```

Here follows an example call to the server using curl:

```bash
curl -X POST "http://0.0.0.0:5000/invocations" -H "Content-Type: application/json" -d '{"inputs": {"n_steps": 7, "target": "btc", "features": ["eth", "xrp", "ada", "doge"]}}'
```

Another test call (with Python requests) can be found in example_request.py.

## License
[MIT](https://choosealicense.com/licenses/mit/)
