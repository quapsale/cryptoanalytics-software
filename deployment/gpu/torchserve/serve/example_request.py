import requests
import json

url = 'http://0.0.0.0:8080/predictions/cryptoanalytics'

payload = json.dumps({
    'n_steps': 7,
    'target': 'btc',
    'features': [
        'eth',
        'xrp',
        'ada',
        'doge'
    ]
})
headers = {
    'Content-Type': 'application/json'
}

if __name__ == '__main__':
    response = requests.request('POST', url, headers=headers, data=payload)
    print(response.text)
