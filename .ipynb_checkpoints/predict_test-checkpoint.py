import requests

url = 'http://localhost:9696/predict'

sample_features = {
    'AMBIENT_TEMPERATURE': 25.0,
    'MODULE_TEMPERATURE': 50.0,
    'IRRADIATION': 0.5,
    'hour': 12,
    'minute': 30,
    'month': 6,
    'day_of_year': 150
}

try:
    response = requests.post(url, json=sample_features)
    if response.status_code == 200:
        print(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to the server. Is app.py running?")
