import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'nitrogen':3, 'phosphorus':37, 'potassium':25, 'temperature':32, 'humidity':74,'soil_ph':5, 'rainfall':10})

print(r.json())