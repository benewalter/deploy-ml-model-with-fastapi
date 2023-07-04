import requests
import json

data = {
    "age": 34,
    "workclass": "Private",
    "fnlgt": 245487,
    "education": "7th-8th",
    "education-num": 4,
    "marital-status": "Married-civ-spouse",
    "occupation": "Transport-moving",
    "relationship": "Husband",
    "race": "Amer-Indian-Eskimo",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "Mexico",
}

response = requests.post("https://udacity-salary-prediction-94c031866332.herokuapp.com/inference/", 
                         data=json.dumps(data))

print("Status Code: " + str(response.status_code))
print("Response: " + str(response.json()))
