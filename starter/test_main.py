from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

# Test get
def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello Visitor!"

# Example
def test_get_malformed():
    r = client.get("/bla")
    assert r.status_code != 200

# Test post low salary
def test_post_low_salary():
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

    response = client.post("/inference/", data=json.dumps(data))
    prediction = response.json()
    
    assert response.status_code == 200
    assert prediction == [0]
    
    
# Test post high salary
def test_post_high_salary():
    data = {
        "age": 50,
        "workclass": "Private",
        "fnlgt": 168212,
        "education": "Doctorate",
        "education-num": 16,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 1902,
        "capital-loss": 0,
        "hours-per-week": 65,
        "native-country": "United-States",
    }

    response = client.post("/inference/", data=json.dumps(data))
    prediction = response.json()
    
    assert response.status_code == 200
    assert prediction == [1]
