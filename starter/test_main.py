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

