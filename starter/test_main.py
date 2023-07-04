from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_get_path():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Hello Visitor!"

# Example
def test_get_path_query():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"fetch": "Fetched 5 of 42"}

# Example
def test_get_malformed():
    r = client.get("/")
    assert r.status_code != 200