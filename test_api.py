from fastapi.testclient import TestClient
from app.main import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_returns_probability_and_label():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "You are a helpful and kind person."})
    body = response.json()
    assert response.status_code == 200
    assert body["label"] in {"toxic", "non-toxic"}
    assert 0 <= body["toxic_probability"] <= 1
    assert body["threshold"] == 0.50


def test_blank_text_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
    assert "Invalid request" in response.json()["detail"]


def test_missing_text_is_rejected():
    with TestClient(app) as client:
        response = client.post("/predict", json={})
    assert response.status_code == 422
