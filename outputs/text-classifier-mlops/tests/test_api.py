from fastapi.testclient import TestClient

from text_classifier.api import create_app


class FakePredictor:
    metadata = {"model_version": "test", "labels": ["billing"]}

    def predict(self, text: str, request_id: str):
        return {
            "label": "billing",
            "confidence": 0.9,
            "model_version": "test",
            "request_id": request_id,
        }


def test_predict_and_edge_cases():
    with TestClient(create_app(FakePredictor())) as client:
        response = client.post("/predict", json={"text": "charged twice"})
        assert response.status_code == 200
        assert response.json()["label"] == "billing"
        assert client.post("/predict", json={"text": "   "}).status_code == 422
        assert client.post("/predict", json={}).status_code == 422
        assert client.post("/predict", json={"text": "valid", "extra": 1}).status_code == 422
