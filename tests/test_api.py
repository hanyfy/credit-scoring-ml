"""Test d'API sur un petit modèle entraîné sur données jouets (sans réseau)."""
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from app import main


@pytest.fixture
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"duration": rng.integers(6, 60, 200).astype(float), "purpose": rng.choice(["car", "tv"], 200)})
    y = (X["duration"] + rng.normal(0, 8, 200) > 33).astype(int)
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    model = Pipeline([("pre", ColumnTransformer([("c", OneHotEncoder(), ["purpose"])], remainder="passthrough")),
                      ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    schema = {"numeric": {"duration": {"min": 6, "max": 60, "median": 24}},
              "categorical": {"purpose": {"values": ["car", "tv"], "mode": "car"}}}
    path = tmp_path / "m.joblib"
    joblib.dump({"model": model, "schema": schema, "threshold": 0.5}, path)
    monkeypatch.setattr(main, "MODEL_PATH", str(path))
    main.get_artifact.cache_clear()
    return TestClient(main.app)


def test_predict_ok(client):
    r = client.post("/predict", json={"features": {"duration": 48, "purpose": "car"}})
    assert r.status_code == 200 and 0 <= r.json()["default_probability"] <= 1


def test_predict_rejects_unknown_category_and_missing_field(client):
    assert client.post("/predict", json={"features": {"duration": 12, "purpose": "boat"}}).status_code == 422
    assert client.post("/predict", json={"features": {"duration": 12}}).status_code == 422


def test_schema_and_index(client):
    assert "duration" in client.get("/schema").json()["numeric"]
    assert client.get("/").status_code == 200
