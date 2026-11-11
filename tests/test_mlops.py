import pytest
from fastapi.testclient import TestClient

from mlops.api import app
from mlops.pipeline import MLOpsPipeline

client = TestClient(app)


def test_drift_and_retrain() -> None:
    pipe = MLOpsPipeline()
    res = pipe.run_training_cycle([1.0, 1.1, 1.0], [5.0, 5.2, 5.1])
    assert res["drift_analysis"]["has_drift"] is True
    assert res["retrained"] is True
    assert res["active_model_version"] == "v2.0"

    no_drift = pipe.run_training_cycle([1.0, 1.0], [1.01, 1.01])
    assert no_drift["drift_analysis"]["has_drift"] is False
    assert no_drift["retrained"] is False
    assert no_drift["active_model_version"] == "v2.0"


def test_drift_validation_rejects_empty_and_non_finite() -> None:
    pipe = MLOpsPipeline()
    with pytest.raises(ValueError, match="non-empty"):
        pipe.run_training_cycle([], [1.0])
    with pytest.raises(ValueError, match="finite"):
        pipe.run_training_cycle([1.0, float("nan")], [1.0])


def test_mlops_api_endpoints() -> None:
    assert client.get("/health").json() == {"status": "ok", "service": "mlops-pipeline"}

    ok = client.post(
        "/api/v1/drift-check",
        json={"reference": [10.0, 10.0], "current": [10.0, 10.0]},
    )
    assert ok.status_code == 200
    assert ok.json()["drift_analysis"]["has_drift"] is False

    bad = client.post("/api/v1/drift-check", json={"reference": [], "current": [1.0]})
    assert bad.status_code == 400
