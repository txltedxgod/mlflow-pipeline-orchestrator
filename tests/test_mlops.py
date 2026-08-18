from mlops.pipeline import MLOpsPipeline

def test_drift_and_retrain():
    pipe = MLOpsPipeline()
    res = pipe.run_training_cycle([1.0, 1.1, 1.0], [5.0, 5.2, 5.1])
    assert res["drift_analysis"]["has_drift"] is True
    assert res["retrained"] is True
    assert res["active_model_version"] == "v2.0"
