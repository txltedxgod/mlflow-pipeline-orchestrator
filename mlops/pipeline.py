import numpy as np
from typing import Dict, Any, List

class DriftDetector:
    def detect_drift(self, reference_data: List[float], current_data: List[float], threshold: float = 0.05) -> Dict[str, Any]:
        ref_mean = np.mean(reference_data)
        curr_mean = np.mean(current_data)
        drift_delta = abs(ref_mean - curr_mean) / (ref_mean if ref_mean != 0 else 1.0)
        return {
            "has_drift": drift_delta > threshold,
            "drift_score": round(float(drift_delta), 4),
            "ref_mean": round(float(ref_mean), 3),
            "curr_mean": round(float(curr_mean), 3)
        }

class MLOpsPipeline:
    def __init__(self):
        self.detector = DriftDetector()
        self.model_version = 1

    def run_training_cycle(self, ref_data: List[float], new_data: List[float]) -> Dict[str, Any]:
        drift = self.detector.detect_drift(ref_data, new_data)
        reloaded = False
        if drift["has_drift"]:
            self.model_version += 1
            reloaded = True
        return {
            "drift_analysis": drift,
            "retrained": reloaded,
            "active_model_version": f"v{self.model_version}.0"
        }
