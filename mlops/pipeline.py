from __future__ import annotations

from typing import Any

import numpy as np


class DriftDetector:
    """Data drift detector validating distribution mean shifts."""

    def detect_drift(
        self,
        reference_data: list[float],
        current_data: list[float],
        threshold: float = 0.05,
    ) -> dict[str, Any]:
        if not reference_data or not current_data:
            raise ValueError("reference_data and current_data must be non-empty")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        ref_arr = np.asarray(reference_data, dtype=float)
        curr_arr = np.asarray(current_data, dtype=float)
        if not (np.all(np.isfinite(ref_arr)) and np.all(np.isfinite(curr_arr))):
            raise ValueError("input series must contain only finite numbers")

        ref_mean = float(np.mean(ref_arr))
        curr_mean = float(np.mean(curr_arr))
        denom = abs(ref_mean) if ref_mean != 0 else 1.0
        drift_delta = abs(ref_mean - curr_mean) / denom
        return {
            "has_drift": bool(drift_delta > threshold),
            "drift_score": round(drift_delta, 4),
            "ref_mean": round(ref_mean, 3),
            "curr_mean": round(curr_mean, 3),
        }


class MLOpsPipeline:
    """Automated drift-aware retraining version controller."""

    def __init__(self) -> None:
        self.detector = DriftDetector()
        self.model_version = 1

    def run_training_cycle(
        self, ref_data: list[float], new_data: list[float]
    ) -> dict[str, Any]:
        drift = self.detector.detect_drift(ref_data, new_data)
        reloaded = False
        if drift["has_drift"]:
            self.model_version += 1
            reloaded = True
        return {
            "drift_analysis": drift,
            "retrained": reloaded,
            "active_model_version": f"v{self.model_version}.0",
        }
