from __future__ import annotations

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

from mlops.pipeline import MLOpsPipeline

app = FastAPI(title="MLOps Pipeline Orchestrator", version="0.1.0")
pipeline = MLOpsPipeline()


class DriftReq(BaseModel):
    reference: list[float]
    current: list[float]


@app.post("/api/v1/drift-check")
def check_drift(req: DriftReq) -> dict:
    try:
        return pipeline.run_training_cycle(req.reference, req.current)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "mlops-pipeline"}
