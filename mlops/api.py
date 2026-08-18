from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from mlops.pipeline import MLOpsPipeline

app = FastAPI(title="MLOps Pipeline Orchestrator", version="0.1.0")
pipeline = MLOpsPipeline()

class DriftReq(BaseModel):
    reference: List[float]
    current: List[float]

@app.post("/api/v1/drift-check")
def check_drift(req: DriftReq):
    return pipeline.run_training_cycle(req.reference, req.current)

@app.get("/health")
def health():
    return {"status": "ok", "service": "mlops-pipeline"}
