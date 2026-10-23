#!/usr/bin/env python3
"""
FastAPI application for the Decide endpoint.
POST /v1/decide

Priority: schema validity > latency > accuracy
Always keeps Phase 1 baseline alive; upgrades to Phase 2 model transparently.
"""

import gc
import os
import time
from pathlib import Path
from typing import Optional

import orjson
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel, field_validator, model_validator

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

# -----------------------------------------------------------------------
# Request / Response schemas
# -----------------------------------------------------------------------

class StateModel(BaseModel):
    subject: Optional[str] = ""
    body: Optional[str] = ""
    customer_tier: Optional[str] = ""
    amount: Optional[float] = 0.0
    draft_reply: Optional[str] = ""
    kb_article: Optional[str] = ""

    @field_validator("subject", "body", "customer_tier", "draft_reply", "kb_article", mode="before")
    @classmethod
    def coerce_none_to_str(cls, v):
        return "" if v is None else str(v)

    @field_validator("amount", mode="before")
    @classmethod
    def coerce_amount(cls, v):
        if v is None:
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0


class DecideRequest(BaseModel):
    request_id: str
    decision: str
    state: StateModel

    @field_validator("decision")
    @classmethod
    def validate_decision(cls, v):
        allowed = {"route", "approve", "grounded"}
        if v not in allowed:
            raise ValueError(f"decision must be one of {sorted(allowed)}, got {v!r}")
        return v

    @field_validator("request_id", mode="before")
    @classmethod
    def coerce_request_id(cls, v):
        if v is None:
            return ""
        return str(v)


# -----------------------------------------------------------------------
# App and model loading
# -----------------------------------------------------------------------

app = FastAPI(title="Decide API", version="1.0.0")

_model = None


def get_model():
    global _model
    if _model is None:
        raise RuntimeError("Model not loaded")
    return _model


def load_model():
    """Load the best available model: Phase 2 encoder or Phase 1 baseline."""
    global _model

    # Try Phase 2 encoder model first
    encoder_path = ARTIFACTS_DIR / "encoder_model.pkl"
    baseline_path = ARTIFACTS_DIR / "baseline_model.pkl"

    if encoder_path.exists():
        try:
            import sys
            sys.path.insert(0, str(ROOT / "src"))
            import pickle
            with open(encoder_path, "rb") as f:
                _model = pickle.load(f)
            print(f"[startup] Loaded Phase 2 encoder model from {encoder_path}")
            return
        except Exception as e:
            print(f"[startup] Phase 2 model failed to load: {e}. Falling back to baseline.")

    if baseline_path.exists():
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        import pickle
        with open(baseline_path, "rb") as f:
            _model = pickle.load(f)

        # Apply calibrated thresholds if available
        thresh_path = ARTIFACTS_DIR / "thresholds.json"
        if thresh_path.exists():
            try:
                import json
                with open(thresh_path, "r") as tf:
                    tdata = json.load(tf)
                if hasattr(_model, "prediction_thresholds") and "prediction" in tdata:
                    _model.prediction_thresholds.update(tdata["prediction"])
                if hasattr(_model, "abstention_thresholds") and "abstention" in tdata:
                    _model.abstention_thresholds.update(tdata["abstention"])
                elif hasattr(_model, "thresholds") and "abstention" in tdata:
                    _model.thresholds.update(tdata["abstention"])
            except Exception as te:
                print(f"[startup] Warning loading thresholds: {te}")

        print(f"[startup] Loaded Phase 1 baseline model from {baseline_path}")
        return

    raise RuntimeError(
        f"No model found in {ARTIFACTS_DIR}. Run: python src/baseline.py"
    )


@app.on_event("startup")
async def startup_event():
    load_model()
    # Warm up with dummy requests
    model = get_model()
    dummy_state = {
        "subject": "test",
        "body": "test body",
        "customer_tier": "free",
        "amount": 0.0,
        "draft_reply": "test reply",
        "kb_article": "test article",
    }
    for decision in ["route", "approve", "grounded"]:
        model.predict(dummy_state, decision, "warmup")
    gc.freeze()
    print("[startup] Model warmed up. Ready to serve.")


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.post("/v1/decide")
async def decide(request: DecideRequest):
    model = get_model()
    state_dict = request.state.model_dump()

    try:
        result = model.predict(state_dict, request.decision, request.request_id)
    except ValueError as e:
        # Unknown decision (should be caught by validator, but belt-and-suspenders)
        return ORJSONResponse(
            status_code=422,
            content={"detail": str(e)}
        )

    return ORJSONResponse(content=result)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/")
async def root():
    return {"message": "Decide API v1. POST /v1/decide"}


# -----------------------------------------------------------------------
# Error handler for validation errors
# -----------------------------------------------------------------------

from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors()), "body": str(getattr(exc, "body", ""))}
    )


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="asyncio",
        log_level="warning",
    )
