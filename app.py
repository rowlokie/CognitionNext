#!/usr/bin/env python3
"""
app.py — Fast, CPU-only ASGI Decision Engine Service.
Runs via: uvicorn app:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints:
- POST /v1/decide : Unified decision endpoint (route, approve, grounded)
- GET /health     : Health check endpoint
- GET /           : Root info endpoint
"""

import gc
import sys
import time
from pathlib import Path
from typing import Optional

import orjson
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

import model

ROOT = Path(__file__).parent
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
# App and model lifecycle
# -----------------------------------------------------------------------

app = FastAPI(title="Decide API", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    model.load_model(ARTIFACTS_DIR)
    # Warm up with dummy requests across all three tasks
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
    print("[startup] Model loaded and warmed up. Ready to serve.")


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------

@app.post("/v1/decide")
async def decide(request: DecideRequest):
    state_dict = request.state.model_dump()

    try:
        result = model.predict(state_dict, request.decision, request.request_id)
    except ValueError as e:
        return Response(
            content=orjson.dumps({"detail": str(e)}),
            status_code=422,
            media_type="application/json"
        )

    # Strictly adhere to judging schema: exactly the required fields, no extra keys
    if request.decision == "route":
        public_resp = {
            "request_id": str(result.get("request_id", request.request_id)),
            "decision": "route",
            "type": "choice",
            "value": str(result.get("value", "general")),
            "probs": result.get("probs", {}),
            "confidence": float(result.get("confidence", 0.0)),
            "abstain": bool(result.get("abstain", False)),
            "latency_ms": float(result.get("latency_ms", 0.0)),
        }
    else:  # approve, grounded
        public_resp = {
            "request_id": str(result.get("request_id", request.request_id)),
            "decision": request.decision,
            "type": "binary",
            "value": bool(result.get("value", False)),
            "p_true": float(result.get("p_true", 0.0)),
            "abstain": bool(result.get("abstain", False)),
            "latency_ms": float(result.get("latency_ms", 0.0)),
        }

    return Response(
        content=orjson.dumps(public_resp),
        media_type="application/json"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model.is_loaded()}


@app.get("/")
async def root():
    return {"message": "Decide API v1. POST /v1/decide"}


# -----------------------------------------------------------------------
# Validation error handler
# -----------------------------------------------------------------------

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={"detail": jsonable_encoder(exc.errors()), "body": str(getattr(exc, "body", ""))}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        loop="asyncio",
        log_level="warning",
    )
