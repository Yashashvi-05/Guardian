"""
GUARDIAN FastAPI Server
========================
OpenEnv-compliant HTTP server for GUARDIAN environment.
Exposes /reset, /step, /state endpoints over HTTP.

Run locally:
    uvicorn guardian.server.app:app --host 0.0.0.0 --port 8000

Docker:
    docker build -t guardian-env .
    docker run -p 8000:8000 guardian-env
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from guardian.models import GuardianAction
from guardian.server.guardian_environment import GUARDIANServerEnvironment


app = FastAPI(title="GUARDIAN Fleet Environment", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_env = GUARDIANServerEnvironment()


class ResetRequest(BaseModel):
    attack_type: Optional[str] = None
    seed: Optional[int] = None


class StepRequest(BaseModel):
    risk_score: float = 0.3
    intervention: str = "allow"
    attack_type: Optional[str] = None
    explanation: str = ""
    rollback_k: int = 2


@app.post("/reset")
def reset(req: ResetRequest):
    obs = _env.reset(attack_type=req.attack_type)
    return {"observation": obs.to_text(), "reward": 0.0, "done": False}


@app.post("/step")
def step(req: StepRequest):
    action = GuardianAction(
        risk_score=req.risk_score,
        intervention=req.intervention,
        attack_type=req.attack_type,
        explanation=req.explanation,
        rollback_k=req.rollback_k,
    )
    result = _env.step(action)
    return {
        "observation": result.observation.to_text(),
        "reward": result.reward,
        "done": result.done,
        "truncated": result.truncated,
        "info": result.info,
    }


@app.get("/state")
def state():
    return _env.state().to_dict()


@app.get("/health")
def health():
    return {"status": "ok", "environment": "guardian-env", "version": "0.2.0"}


@app.get("/")
def root():
    return {
        "name": "GUARDIAN Fleet Environment",
        "version": "0.2.0",
        "endpoints": ["/reset", "/step", "/state", "/health"],
        "attacks": ["authority_spoofing", "prompt_injection", "approval_bypass",
                    "data_exfiltration", "confused_deputy", "approval_laundering",
                    "salami_slicing", "schema_drift_exploit"],
        "interventions": ["allow", "rewrite", "shadow", "interrogate", "reduce_privs",
                          "require_justification", "quarantine_tool", "rollback_k",
                          "emergency_fork", "canary_inject", "escalate_human"],
    }
