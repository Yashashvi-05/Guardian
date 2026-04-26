"""
GUARDIAN FastAPI Server
========================
OpenEnv-compliant HTTP server for GUARDIAN environment.
Exposes /reset, /step, /state, /hitl/* endpoints over HTTP.

Run locally:
    uvicorn guardian.server.app:app --host 0.0.0.0 --port 8000

Docker:
    docker build -t guardian-env .
    docker run -p 8000:8000 guardian-env

HITL Setup (optional, for n8n/Telegram escalation):
    export N8N_WEBHOOK_URL=http://localhost:5678/webhook/guardian-alert
    export HITL_TIMEOUT_S=30
    # Then configure n8n_guardian_workflow.json and your Telegram bot.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from guardian.models import GuardianAction
from guardian.server.guardian_environment import GUARDIANServerEnvironment
from guardian.hitl.hitl_router import hitl_router
from guardian.hitl.escalation import hitl_manager


app = FastAPI(title="GUARDIAN Fleet Environment", version="0.3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount HITL router — /hitl/pending, /hitl/decision, /hitl/status/<id>
app.include_router(hitl_router)

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


@app.get("/baselines")
def baselines():
    import glob
    eval_dir = "guardian/data/eval_data"
    if not os.path.exists(eval_dir):
        return {"baselines": []}
    results = []
    for path in glob.glob(os.path.join(eval_dir, "baseline_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            results.append(data)
        except Exception:
            pass
    results.sort(key=lambda x: x.get("mean_reward", 0), reverse=True)
    return {"baselines": results}


@app.get("/elo")
def elo_ratings():
    path = "guardian/data/elo_ratings.json"
    if not os.path.exists(path):
        return {"guardian_elo": 1000, "attack_elos": {}, "leaderboard": []}
    with open(path) as f:
        return json.load(f)


@app.get("/compare_checkpoints")
def compare_checkpoints(a: str = "episode_50", b: str = "final"):
    scorecard_file = "guardian/data/scorecards.jsonl"
    if not os.path.exists(scorecard_file):
        return {"error": "No scorecard data found"}
    scorecards = []
    with open(scorecard_file) as f:
        for line in f:
            if line.strip():
                try:
                    scorecards.append(json.loads(line))
                except Exception:
                    pass
    half = len(scorecards) // 2
    checkpoint_a = scorecards[:half]
    checkpoint_b = scorecards[half:]

    def metrics(scs):
        if not scs:
            return {}
        attack_eps = [s for s in scs if s.get("attack_type") != "clean"]
        clean_eps = [s for s in scs if s.get("attack_type") == "clean"]
        detect = sum(1 for s in attack_eps if s.get("reward_components", {}).get("attack_classification_f1", 0) > 0.2)
        return {
            "n": len(scs),
            "mean_reward": round(sum(s.get("reward_total", 0) for s in scs) / len(scs), 4),
            "detection_rate": round(detect / len(attack_eps), 3) if attack_eps else 0,
            "false_alarm_rate": round(sum(1 for s in clean_eps if s.get("fork_triggered")) / len(clean_eps), 3) if clean_eps else 0,
        }

    return {
        "checkpoint_a": {"label": a, **metrics(checkpoint_a)},
        "checkpoint_b": {"label": b, **metrics(checkpoint_b)},
        "improvement": round(metrics(checkpoint_b).get("mean_reward", 0) - metrics(checkpoint_a).get("mean_reward", 0), 4),
    }


# ── WebSocket — live dashboard updates ────────────────────────────────────────
_ws_clients: list = []

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        _ws_clients.remove(websocket)


# ── HITL escalation trigger ────────────────────────────────────────────────────

class EscalateRequest(BaseModel):
    tool_name: str
    classified_attack: str = ""
    risk_score: float = 0.65
    domain: str = "enterprise"
    tool_arguments: dict = {}
    guardian_reasoning: str = ""
    counterfactual: str = ""

@app.post("/hitl/escalate")
def escalate(req: EscalateRequest):
    """
    Trigger a HITL escalation — fires n8n webhook → Telegram alert.
    Used by the dashboard and test scripts.
    """
    from guardian.hitl.escalation import get_counterfactual
    cf = req.counterfactual or get_counterfactual(req.tool_name, req.domain)
    ctx = hitl_manager.create_escalation(
        tool_name=req.tool_name,
        tool_arguments=req.tool_arguments,
        risk_score=req.risk_score,
        classified_attack=req.classified_attack,
        guardian_reasoning=req.guardian_reasoning,
        counterfactual=cf,
        domain=req.domain,
    )
    return {
        "ok": True,
        "context_id": ctx.context_id,
        "risk_score": ctx.risk_score,
        "counterfactual": ctx.counterfactual,
        "message": f"Escalation created — Telegram alert sent via n8n.",
    }


# ── HITL replay stats ─────────────────────────────────────────────────────────

@app.get("/hitl/replay_stats")
def replay_stats():
    """Returns statistics about the human-labeled training replay buffer."""
    return hitl_manager.get_replay_buffer_stats()


@app.get("/")

def root():
    return {
        "name": "GUARDIAN Fleet Environment",
        "version": "0.3.0",
        "endpoints": ["/reset", "/step", "/state", "/health",
                      "/hitl/pending", "/hitl/decision", "/hitl/status/{id}"],
        "attacks": ["authority_spoofing", "prompt_injection", "approval_bypass",
                    "data_exfiltration", "confused_deputy", "approval_laundering",
                    "salami_slicing", "schema_drift_exploit", "rogue_internal_ai"],
        "interventions": ["allow", "rewrite", "shadow", "interrogate", "reduce_privs",
                          "require_justification", "quarantine_tool", "quarantine_agent",
                          "rollback_k", "emergency_fork", "canary_inject", "escalate_human"],
        "hitl": {
            "description": "Human-in-the-Loop gray-zone escalation via n8n + Telegram",
            "workflow": "n8n_guardian_workflow.json",
            "env_vars": ["N8N_WEBHOOK_URL", "HITL_TIMEOUT_S", "HITL_DEFAULT"],
            "openclaw_poll": "GET /hitl/pending every 10s",
            "openclaw_decide": "POST /hitl/decision {context_id, decision}",
        },
    }
