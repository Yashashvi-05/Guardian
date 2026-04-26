"""
GUARDIAN HITL FastAPI Router
==============================
Exposes the /hitl/pending and /hitl/decision endpoints consumed by:
  - n8n_guardian_workflow.json (POST /hitl/decision from n8n)
  - OpenClaw poller (GET /hitl/pending every 10s)

Mount this router in guardian/server/app.py:
    from guardian.hitl.hitl_router import hitl_router
    app.include_router(hitl_router, prefix="")

Endpoints
---------
GET  /hitl/pending
    Returns all escalations awaiting a human decision.
    Body: { "<context_id>": { "payload": {...}, "age_s": 12.3 }, ... }

POST /hitl/decision
    Body: { "context_id": "<id>", "decision": "allow|block|shadow" }
    Called by n8n when a human clicks a Telegram inline button.
    Returns: { "ok": true, "context_id": "<id>", "decision": "shadow" }

GET  /hitl/status/<context_id>
    Returns the current resolution status of a specific escalation.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from guardian.hitl.hitl_manager import get_hitl_manager, VALID_DECISIONS

hitl_router = APIRouter(tags=["HITL"])


# ── Pydantic models ────────────────────────────────────────────────────────────

class DecisionPayload(BaseModel):
    context_id: str
    decision: str   # "allow" | "block" | "shadow"


# ── Endpoints ──────────────────────────────────────────────────────────────────

@hitl_router.get("/hitl/pending")
def get_pending_escalations():
    """
    Returns all escalations waiting for a human decision.
    Used by OpenClaw to poll for open gray-zone alerts.
    """
    manager = get_hitl_manager()
    manager.cleanup_resolved()
    return manager.list_pending()


@hitl_router.post("/hitl/decision")
def post_human_decision(payload: DecisionPayload):
    """
    Receives a human decision from n8n (Telegram callback).
    Always returns 200 so n8n proceeds to send the Telegram confirmation.
    If context_id is not in memory (server restarted), logs a headless entry.
    """
    manager = get_hitl_manager()
    with open("decision_log.txt", "a") as f:
        f.write(f"RECEIVED DECISION: {payload.context_id} -> {payload.decision}\n")

    if payload.decision not in VALID_DECISIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision '{payload.decision}'. Must be one of: {sorted(VALID_DECISIONS)}"
        )

    resolved = manager.resolve(payload.context_id, payload.decision)
    # Always return 200 — even if context was not in memory (headless entry written)
    return {
        "ok": True,
        "context_id": payload.context_id,
        "decision": payload.decision,
        "message": f"Decision '{payload.decision}' recorded for escalation {payload.context_id}.",
        "was_pending": resolved is not None,
    }


@hitl_router.get("/hitl/status/{context_id}")
def get_escalation_status(context_id: str):
    """Returns the current resolution status of a specific escalation."""
    manager = get_hitl_manager()
    pending = manager.list_pending()

    if context_id in pending:
        return {"context_id": context_id, "status": "pending", **pending[context_id]}

    # Check if resolved (not in pending means resolved or unknown)
    return {"context_id": context_id, "status": "resolved_or_unknown"}
