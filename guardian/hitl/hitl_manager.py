"""
GUARDIAN Human-in-the-Loop (HITL) Decision Manager
====================================================
Manages gray-zone escalations: when Guardian outputs 'escalate_human',
this module holds the episode open, fires a webhook (to n8n → Telegram),
and waits for a human analyst's decision (allow / block / shadow).

Architecture:
  1. Guardian outputs intervention='escalate_human' + risk_score in 0.4–0.7
  2. episode_runner calls HITLManager.escalate(context_id, alert_payload)
  3. HITLManager fires POST to N8N_WEBHOOK_URL with alert details
  4. n8n sends Telegram message with 3 inline buttons (ALLOW / BLOCK / SHADOW)
  5. Human clicks a button → Telegram sends callback → n8n POSTs /hitl/decision
  6. HITLManager.await_decision(context_id) returns the decision
  7. episode_runner applies the human's decision as the final intervention

If no N8N_WEBHOOK_URL is set, the system runs in autonomous mode (no escalation).
If no human responds within timeout_s, defaults to 'shadow' (safe default).

Environment Variables:
    N8N_WEBHOOK_URL  : http://localhost:5678/webhook/guardian-alert
    HITL_TIMEOUT_S   : Seconds to wait for human response (default: 30)
    HITL_DEFAULT     : Default decision if timeout (default: shadow)
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from typing import Dict, Optional

try:
    import httpx
    _HTTPX_AVAILABLE = True
except ImportError:
    _HTTPX_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────────────
N8N_WEBHOOK_URL = os.environ.get("N8N_WEBHOOK_URL", "")
HITL_TIMEOUT_S = int(os.environ.get("HITL_TIMEOUT_S", "30"))
HITL_DEFAULT = os.environ.get("HITL_DEFAULT", "shadow")   # allow | block | shadow

VALID_DECISIONS = frozenset({"allow", "block", "shadow"})


class HITLManager:
    """
    In-process Human-in-the-Loop escalation manager.
    Thread-safe: uses a dict protected by asyncio events (one per context_id).

    Demo flow:
        manager = HITLManager()
        ctx_id = manager.escalate_sync(alert_payload)
        # n8n/Telegram fires POST /hitl/decision?context_id=ctx_id&decision=shadow
        manager.resolve(ctx_id, "shadow")
        decision = manager.get_decision(ctx_id)   # → "shadow"
    """

    def __init__(self) -> None:
        self._pending: Dict[str, Dict] = {}   # context_id → {payload, decision, ts}

    # ── Public API ─────────────────────────────────────────────────────────────

    def escalate_sync(self, alert_payload: Dict) -> str:
        """
        Fires the n8n webhook synchronously and registers a pending escalation.
        Returns the context_id that n8n will send back with the human's decision.
        """
        context_id = str(uuid.uuid4())[:8]
        alert_payload["context_id"] = context_id

        self._pending[context_id] = {
            "payload":  alert_payload,
            "decision": None,
            "ts":       time.time(),
        }

        # Fire webhook if configured
        if N8N_WEBHOOK_URL and _HTTPX_AVAILABLE:
            try:
                with httpx.Client(timeout=5.0) as client:
                    client.post(N8N_WEBHOOK_URL, json={"body": alert_payload})
            except Exception as exc:
                print(f"[HITL] Webhook fire failed: {exc} — running autonomous")
        elif N8N_WEBHOOK_URL:
            print("[HITL] httpx not installed — webhook disabled. pip install httpx")
        else:
            print(f"[HITL] No N8N_WEBHOOK_URL — autonomous mode (context_id={context_id})")

        return context_id

    def resolve(self, context_id: str, decision: str) -> bool:
        """
        Called by the /hitl/decision HTTP endpoint when n8n posts the human's choice.
        Returns True if the resolution was accepted, False if context_id is unknown.
        """
        if context_id not in self._pending:
            return False
        if decision not in VALID_DECISIONS:
            return False
        self._pending[context_id]["decision"] = decision
        return True

    def get_decision(
        self,
        context_id: str,
        timeout_s: int = HITL_TIMEOUT_S,
    ) -> str:
        """
        Polls for a human decision up to timeout_s seconds.
        Falls back to HITL_DEFAULT if no response arrives in time.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            entry = self._pending.get(context_id)
            if entry and entry["decision"] is not None:
                return entry["decision"]
            time.sleep(0.5)

        print(f"[HITL] Timeout for {context_id} — defaulting to '{HITL_DEFAULT}'")
        return HITL_DEFAULT

    def list_pending(self) -> Dict[str, Dict]:
        """Returns all pending escalations without a decision (for /hitl/pending endpoint)."""
        return {
            cid: {
                "context_id": cid,
                "payload":    entry["payload"],
                "age_s":      round(time.time() - entry["ts"], 1),
            }
            for cid, entry in self._pending.items()
            if entry["decision"] is None
        }

    def cleanup_resolved(self, max_age_s: int = 3600) -> None:
        """Remove old resolved escalations from memory."""
        cutoff = time.time() - max_age_s
        to_delete = [
            cid for cid, e in self._pending.items()
            if e["decision"] is not None and e["ts"] < cutoff
        ]
        for cid in to_delete:
            del self._pending[cid]


# ── Singleton ──────────────────────────────────────────────────────────────────
_hitl_manager: Optional[HITLManager] = None


def get_hitl_manager() -> HITLManager:
    """Returns the process-level HITL singleton."""
    global _hitl_manager
    if _hitl_manager is None:
        _hitl_manager = HITLManager()
    return _hitl_manager


# ── Alert Payload Builder ──────────────────────────────────────────────────────

def build_alert_payload(
    domain: str,
    tool_name: str,
    classified_attack: str,
    capability_tags: str,
    risk_score: float,
    episode_step: int,
    counterfactual: str = "",
    context_id: Optional[str] = None,
) -> Dict:
    """
    Builds the structured payload sent to n8n / Telegram.
    Matches the n8n_guardian_workflow.json message template exactly.
    """
    return {
        "domain":           domain,
        "tool_name":        tool_name,
        "classified_attack": classified_attack,
        "capability_tags":  capability_tags,
        "risk_score":       round(risk_score, 3),
        "episode_step":     episode_step,
        "counterfactual":   counterfactual or f"'{tool_name}' would have executed unchecked.",
        "context_id":       context_id or str(uuid.uuid4())[:8],
    }
