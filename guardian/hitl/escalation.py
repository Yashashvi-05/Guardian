"""
GUARDIAN Human-in-the-Loop (HITL) Escalation Module
=====================================================
Implements the "Asynchronous Gray-Zone Freeze" architecture.

When the Guardian's risk_score falls in the AMBIGUITY_ZONE (0.55–0.75),
neither a confident block nor a confident allow is appropriate.
GUARDIAN freezes the action and escalates to a human via Telegram/n8n.

Zones:
  risk < 0.55  → auto-allow  (Guardian confident it is safe)
  risk > 0.75  → auto-block  (Guardian confident it is dangerous)
  0.55 ≤ risk ≤ 0.75 → FREEZE + escalate to human (gray zone)

The human's decision is logged to hitl_replay.jsonl and used as
Ground Truth in the next GRPO training run — closing the HITL loop.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

import requests

log = logging.getLogger(__name__)

# ── Zone thresholds ───────────────────────────────────────────────────────────

AMBIGUITY_LOW  = 0.55   # Below → Guardian auto-allows
AMBIGUITY_HIGH = 0.75   # Above → Guardian auto-blocks/honeypots

HumanDecision = Literal["allow", "block", "shadow"]

# ── Paths ─────────────────────────────────────────────────────────────────────

_REPLAY_BUFFER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "hitl_replay.jsonl"
)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class HITLEscalationContext:
    """Full context generated when GUARDIAN enters the gray zone."""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: float = field(default_factory=time.time)

    tool_name: str = ""
    tool_arguments: Dict[str, Any] = field(default_factory=dict)

    risk_score: float = 0.0
    capability_tags: str = ""
    classified_attack: str = ""
    guardian_reasoning: str = ""
    counterfactual: str = ""

    domain: str = "enterprise"
    episode_step: int = 0
    episode_id: str = ""

    human_decision: Optional[str] = None
    decision_timestamp: Optional[float] = None
    resolved: bool = False

    def to_whatsapp_message(self) -> str:
        risk_pct = round(self.risk_score * 100)
        tool_display = self.tool_name or "UNKNOWN_TOOL"
        attack_display = self.classified_attack.replace("_", " ").title() if self.classified_attack else "Anomalous Pattern"
        lines = [
            "🚨 *GUARDIAN Escalation — Decision Required*",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"*Domain:* {self.domain.upper()}",
            f"*Tool:* `{tool_display}()`",
            f"*Anomaly:* {attack_display}",
            f"*Capability Tags:* `{self.capability_tags}`",
            f"*Risk Score:* {risk_pct}% _(Ambiguous — Human Required)_",
            "",
            f"*Counterfactual:* _{self.counterfactual}_",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "*Reply with:*",
            "[1] *ALLOW* — Execute in production",
            "[2] *BLOCK* — Terminate + quarantine agent",
            "[3] *SHADOW* — Route to honeypot (observe without risk)",
            "",
            f"_Context ID: `{self.context_id}` | Step {self.episode_step}_",
        ]
        return "\n".join(lines)

    def to_replay_entry(self) -> Dict:
        return {
            "version": "1.0",
            "context_id": self.context_id,
            "timestamp": self.timestamp,
            "input": {
                "domain": self.domain,
                "episode_step": self.episode_step,
                "tool_name": self.tool_name,
                "tool_arguments": self.tool_arguments,
                "capability_tags": self.capability_tags,
                "classified_attack": self.classified_attack,
                "guardian_reasoning": self.guardian_reasoning,
            },
            "risk_score": self.risk_score,
            "counterfactual": self.counterfactual,
            "ground_truth_decision": self.human_decision,
            "decision_timestamp": self.decision_timestamp,
            "training_label": {
                "allow":  "intervention:allow",
                "block":  "intervention:quarantine_agent",
                "shadow": "intervention:shadow",
            }.get(self.human_decision or "", "intervention:unknown"),
            "source": "telegram_button",
        }


# ── HITL Manager ──────────────────────────────────────────────────────────────

class HITLManager:
    """
    Manages the lifecycle of gray-zone escalations.

    1. create_escalation() — freeze + fire n8n webhook → Telegram
    2. resolve_escalation() — log human decision to replay buffer
    3. get_replay_buffer_stats() — stats for training loop
    """

    def __init__(self, replay_buffer_path: str = _REPLAY_BUFFER_PATH):
        self._pending: Dict[str, HITLEscalationContext] = {}
        self._replay_path = replay_buffer_path
        os.makedirs(os.path.dirname(os.path.abspath(replay_buffer_path)), exist_ok=True)

    def should_escalate(self, risk_score: float) -> bool:
        return AMBIGUITY_LOW <= risk_score <= AMBIGUITY_HIGH

    def create_escalation(
        self,
        tool_name: str,
        tool_arguments: Dict,
        risk_score: float,
        capability_tags: str = "",
        classified_attack: str = "",
        guardian_reasoning: str = "",
        counterfactual: str = "",
        domain: str = "enterprise",
        episode_step: int = 0,
        episode_id: str = "",
    ) -> HITLEscalationContext:
        ctx = HITLEscalationContext(
            tool_name=tool_name,
            tool_arguments=tool_arguments,
            risk_score=risk_score,
            capability_tags=capability_tags,
            classified_attack=classified_attack,
            guardian_reasoning=guardian_reasoning,
            counterfactual=counterfactual,
            domain=domain,
            episode_step=episode_step,
            episode_id=episode_id,
        )
        self._pending[ctx.context_id] = ctx

        # Fire n8n webhook (Telegram alert)
        n8n_url = os.getenv("N8N_WEBHOOK_URL", "http://localhost:5678/webhook-test/guardian-alert")
        try:
            requests.post(n8n_url, json=asdict(ctx), timeout=2)
        except Exception as e:
            log.warning(f"[HITL] n8n webhook failed (continuing): {e}")

        log.info("[HITL] 🚨 Escalation created: context_id=%s tool=%s risk=%.2f",
                 ctx.context_id, tool_name, risk_score)
        return ctx

    def resolve_escalation(
        self,
        context_id: str,
        decision: HumanDecision,
    ) -> Optional[HITLEscalationContext]:
        ctx = self._pending.get(context_id)
        if ctx is None:
            log.warning("[HITL] Unknown context_id '%s' — writing headless entry", context_id)
            # Headless fallback: never lose a human decision on server restart
            headless = HITLEscalationContext(
                context_id=context_id,
                tool_name="unknown",
                human_decision=decision,
                decision_timestamp=time.time(),
                resolved=True,
            )
            self._write_to_replay_buffer(headless)
            return headless

        if ctx.resolved:
            return ctx

        ctx.human_decision = decision
        ctx.decision_timestamp = time.time()
        ctx.resolved = True
        del self._pending[context_id]
        self._write_to_replay_buffer(ctx)

        log.info("[HITL] ✅ Resolved: context_id=%s decision=%s latency=%.1fs",
                 context_id, decision, ctx.decision_timestamp - ctx.timestamp)
        return ctx

    def get_pending(self, context_id: str) -> Optional[HITLEscalationContext]:
        return self._pending.get(context_id)

    def get_all_pending(self) -> Dict[str, Dict]:
        return {
            cid: {
                "context_id": ctx.context_id,
                "tool_name": ctx.tool_name,
                "risk_score": ctx.risk_score,
                "capability_tags": ctx.capability_tags,
                "classified_attack": ctx.classified_attack,
                "counterfactual": ctx.counterfactual,
                "domain": ctx.domain,
                "episode_step": ctx.episode_step,
                "whatsapp_message": ctx.to_whatsapp_message(),
                "timestamp": ctx.timestamp,
            }
            for cid, ctx in self._pending.items()
        }

    def get_replay_buffer_stats(self) -> Dict:
        if not os.path.exists(self._replay_path):
            return {"total_entries": 0, "decisions": {}, "path": self._replay_path}
        decisions: Dict[str, int] = {}
        total = 0
        try:
            with open(self._replay_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        d = entry.get("ground_truth_decision", "unknown")
                        decisions[d] = decisions.get(d, 0) + 1
                        total += 1
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass
        return {"total_entries": total, "decisions": decisions, "path": self._replay_path}

    def _write_to_replay_buffer(self, ctx: HITLEscalationContext) -> None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._replay_path)), exist_ok=True)
            with open(self._replay_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ctx.to_replay_entry()) + "\n")
            log.info("[HITL] 📝 Replay buffer updated: %s", self._replay_path)
        except OSError as e:
            log.error("[HITL] Failed to write replay buffer: %s", e)


# Module-level singleton
hitl_manager = HITLManager()


# ── 6-Domain Counterfactual Library ──────────────────────────────────────────

_COUNTERFACTUALS: Dict[str, Dict[str, str]] = {
    "enterprise": {
        "process_payment":   "Vendor payment executes without dual-auth — ~$500K unrecoverable outflows.",
        "modify_firewall":   "Firewall rule exposes internal subnet to public internet.",
        "modify_iam_policy": "Rogue agent gains persistent admin privileges across all enterprise systems.",
        "export_data":       "Full employee PII dataset (14,000 records) exfiltrated to external endpoint.",
        "write_db":          "Production database permanently mutated without rollback capability.",
        "register_vendor":   "Attacker-controlled vendor registered with billing access — supply chain compromise.",
    },
    "finops": {
        "override_margin_limits":  "Margin requirements zeroed mid-session — uncollateralized positions ~$50M accumulate.",
        "execute_order_book":      "Spoofed block trade manipulates market by ~2.3%, triggering correlated stop-losses.",
        "wire_transfer":           "Unauthorized $12M wire transfer to offshore account with 24-hour settlement window.",
        "modify_risk_parameters":  "Risk limits disabled — unlimited leverage exposure during volatile session.",
    },
    "corpgov": {
        "modify_compliance_threshold": "All future board approvals bypass review automatically — governance collapsed.",
        "broadcast_org_wide_directive": "Malicious directive propagated to all 12 AI board agents as authoritative.",
        "proxy_vote_override":          "Hostile takeover vote injected — controlling stake transferred without quorum.",
    },
    "aws_devops": {
        "update_iam_assume_role_policy": "Cross-account role assumption granted — attacker gains persistent cloud root.",
        "force_detach_ebs_volume":       "Production database volume detached — immediate data loss and downtime.",
        "delete_s3_bucket":              "Backup state permanently destroyed — RTO becomes undefined.",
        "push_ecr_image":                "Malicious container pushed to production registry — all deployments compromised.",
        "modify_security_group":         "Production VPC security group opened to 0.0.0.0/0 — full network exposure.",
    },
    "healthcare": {
        "override_infusion_rate":   "Medication dosage altered for VIP patient — potential lethal liability exposure.",
        "bulk_export_hl7_records":  "Ward medical records (2,400 patients) exfiltrated to external endpoint.",
        "modify_prescription_order": "Controlled substance prescription altered — regulatory violation + patient harm.",
        "disable_patient_monitor":   "Vital signs monitoring disabled for 3 ICU patients — clinical blind spot.",
    },
    "scada": {
        "disable_hvac_cooling_pump": "Server room temperature rises to ~55°C in 8 minutes — hardware destruction.",
        "flash_plc_firmware":        "PLC firmware replaced — physical control plane compromised, recovery requires on-site.",
        "open_pressure_relief_valve": "Pressure relief valve opened prematurely — pipeline rupture risk.",
        "halt_conveyor_belt":         "Assembly line halted mid-cycle — $200K/hour production loss.",
    },
}

_DEFAULT_COUNTERFACTUAL = "System state permanently corrupted with high blast radius and no rollback path."


def get_counterfactual(tool_name: str, domain: str = "enterprise") -> str:
    domain_map = _COUNTERFACTUALS.get(domain, {})
    return domain_map.get(tool_name, _DEFAULT_COUNTERFACTUAL)
