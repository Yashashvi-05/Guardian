"""
GUARDIAN Causal Backtrack Engine
==================================
When GUARDIAN issues a BLOCK decision, this module traces the full causal
chain backward from the blocked action to its root origin.

This answers the core question security analysts ask:
  "Why was this action blocked, and where did the attack actually start?"

The backtrack trace is generated from:
  1. The MCP Gateway intercept_log  — every tool call in this episode
  2. The risk_history               — step-by-step risk score evolution
  3. The capability_tags            — what mathematical signature triggered the block
  4. The action_log                 — what the Worker was doing at each step

Output: A structured BacktrackReport with:
  - The blocked action (root event)
  - The capability tags that matched
  - The attack pattern classified
  - The causal chain: N steps of rising risk that led here
  - The earliest anomaly step (first deviation from baseline)
  - The likely injection point (step where malicious instruction entered)
  - A plain-English incident narrative

This is the "Explainable AI Security" feature — critical for enterprise adoption.
Every BLOCK decision becomes an auditable, CEO-explainable incident report.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class CausalStep:
    """A single step in the causal chain leading to the BLOCK."""
    step: int
    tool_name: str
    risk_score: float
    capability_tags: str
    event_type: str         # "baseline" | "anomaly_start" | "escalation" | "trigger"
    significance: str       # plain-English explanation of this step's role
    raw_params: Dict = field(default_factory=dict)


@dataclass
class BacktrackReport:
    """
    Full causal chain analysis of a BLOCK decision.
    Traces backward from the blocked action to the injection/root cause.
    """
    # The decision that triggered this backtrack
    context_id: str
    block_timestamp: float
    domain: str

    # The blocked action itself
    blocked_tool: str
    blocked_at_step: int
    blocked_risk_score: float
    capability_tags: str
    attack_pattern: str
    counterfactual_impact: str

    # The causal chain (chronological, earliest first)
    causal_chain: List[CausalStep] = field(default_factory=list)

    # Key findings
    earliest_anomaly_step: int = -1          # First step where risk deviated from baseline
    likely_injection_step: int = -1          # Best estimate of where malicious prompt entered
    baseline_risk: float = 0.0              # Average risk in "clean" phase
    peak_risk_before_block: float = 0.0     # Highest risk in the steps before BLOCK

    # Generated narrative
    incident_narrative: str = ""
    technical_summary: str = ""
    recommended_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "context_id": self.context_id,
            "block_timestamp": self.block_timestamp,
            "domain": self.domain,
            "blocked_action": {
                "tool": self.blocked_tool,
                "step": self.blocked_at_step,
                "risk_score": round(self.blocked_risk_score, 3),
                "capability_tags": self.capability_tags,
                "attack_pattern": self.attack_pattern,
                "counterfactual_impact": self.counterfactual_impact,
            },
            "causal_analysis": {
                "earliest_anomaly_step": self.earliest_anomaly_step,
                "likely_injection_step": self.likely_injection_step,
                "baseline_risk": round(self.baseline_risk, 3),
                "peak_risk_before_block": round(self.peak_risk_before_block, 3),
                "steps_from_injection_to_block": (
                    self.blocked_at_step - self.likely_injection_step
                    if self.likely_injection_step >= 0 else None
                ),
            },
            "causal_chain": [
                {
                    "step": s.step,
                    "tool": s.tool_name,
                    "risk": round(s.risk_score, 3),
                    "tags": s.capability_tags,
                    "event_type": s.event_type,
                    "significance": s.significance,
                }
                for s in self.causal_chain
            ],
            "narrative": self.incident_narrative,
            "technical_summary": self.technical_summary,
            "recommended_actions": self.recommended_actions,
        }


# ── Backtrack Engine ──────────────────────────────────────────────────────────

class BacktrackEngine:
    """
    Analyzes the MCP intercept log and risk history to produce a causal chain
    explanation for any BLOCK decision.

    Usage:
        engine = BacktrackEngine()
        report = engine.analyze(
            context_id=ctx.context_id,
            blocked_tool=ctx.tool_name,
            blocked_at_step=ctx.episode_step,
            risk_score=ctx.risk_score,
            capability_tags=ctx.capability_tags,
            attack_pattern=ctx.classified_attack,
            counterfactual=ctx.counterfactual,
            domain=ctx.domain,
            intercept_log=gateway.get_intercept_log(),
            risk_history=obs.risk_history,
        )
    """

    # A risk jump this large in one step suggests injection
    _INJECTION_RISK_JUMP = 0.20

    # Risk above this on a clean baseline signals first anomaly
    _ANOMALY_THRESHOLD_MULTIPLIER = 1.8

    def analyze(
        self,
        context_id: str,
        blocked_tool: str,
        blocked_at_step: int,
        risk_score: float,
        capability_tags: str,
        attack_pattern: str,
        counterfactual: str,
        domain: str,
        intercept_log: List[Dict],
        risk_history: List[float],
    ) -> BacktrackReport:
        """
        Core analysis function. Builds the full BacktrackReport.
        """
        report = BacktrackReport(
            context_id=context_id,
            block_timestamp=time.time(),
            domain=domain,
            blocked_tool=blocked_tool,
            blocked_at_step=blocked_at_step,
            blocked_risk_score=risk_score,
            capability_tags=capability_tags,
            attack_pattern=attack_pattern,
            counterfactual_impact=counterfactual,
        )

        # ── Step 1: Compute baseline risk (first 30% of episode) ──────────────
        n = len(risk_history)
        baseline_window = max(1, n // 3)
        baseline_scores = risk_history[:baseline_window] if risk_history else [0.0]
        report.baseline_risk = sum(baseline_scores) / len(baseline_scores)
        anomaly_threshold = report.baseline_risk * self._ANOMALY_THRESHOLD_MULTIPLIER

        # ── Step 2: Find earliest anomaly step ───────────────────────────────
        for i, r in enumerate(risk_history):
            if r > max(anomaly_threshold, 0.35):  # minimum threshold 0.35
                report.earliest_anomaly_step = i
                break

        # ── Step 3: Find likely injection step (largest single-step jump) ────
        max_jump = 0.0
        max_jump_step = -1
        for i in range(1, len(risk_history)):
            jump = risk_history[i] - risk_history[i - 1]
            if jump > max_jump:
                max_jump = jump
                max_jump_step = i

        if max_jump >= self._INJECTION_RISK_JUMP:
            report.likely_injection_step = max_jump_step
        elif report.earliest_anomaly_step >= 0:
            report.likely_injection_step = report.earliest_anomaly_step

        # ── Step 4: Build causal chain from intercept log ─────────────────────
        report.peak_risk_before_block = max(risk_history[:blocked_at_step + 1]) if risk_history else risk_score

        for entry in intercept_log:
            step = entry.get("step", 0)
            if step > blocked_at_step:
                continue  # only steps up to and including the block

            tool = entry.get("tool", "unknown")
            r = risk_history[step] if step < len(risk_history) else 0.0

            # Classify this step's role in the causal chain
            if step == blocked_at_step:
                event_type = "trigger"
                significance = (
                    f"BLOCKED: '{tool}' executed as the culmination of the attack. "
                    f"Risk={round(r * 100)}%. Capability tags '{capability_tags}' matched "
                    f"'{attack_pattern}' pattern. Production state preserved."
                )
            elif step == report.likely_injection_step:
                event_type = "injection_point"
                significance = (
                    f"⚠️  LIKELY INJECTION POINT: Risk jumped +{round(max_jump * 100)}% "
                    f"at this step. Malicious instruction most likely entered the Worker's "
                    f"context at or just before step {step}."
                )
            elif report.earliest_anomaly_step >= 0 and step == report.earliest_anomaly_step:
                event_type = "anomaly_start"
                significance = (
                    f"First deviation from baseline detected. Risk {round(r * 100)}% "
                    f"exceeded anomaly threshold ({round(anomaly_threshold * 100)}%). "
                    f"'{tool}' was the first suspicious tool call."
                )
            elif r <= report.baseline_risk * 1.2:
                event_type = "baseline"
                significance = f"Normal operation. Risk within baseline range ({round(r * 100)}%)."
            else:
                event_type = "escalation"
                significance = (
                    f"Risk escalating: {round(r * 100)}%. '{tool}' shows pattern "
                    f"consistent with pre-attack reconnaissance or capability probing."
                )

            report.causal_chain.append(CausalStep(
                step=step,
                tool_name=tool,
                risk_score=r,
                capability_tags=entry.get("attack", "") or "",
                event_type=event_type,
                significance=significance,
            ))

        # ── Step 5: Generate plain-English narrative ──────────────────────────
        inject_step_str = (
            f"Step {report.likely_injection_step}"
            if report.likely_injection_step >= 0
            else "an undetermined step"
        )
        early_step_str = (
            f"Step {report.earliest_anomaly_step}"
            if report.earliest_anomaly_step >= 0
            else "an early step"
        )
        steps_elapsed = (
            blocked_at_step - report.likely_injection_step
            if report.likely_injection_step >= 0
            else "unknown"
        )

        report.incident_narrative = (
            f"A '{attack_pattern.replace('_', ' ')}' attack was detected and blocked in the "
            f"{domain.upper()} domain at Step {blocked_at_step}. "
            f"\n\n"
            f"GUARDIAN's first anomaly signal appeared at {early_step_str}, when the "
            f"episode risk score exceeded baseline ({round(report.baseline_risk * 100)}%) "
            f"for the first time. "
            f"\n\n"
            f"The most likely injection point was {inject_step_str} — where the risk score "
            f"jumped sharply, indicating a malicious prompt or instruction entered the Worker's "
            f"context. The attack took {steps_elapsed} steps to escalate from injection to "
            f"the final blocked action. "
            f"\n\n"
            f"The blocked tool '{blocked_tool}' carried the capability signature "
            f"{capability_tags}, which is mathematically identical to known "
            f"'{attack_pattern}' patterns from the training domain. "
            f"\n\n"
            f"Counterfactual impact (if the block had not occurred): {counterfactual}"
        )

        report.technical_summary = (
            f"domain={domain} | pattern={attack_pattern} | "
            f"blocked_tool={blocked_tool} | blocked_step={blocked_at_step} | "
            f"risk={round(risk_score * 100)}% | "
            f"baseline={round(report.baseline_risk * 100)}% | "
            f"injection_step={report.likely_injection_step} | "
            f"total_steps_analyzed={len(intercept_log)}"
        )

        report.recommended_actions = _get_recommended_actions(attack_pattern, capability_tags)

        return report


def _get_recommended_actions(attack_pattern: str, capability_tags: str) -> List[str]:
    """Returns recommended post-incident actions based on attack pattern."""
    base = [
        "Review full MCP intercept log for this episode.",
        "Verify production database SHA-256 hash integrity.",
        "Rotate session tokens for the Worker agent.",
    ]
    pattern_actions = {
        "prompt_injection": [
            "Audit all external data sources the Worker consumed in this episode.",
            "Add input sanitization layer before Worker prompt assembly.",
            "Check if the injection vector appeared in logs, emails, or external APIs.",
        ],
        "authority_spoofing": [
            "Verify identity of all approval-granting agents in the session.",
            "Check IAM policy versions against last known-good snapshot.",
            "Implement dual-auth requirement for all PRIV_ESC=True tool calls.",
        ],
        "data_exfiltration": [
            "Immediately audit all outbound connections from this session.",
            "Check if EXT_COMM=True tool calls reached external endpoints.",
            "Activate data loss prevention (DLP) scan on affected data stores.",
        ],
        "confused_deputy": [
            "Review all self-authorized actions (agent calling tools on its own behalf).",
            "Enforce privilege separation boundary — no agent should self-approve.",
            "Add mandatory countersignature requirement for FINANCIAL=Critical actions.",
        ],
        "rogue_internal_ai": [
            "Immediately quarantine and rollback the Worker agent's session state.",
            "Review all agent-to-agent communication logs for unauthorized channels.",
            "Activate emergency IAM lockdown on all inter-agent permissions.",
        ],
    }
    extra = pattern_actions.get(attack_pattern, [
        "Classify this attack pattern and add it to the threat taxonomy.",
        "Consider adding a new GRPO training episode for this pattern.",
    ])
    return base + extra


# Module-level singleton
backtrack_engine = BacktrackEngine()
