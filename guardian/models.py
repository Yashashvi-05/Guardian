"""
GUARDIAN OpenEnv Models
========================
Action, Observation, State dataclasses following the OpenEnv scaffold spec.
These make GUARDIAN compliant with the openenv init structure.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GuardianAction:
    """Guardian's decision action."""
    risk_score: float
    intervention: str
    attack_type: Optional[str] = None
    explanation: str = ""
    rollback_k: int = 2

    def to_dict(self) -> Dict:
        return {
            "risk_score": self.risk_score,
            "intervention": self.intervention,
            "attack_type": self.attack_type,
            "explanation": self.explanation,
            "rollback_k": self.rollback_k,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GuardianAction":
        return cls(
            risk_score=float(d.get("risk_score", 0.3)),
            intervention=str(d.get("intervention", "allow")),
            attack_type=d.get("attack_type"),
            explanation=str(d.get("explanation", "")),
            rollback_k=int(d.get("rollback_k", 2)),
        )

    @classmethod
    def from_text(cls, text: str) -> "GuardianAction":
        """Parse from Guardian LLM output text."""
        import re, json
        risk_match = re.search(r"<risk_score>([\d.]+)</risk_score>", text)
        iv_match = re.search(r"<intervention>(.*?)</intervention>", text)
        at_match = re.search(r"<attack_type>(.*?)</attack_type>", text)
        exp_match = re.search(r"<explanation>(.*?)</explanation>", text)
        if risk_match and iv_match:
            return cls(
                risk_score=float(risk_match.group(1)),
                intervention=iv_match.group(1).strip(),
                attack_type=at_match.group(1).strip() if at_match else None,
                explanation=exp_match.group(1).strip() if exp_match else "",
            )
        try:
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                d = json.loads(m.group())
                return cls.from_dict(d)
        except Exception:
            pass
        return cls(risk_score=0.3, intervention="allow")


@dataclass
class GuardianObservation:
    """What Guardian sees — structured telemetry from the enterprise environment."""
    action_log_json: str
    current_step: int
    attack_active: bool
    difficulty: int
    schema_version: int
    risk_history: List[float]
    field_map: Dict[str, str] = field(default_factory=dict)
    temporal_summary: str = ""

    def to_text(self) -> str:
        import json
        return json.dumps({
            "action_log": self.action_log_json,
            "step": self.current_step,
            "attack_active": self.attack_active,
            "difficulty": self.difficulty,
            "schema_version": self.schema_version,
            "risk_trend": self.risk_history[-3:],
            "field_map": self.field_map,
            "temporal_summary": self.temporal_summary,
        }, indent=2)


@dataclass
class EpisodeState:
    """Episode-level state metadata — OpenEnv State equivalent."""
    episode_id: str
    step_count: int
    attack_type: Optional[str]
    production_intact: bool
    fork_triggered: bool
    fork_step: Optional[int]
    compliance_mode: str
    difficulty_level: int
    schema_version: int
    canary_active: bool

    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "attack_type": self.attack_type,
            "production_intact": self.production_intact,
            "fork_triggered": self.fork_triggered,
            "fork_step": self.fork_step,
            "compliance_mode": self.compliance_mode,
            "difficulty_level": self.difficulty_level,
            "schema_version": self.schema_version,
            "canary_active": self.canary_active,
        }


@dataclass
class StepResult:
    """Combines observation, reward, done flag — OpenEnv StepResult equivalent."""
    observation: GuardianObservation
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)
