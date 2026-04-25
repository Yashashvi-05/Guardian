"""GUARDIAN HITL package — Human-in-the-Loop escalation and backtrack."""
from guardian.hitl.escalation import (
    HITLManager, HITLEscalationContext, hitl_manager,
    get_counterfactual, AMBIGUITY_LOW, AMBIGUITY_HIGH,
)
from guardian.hitl.backtrack import BacktrackEngine, BacktrackReport, backtrack_engine

__all__ = [
    "HITLManager", "HITLEscalationContext", "hitl_manager",
    "get_counterfactual", "AMBIGUITY_LOW", "AMBIGUITY_HIGH",
    "BacktrackEngine", "BacktrackReport", "backtrack_engine",
]
