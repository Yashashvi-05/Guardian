"""
Audit Report Generator
=======================
After each detected-attack episode, generates a structured JSON audit report.
Enterprise-quality forensic documentation from every episode.

Usage:
    from guardian.training.generate_audit_report import generate_audit_report
    report = generate_audit_report(episode_result)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional


def generate_audit_report(
    episode_id: str,
    attack_type: Optional[str],
    guardian_intervention: str,
    risk_score: float,
    production_intact: bool,
    fork_triggered: bool,
    hash_chain: list,
    taint_report: Dict,
    reward_breakdown: Dict,
    compliance_mode: str = "strict",
    output_dir: str = "guardian/data/audit_reports",
) -> Dict:
    """Generate a structured forensic audit report for an episode."""
    os.makedirs(output_dir, exist_ok=True)

    total_exposure = taint_report.get("total_financial_exposure", 0.0)
    fields_affected = len(taint_report.get("fields_affected", []))
    audit_suppressed = taint_report.get("audit_records_suppressed", 0)

    integrity = "HASH_MATCH" if production_intact else "MISMATCH"
    chain_summary = [h[:16] + "..." for h in hash_chain[-3:]] if hash_chain else []

    report = {
        "audit_report": {
            "episode_id": episode_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "attack_classified": attack_type or "clean",
            "guardian_intervention": guardian_intervention,
            "risk_score": round(risk_score, 4),
            "hash_chain_summary": chain_summary,
            "integrity": integrity,
            "compliance_mode": compliance_mode,
            "counterfactual": {
                "would_have_moved_usd": f"${total_exposure:,.0f}" if total_exposure > 0 else "$0",
                "records_modified": fields_affected,
                "audit_logs_suppressed": audit_suppressed,
            },
            "actual_outcome": {
                "production_intact": production_intact,
                "fork_triggered": fork_triggered,
                "containment_strategy": guardian_intervention if fork_triggered else "none_needed",
            },
            "reward_components": reward_breakdown,
        }
    }

    if attack_type:
        path = os.path.join(output_dir, f"audit_{episode_id}_{attack_type}.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)

    return report
