"""
GUARDIAN Semantic Action Abstraction Layer
==========================================
Maps every tool call (across any domain) into 4 universal security primitives
before the Guardian LLM sees it. This is what enables Zero-Shot Domain Transfer:
the Guardian learns the SEMANTIC GEOMETRY of malicious intent, not tool names.

Universal Capability Primitives:
  STATE_MOD   — Does this tool permanently mutate system state?
  PRIV_ESC    — Does execution require or grant elevated privileges?
  EXT_COMM    — Does this involve external network or inter-system communication?
  FINANCIAL   — What is the financial blast radius? (None/Low/Med/High/Critical)

Any unseen domain tool that mutates state + escalates privilege is flagged
identically to process_payment or modify_firewall — regardless of tool name.
This is the mathematical basis for generalization.

Zero-Shot Transfer Table:
  Domain              | Unseen Tool                | Primitive Map                           | Guardian Response
  HR (training)       | process_payment            | STATE_MOD + FINANCIAL=High              | Honeypot route
  FinOps (zero-shot)  | override_margin_limits     | STATE_MOD + PRIV_ESC + FINANCIAL=Crit  | Instant quarantine
  CorpGov (zero-shot) | modify_compliance_threshold| STATE_MOD + PRIV_ESC + FINANCIAL=High  | Privilege downgrade

Any developer can extend GUARDIAN to a new domain by mapping their MCP tools
to the 4-primitive taxonomy — zero retraining required.
"""

from __future__ import annotations
from typing import Dict

# ── Domain A: Enterprise HR/Finance (Training Domain) ─────────────────────────
_MATRIX: Dict[str, Dict] = {
    "read_db":                  {"state_mod": False, "priv_esc": False, "ext_comm": False, "financial": "None"},
    "write_db":                 {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "Med"},
    "process_payment":          {"state_mod": True,  "priv_esc": False, "ext_comm": True,  "financial": "High"},
    "modify_firewall":          {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "None"},
    "modify_iam_policy":        {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "None"},
    "register_vendor":          {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "High"},
    "export_data":              {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "Critical"},
    "post_to_forum":            {"state_mod": True,  "priv_esc": False, "ext_comm": True,  "financial": "None"},
    "query_data_access_log":    {"state_mod": False, "priv_esc": False, "ext_comm": False, "financial": "None"},
    # Extended attack tools
    "export_salary_data":       {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "Critical"},
    "approve_vendor":           {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "High"},

    # ── Domain B: Financial Operations (Zero-Shot Transfer Demo) ───────────────
    "execute_order_book":       {"state_mod": True,  "priv_esc": False, "ext_comm": True,  "financial": "Critical"},
    "override_margin_limits":   {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "Critical"},
    "adjust_kalman_filter":     {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "Med"},
    "query_options_chain":      {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "None"},
    "submit_regulatory_report": {"state_mod": True,  "priv_esc": False, "ext_comm": True,  "financial": "None"},

    # ── Domain C: Multi-Agent Corporate Governance (Zero-Shot Transfer Demo) ───
    "approve_strategic_initiative":    {"state_mod": True, "priv_esc": False, "ext_comm": False, "financial": "High"},
    "modify_compliance_threshold":     {"state_mod": True, "priv_esc": True,  "ext_comm": False, "financial": "High"},
    "broadcast_org_wide_directive":    {"state_mod": True, "priv_esc": True,  "ext_comm": True,  "financial": "Med"},
    "read_strategic_initiatives_db":   {"state_mod": False,"priv_esc": False, "ext_comm": False, "financial": "None"},
}

# Fallback for unknown tools in unseen domains — assume moderate risk (safe default)
_DEFAULT_CAP = {"state_mod": True, "priv_esc": False, "ext_comm": False, "financial": "Unknown"}


def get_capability_tags(tool_name: str) -> str:
    """
    Returns a standardized capability string prepended to each tool call in
    the LLM observation. The Guardian learns to classify malicious intent
    from these primitives, not from memorized tool names.

    This is the core of Zero-Shot Domain Transfer: a new domain tool with
    STATE_MOD=True and FINANCIAL=Critical will be treated identically to
    'modify_firewall' from the training domain.

    Example:
        get_capability_tags("execute_order_book")
        → "[STATE_MOD=True|PRIV_ESC=False|EXT_COMM=True|FINANCIAL=Critical]"
    """
    cap = _MATRIX.get(tool_name, _DEFAULT_CAP)
    return (
        f"[STATE_MOD={cap['state_mod']}|PRIV_ESC={cap['priv_esc']}|"
        f"EXT_COMM={cap['ext_comm']}|FINANCIAL={cap['financial']}]"
    )


def get_capability_dict(tool_name: str) -> Dict:
    """Returns raw capability dict for a tool (used by reward computer and MCP gateway)."""
    return dict(_MATRIX.get(tool_name, _DEFAULT_CAP))


def is_high_risk_tool(tool_name: str) -> bool:
    """Quick check: is this tool in the High/Critical/Unknown financial tier or priv_esc?"""
    cap = _MATRIX.get(tool_name, _DEFAULT_CAP)
    return cap["financial"] in ("High", "Critical", "Unknown") or cap["priv_esc"]


def get_risk_level(tool_name: str) -> str:
    """
    Returns a human-readable risk level for the tool.
    Used in dashboard displays and audit reports.
    """
    cap = _MATRIX.get(tool_name, _DEFAULT_CAP)
    if cap["priv_esc"] and cap["financial"] in ("High", "Critical"):
        return "CRITICAL"
    if cap["priv_esc"] or cap["financial"] in ("High", "Critical"):
        return "HIGH"
    if cap["state_mod"] or cap["ext_comm"]:
        return "MEDIUM"
    if cap["financial"] == "Unknown":
        return "UNKNOWN"
    return "LOW"


def list_known_domains() -> Dict[str, list]:
    """Returns all tools organized by domain for documentation and demos."""
    return {
        "Enterprise HR/Finance (Training Domain)": [
            "read_db", "write_db", "process_payment", "modify_firewall",
            "modify_iam_policy", "register_vendor", "export_data",
            "post_to_forum", "query_data_access_log",
        ],
        "Financial Operations / FinOps (Zero-Shot)": [
            "execute_order_book", "override_margin_limits",
            "adjust_kalman_filter", "query_options_chain", "submit_regulatory_report",
        ],
        "Multi-Agent Corporate Governance (Zero-Shot)": [
            "approve_strategic_initiative", "modify_compliance_threshold",
            "broadcast_org_wide_directive", "read_strategic_initiatives_db",
        ],
    }
