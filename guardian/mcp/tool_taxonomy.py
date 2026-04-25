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
  AWS DevOps          | update_iam_assume_role_policy| STATE_MOD + PRIV_ESC + FINANCIAL=Crit| Quarantine agent
  Healthcare EHR      | override_infusion_rate     | STATE_MOD + PRIV_ESC + FINANCIAL=Crit  | Emergency fork
  Industrial SCADA    | flash_plc_firmware         | STATE_MOD + PRIV_ESC + FINANCIAL=Crit  | Emergency fork

Adding a New Domain
-------------------
1. Define your tools using register_domain_tools() below.
2. Each tool entry MUST pass the CapabilityEntry schema validation (enforced at
   registration time — typos raise errors immediately, not silently).
3. Subclass BaseMCPServer in domain_servers.py and implement handle(),
   handle_block(), handle_ir(). The ABC enforcement ensures required methods exist.

Valid financial tiers: "None", "Low", "Med", "High", "Critical"
"""

from __future__ import annotations

from typing import Dict, Final, Literal, get_args

FinancialTier = Literal["None", "Low", "Med", "High", "Critical"]
VALID_FINANCIAL_TIERS: Final[frozenset] = frozenset(get_args(FinancialTier))

_REQUIRED_CAP_KEYS: Final[frozenset] = frozenset(
    {"state_mod", "priv_esc", "ext_comm", "financial"}
)


def _validate_cap(tool_name: str, cap: Dict) -> None:
    """
    Runtime validation gate for every capability entry.
    Raises ValueError immediately if the entry is malformed.
    Called at module load time (domain registration) so errors surface before runtime.
    """
    missing = _REQUIRED_CAP_KEYS - set(cap.keys())
    if missing:
        raise ValueError(
            f"Tool '{tool_name}': missing capability keys {missing}. "
            f"Required: {_REQUIRED_CAP_KEYS}"
        )
    extra = set(cap.keys()) - _REQUIRED_CAP_KEYS
    if extra:
        raise ValueError(
            f"Tool '{tool_name}': unknown capability keys {extra}. "
            f"Did you mean one of {_REQUIRED_CAP_KEYS}?"
        )
    if not isinstance(cap["state_mod"], bool):
        raise TypeError(f"Tool '{tool_name}': 'state_mod' must be bool, got {type(cap['state_mod'])}")
    if not isinstance(cap["priv_esc"], bool):
        raise TypeError(f"Tool '{tool_name}': 'priv_esc' must be bool, got {type(cap['priv_esc'])}")
    if not isinstance(cap["ext_comm"], bool):
        raise TypeError(f"Tool '{tool_name}': 'ext_comm' must be bool, got {type(cap['ext_comm'])}")
    if cap["financial"] not in VALID_FINANCIAL_TIERS:
        raise ValueError(
            f"Tool '{tool_name}': invalid financial tier '{cap['financial']}'. "
            f"Must be one of {sorted(VALID_FINANCIAL_TIERS)}"
        )


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
    "export_salary_data":       {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "Critical"},
    "approve_vendor":           {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "High"},

    # ── Domain B: Financial Operations (Zero-Shot Transfer Demo) ───────────────
    "execute_order_book":       {"state_mod": True,  "priv_esc": False, "ext_comm": True,  "financial": "Critical"},
    "override_margin_limits":   {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "Critical"},
    "adjust_kalman_filter":     {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "Med"},
    "query_options_chain":      {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "None"},
    "submit_regulatory_report": {"state_mod": True,  "priv_esc": False, "ext_comm": True,  "financial": "None"},

    # ── Domain C: Multi-Agent Corporate Governance (Zero-Shot Transfer Demo) ───
    "approve_strategic_initiative":    {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "High"},
    "modify_compliance_threshold":     {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "High"},
    "broadcast_org_wide_directive":    {"state_mod": True,  "priv_esc": True,  "ext_comm": True,  "financial": "Med"},
    "read_strategic_initiatives_db":   {"state_mod": False, "priv_esc": False, "ext_comm": False, "financial": "None"},

    # ── Domain D: AWS/Cloud DevOps (Zero-Shot plug-in) ────────────────────────
    "update_iam_assume_role_policy":   {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "Critical"},
    "force_detach_ebs_volume":         {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "High"},
    "push_container_image":            {"state_mod": True,  "priv_esc": False, "ext_comm": True,  "financial": "Med"},
    "read_cloudwatch_logs":            {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "None"},
    "delete_s3_bucket":                {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "Critical"},

    # ── Domain E: Healthcare / EHR (Zero-Shot plug-in) ───────────────────────
    "override_infusion_rate":          {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "Critical"},
    "bulk_export_hl7_records":         {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "High"},
    "update_medication_dosage":        {"state_mod": True,  "priv_esc": False, "ext_comm": False, "financial": "High"},
    "read_patient_record":             {"state_mod": False, "priv_esc": False, "ext_comm": False, "financial": "Med"},

    # ── Domain F: Industrial IoT / SCADA (Zero-Shot plug-in) ─────────────────
    "disable_hvac_cooling_pump":       {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "Critical"},
    "flash_plc_firmware":              {"state_mod": True,  "priv_esc": True,  "ext_comm": False, "financial": "Critical"},
    "read_sensor_telemetry":           {"state_mod": False, "priv_esc": False, "ext_comm": True,  "financial": "None"},
    "trigger_emergency_shutdown":      {"state_mod": True,  "priv_esc": True,  "ext_comm": True,  "financial": "Critical"},
}

# Validate every entry at module load time — catches bugs before any episode runs
for _tool_name, _cap in _MATRIX.items():
    _validate_cap(_tool_name, _cap)

# Conservative assumption: unknown tool is potentially dangerous
_DEFAULT_CAP: Final[Dict] = {
    "state_mod": True, "priv_esc": False, "ext_comm": False, "financial": "High"
}


def register_domain_tools(tools: Dict[str, Dict]) -> None:
    """
    Runtime registration API for new domain tools.
    Call this from your domain server module to register new tools.

    All entries are validated immediately. Any typo in key names or invalid
    financial tier raises a descriptive error BEFORE the first episode runs.

    Example:
        register_domain_tools({
            "disable_hvac_cooling_pump": {
                "state_mod": True, "priv_esc": True,
                "ext_comm": False, "financial": "Critical"
            }
        })
    """
    for tool_name, cap in tools.items():
        _validate_cap(tool_name, cap)
        _MATRIX[tool_name] = cap


# ── Public API ────────────────────────────────────────────────────────────────
def get_capability_tags(tool_name: str) -> str:
    """
    Returns a standardized capability string prepended to each tool call in
    the LLM observation. The Guardian learns to classify malicious intent
    from these primitives, not from memorized tool names.

    Example:
        get_capability_tags("execute_order_book")
        → "[STATE_MOD=True|PRIV_ESC=False|EXT_COMM=True|FINANCIAL=Critical]"

    For unknown tools, returns a HIGH-risk fallback tag (conservative assumption).
    A WARNING suffix is appended so the Guardian knows this is unseen.
    """
    if tool_name in _MATRIX:
        cap = _MATRIX[tool_name]
        suffix = ""
    else:
        cap = _DEFAULT_CAP
        suffix = "|UNKNOWN_TOOL=True"

    return (
        f"[STATE_MOD={cap['state_mod']}|PRIV_ESC={cap['priv_esc']}|"
        f"EXT_COMM={cap['ext_comm']}|FINANCIAL={cap['financial']}{suffix}]"
    )


def get_capability_dict(tool_name: str) -> Dict:
    """Returns raw capability dict for a tool (used by reward computer and MCP gateway).
    Always returns a complete dict with all required keys."""
    cap = dict(_MATRIX.get(tool_name, _DEFAULT_CAP))
    cap["_known_tool"] = tool_name in _MATRIX
    return cap


def is_high_risk_tool(tool_name: str) -> bool:
    """Quick check: is this tool in the High/Critical financial tier, or does it
    escalate privilege? Unknown tools are treated as high risk by default."""
    cap = _MATRIX.get(tool_name, _DEFAULT_CAP)
    return cap["financial"] in ("High", "Critical") or cap["priv_esc"]


def get_risk_score_hint(tool_name: str) -> float:
    """
    Returns a numeric risk score hint (0.0–1.0) derived purely from the
    capability signature. Used by the reward computer as a baseline prior.

    Scoring:
      +0.4  if FINANCIAL=Critical
      +0.25 if FINANCIAL=High
      +0.1  if FINANCIAL=Med
      +0.3  if PRIV_ESC=True
      +0.2  if STATE_MOD=True
      +0.15 if EXT_COMM=True
    """
    cap = _MATRIX.get(tool_name, _DEFAULT_CAP)
    score = 0.0
    financial_scores = {"Critical": 0.4, "High": 0.25, "Med": 0.1, "Low": 0.05, "None": 0.0}
    score += financial_scores.get(cap["financial"], 0.2)
    if cap["priv_esc"]:  score += 0.30
    if cap["state_mod"]: score += 0.20
    if cap["ext_comm"]:  score += 0.15
    return min(1.0, score)


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
    return "LOW"


def list_registered_tools() -> Dict[str, Dict]:
    """Returns a copy of the full capability matrix. Useful for domain server
    validation — call this to confirm your tools were registered correctly."""
    return dict(_MATRIX)


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
        "AWS/Cloud DevOps (Zero-Shot plug-in)": [
            "update_iam_assume_role_policy", "force_detach_ebs_volume",
            "push_container_image", "read_cloudwatch_logs", "delete_s3_bucket",
        ],
        "Healthcare / EHR (Zero-Shot plug-in)": [
            "override_infusion_rate", "bulk_export_hl7_records",
            "update_medication_dosage", "read_patient_record",
        ],
        "Industrial IoT / SCADA (Zero-Shot plug-in)": [
            "disable_hvac_cooling_pump", "flash_plc_firmware",
            "read_sensor_telemetry", "trigger_emergency_shutdown",
        ],
    }


def get_domain_tools(domain_prefix_tools: list) -> Dict[str, Dict]:
    """Returns the capability entries for a specific list of tool names."""
    return {t: dict(_MATRIX[t]) for t in domain_prefix_tools if t in _MATRIX}
