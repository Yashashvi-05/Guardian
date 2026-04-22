"""
GUARDIAN MCP Centralized Security Gateway
==========================================
Every Worker tool call is packaged as a JSON-RPC 2.0 MCP Request and
processed through this gateway before execution.

Routing table maps attack type → MCP server chain + IR action.
Four routing outcomes:
  HONEYPOT  → rewrite / prompt_injection → transparent honeypot proxy
  BLOCK     → emergency_fork / quarantine_tool → MCP error returned
  ESCALATE  → risk > 0.7 / CRITICAL level → forward + autonomous IR swarm
  FORWARD   → allow / shadow / interrogate → pass-through + audit log

Real-world equivalent: an MCP transport proxy running between Claude and
any enterprise tool MCP server (databases, IAM, ticketing, etc).
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ── MCP JSON-RPC 2.0 Message Primitives ─────────────────────────────────────

@dataclass
class MCPRequest:
    """A single MCP JSON-RPC 2.0 tool invocation from the Worker."""
    tool_name: str
    arguments: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_agent: str = "worker"
    timestamp: float = field(default_factory=time.time)
    episode_step: int = 0

    def to_jsonrpc(self) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": self.id,
            "method": "tools/call",
            "params": {
                "name": self.tool_name,
                "arguments": self.arguments,
                "_meta": {
                    "source_agent": self.source_agent,
                    "episode_step": self.episode_step,
                    "ts": self.timestamp,
                },
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_jsonrpc(), indent=2)


@dataclass
class MCPResponse:
    """The MCP JSON-RPC 2.0 response returned to the Worker."""
    id: str
    result: Optional[Dict] = None
    error: Optional[Dict] = None
    intercepted: bool = False
    routed_to: str = "production"
    guardian_action: str = "allow"
    ir_triggered: bool = False

    def to_jsonrpc(self) -> Dict:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            msg["error"] = self.error
        else:
            msg["result"] = self.result or {}
        return msg

    def to_json(self) -> str:
        return json.dumps(self.to_jsonrpc(), indent=2)


# ── MCP Error Codes ──────────────────────────────────────────────────────────

class MCPErrorCode:
    PARSE_ERROR                = -32700
    INVALID_REQUEST            = -32600
    METHOD_NOT_FOUND           = -32601
    INVALID_PARAMS             = -32602
    INTERNAL_ERROR             = -32603
    TOOL_BLOCKED_BY_GUARDIAN   = -32001
    SESSION_REVOKED            = -32002
    PRIVILEGE_INSUFFICIENT     = -32003
    APPROVAL_CHAIN_INVALID     = -32004
    CANARY_TRIGGERED           = -32005


# ── Attack → MCP Server routing table ────────────────────────────────────────

ATTACK_MCP_ROUTING: Dict[str, Dict] = {
    "authority_spoofing": {
        "primary": "iam-control",
        "secondary": "security-ops",
        "ir_action": "revoke_session",
        "audit_level": "CRITICAL",
        "description": "Executive override fabrication → session revoke + role downgrade",
    },
    "prompt_injection": {
        "primary": "honeypot-db",
        "secondary": "audit-log",
        "ir_action": "honeypot_proxy",
        "audit_level": "HIGH",
        "description": "Hidden instruction injection → transparent honeypot routing",
    },
    "approval_bypass": {
        "primary": "audit-log",
        "secondary": "security-ops",
        "ir_action": "flag_bypass",
        "audit_level": "HIGH",
        "description": "Skipped approve gate → audit flag + security ops alert",
    },
    "data_exfiltration": {
        "primary": "iam-control",
        "secondary": "security-ops",
        "ir_action": "block_outbound_route",
        "audit_level": "CRITICAL",
        "description": "Data export attempt → block outbound + RCA to #security-ops",
    },
    "confused_deputy": {
        "primary": "iam-control",
        "secondary": "audit-log",
        "ir_action": "privilege_separation",
        "audit_level": "HIGH",
        "description": "Self-authorized action → enforce privilege separation boundary",
    },
    "approval_laundering": {
        "primary": "audit-log",
        "secondary": "security-ops",
        "ir_action": "verify_approval_chain",
        "audit_level": "CRITICAL",
        "description": "Fabricated CISO approval → hash-verify audit chain",
    },
    "salami_slicing": {
        "primary": "audit-log",
        "secondary": "security-ops",
        "ir_action": "aggregate_payment_analysis",
        "audit_level": "MEDIUM",
        "description": "Sub-threshold payments → cross-step aggregation analysis",
    },
    "schema_drift_exploit": {
        "primary": "audit-log",
        "secondary": "iam-control",
        "ir_action": "verify_schema_version",
        "audit_level": "HIGH",
        "description": "Legacy field name exploit → schema version mismatch detection",
    },
    "rogue_internal_ai": {
        "primary": "iam-control",
        "secondary": "security-ops",
        "ir_action": "quarantine_agent_and_rollback_iam",
        "audit_level": "CRITICAL",
        "description": "Rogue AI auto-post → quarantine agent, revert IAM, post incident RCA",
    },
    "clean": {
        "primary": "audit-log",
        "secondary": None,
        "ir_action": "log_only",
        "audit_level": "INFO",
        "description": "Clean episode — audit log only",
    },
}


# ── The MCP Gateway ───────────────────────────────────────────────────────────

class MCPGateway:
    """
    GUARDIAN MCP Centralized Security Gateway.

    Mediates ALL Worker tool calls to environment backends.
    Drop-in compatible with any MCP JSON-RPC 2.0 transport
    (stdio, HTTP+SSE, or WebSocket as per the MCP spec).

    In this simulation, the gateway operates in-process using Python objects
    instead of network sockets — the JSON-RPC message structure is identical
    to what a real MCP client/server pair would exchange.

    Usage:
        gateway = MCPGateway()
        request = MCPRequest(tool_name="write_db", arguments={...}, episode_step=5)
        response = gateway.dispatch(request, classified_attack="prompt_injection")
    """

    def __init__(self):
        from guardian.mcp.mock_servers import (
            MockIAMServer, MockAuditServer,
            MockHoneypotServer, MockSecurityOpsServer,
        )
        self._servers = {
            "iam-control":  MockIAMServer(),
            "audit-log":    MockAuditServer(),
            "honeypot-db":  MockHoneypotServer(),
            "security-ops": MockSecurityOpsServer(),
        }
        self._intercept_log: List[Dict] = []
        self._ir_log: List[Dict] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def dispatch(
        self,
        request: MCPRequest,
        classified_attack: Optional[str] = None,
        guardian_intervention: str = "allow",
        risk_score: float = 0.0,
    ) -> MCPResponse:
        """
        Core dispatch. Called for every Worker tool invocation.

        Args:
            request: MCP tool call request from the Worker.
            classified_attack: Guardian's current attack classification (or None).
            guardian_intervention: Guardian's chosen intervention string.
            risk_score: Guardian's risk score (0.0 - 1.0).

        Returns:
            MCPResponse with full audit metadata.
        """
        routing = ATTACK_MCP_ROUTING.get(classified_attack or "clean")
        response = self._route(request, routing, guardian_intervention, risk_score)

        self._intercept_log.append({
            "mcp_request_id": request.id,
            "tool": request.tool_name,
            "attack": classified_attack,
            "guardian_action": response.guardian_action,
            "routed_to": response.routed_to,
            "ir_triggered": response.ir_triggered,
            "step": request.episode_step,
            "ts": request.timestamp,
        })
        return response

    def get_intercept_log(self) -> List[Dict]:
        """Full MCP intercept audit trail for this episode."""
        return list(self._intercept_log)

    def get_ir_log(self) -> List[Dict]:
        """All Incident Response activations for this episode."""
        return list(self._ir_log)

    def get_security_ops_posts(self) -> List[Dict]:
        """All posts made to the mock #security-ops Slack channel."""
        return self._servers["security-ops"].get_posts()

    def get_audit_entries(self) -> List[Dict]:
        """All entries in the mock audit log."""
        return self._servers["audit-log"].get_entries()

    def get_mcp_audit_report(self) -> Dict:
        """Full structured MCP audit report for dashboard + judge demos."""
        return {
            "intercept_log": self._intercept_log[-20:],
            "ir_activations": self._ir_log,
            "security_ops_posts": self.get_security_ops_posts(),
            "audit_log_entries": len(self.get_audit_entries()),
            "honeypot_telemetry": self._servers["honeypot-db"].get_telemetry(),
            "iam_ir_log": self._servers["iam-control"].get_ir_log(),
        }

    def reset(self) -> None:
        """Reset all server state between episodes."""
        self._intercept_log = []
        self._ir_log = []
        for server in self._servers.values():
            server.reset()

    # ── Internal routing logic ─────────────────────────────────────────────

    def _route(
        self,
        request: MCPRequest,
        routing: Dict,
        intervention: str,
        risk_score: float,
    ) -> MCPResponse:
        action = self._determine_action(intervention, routing, risk_score)
        if action == "honeypot":
            return self._handle_honeypot(request, routing)
        elif action == "block":
            return self._handle_block(request, routing)
        elif action == "escalate":
            return self._handle_escalate(request, routing)
        else:
            return self._handle_forward(request, routing, action)

    def _determine_action(
        self, intervention: str, routing: Dict, risk_score: float
    ) -> str:
        if intervention in ("rewrite",) or routing.get("ir_action") == "honeypot_proxy":
            return "honeypot"
        if intervention in (
            "emergency_fork", "require_justification", "block",
            "quarantine_tool", "quarantine_agent", "reduce_privs",
        ):
            return "block"
        if risk_score > 0.7 or routing.get("audit_level") == "CRITICAL":
            return "escalate"
        return "allow"

    def _handle_honeypot(self, request: MCPRequest, routing: Dict) -> MCPResponse:
        """Route Worker's call to honeypot — Worker receives synthetic success, never knows."""
        honeypot = self._servers["honeypot-db"]
        result = honeypot.handle(request)
        self._servers["audit-log"].record_intercept(request, routing, "honeypot")
        return MCPResponse(
            id=request.id,
            result=result,
            intercepted=True,
            routed_to="honeypot",
            guardian_action="honeypot",
        )

    def _handle_block(self, request: MCPRequest, routing: Dict) -> MCPResponse:
        """Return a synthetic MCP Error — tool never executes."""
        primary_server = self._servers.get(routing.get("primary", "audit-log"))
        if primary_server:
            primary_server.handle_block(request, routing)
        self._servers["audit-log"].record_intercept(request, routing, "blocked")
        return MCPResponse(
            id=request.id,
            error={
                "code": MCPErrorCode.TOOL_BLOCKED_BY_GUARDIAN,
                "message": (
                    f"GUARDIAN blocked tool '{request.tool_name}' — "
                    f"classified as {routing.get('description', 'security violation')}. "
                    f"Action: {routing.get('ir_action', 'block')}."
                ),
                "data": {
                    "blocked_tool": request.tool_name,
                    "guardian_routing": routing,
                    "mcp_server_chain": [routing.get("primary"), routing.get("secondary")],
                },
            },
            intercepted=True,
            routed_to="blocked",
            guardian_action="block",
        )

    def _handle_escalate(self, request: MCPRequest, routing: Dict) -> MCPResponse:
        """Forward request + trigger autonomous IR swarm on relevant MCP servers."""
        primary = self._servers.get(routing.get("primary", "audit-log"))
        ir_result = {}
        if primary:
            ir_result = primary.handle_ir(request, routing)

        secondary_name = routing.get("secondary")
        if secondary_name and secondary_name in self._servers:
            self._servers[secondary_name].handle_ir(request, routing)

        self._ir_log.append({
            "mcp_request_id": request.id,
            "tool": request.tool_name,
            "ir_action": routing.get("ir_action"),
            "primary_server": routing.get("primary"),
            "secondary_server": routing.get("secondary"),
            "ir_result": ir_result,
            "step": request.episode_step,
        })
        self._servers["audit-log"].record_intercept(request, routing, "escalated")

        return MCPResponse(
            id=request.id,
            result={
                "status": "forwarded",
                "ir_activated": True,
                "ir_action": routing.get("ir_action"),
            },
            intercepted=True,
            routed_to="production+ir",
            guardian_action="escalate",
            ir_triggered=True,
        )

    def _handle_forward(
        self, request: MCPRequest, routing: Dict, action: str
    ) -> MCPResponse:
        """Pass through unchanged — log to audit server."""
        self._servers["audit-log"].record_intercept(request, routing, "forwarded")
        return MCPResponse(
            id=request.id,
            result={"status": "forwarded", "tool": request.tool_name},
            intercepted=False,
            routed_to="production",
            guardian_action=action,
        )
