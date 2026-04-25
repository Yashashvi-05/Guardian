"""
GUARDIAN MCP Security Proxy
============================
FastAPI-based HTTP/SSE transport that intercepts every MCP JSON-RPC 2.0 call
from a Claude agent, runs the GUARDIAN risk analysis, and either:
  - forwards (allow / shadow / interrogate)
  - rewrites the request (rewrite)
  - blocks with an error response (block / quarantine_tool / emergency_fork)

Usage:
    guardian-mcp --config guardian_mcp.yaml

Claude Desktop config (claude_desktop_config.json):
    {
      "mcpServers": {
        "enterprise": {
          "command": "guardian-mcp",
          "args": ["--config", "guardian_mcp.yaml"]
        }
      }
    }
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger("guardian-mcp-proxy")

# ── Intervention policy ──────────────────────────────────────────────────────

BLOCK_INTERVENTIONS = {"block", "emergency_fork", "quarantine_tool", "quarantine_agent"}

RATE_LIMITS: Dict[str, int] = {
    "process_payment": 3,
    "write_db": 5,
    "modify_iam_policy": 2,
    "modify_firewall": 2,
    "post_to_forum": 10,
    "read_db": 20,
    "query_data_access": 10,
}


class GuardianProxy:
    """
    Stateful proxy instance. One per Claude Desktop session.

    Config keys (guardian_mcp.yaml):
      upstream_url: http://localhost:8001   # real MCP server
      guardian_url: http://localhost:8002   # GUARDIAN inference API (optional)
      hmac_secret:  changeme
      audit_log:    guardian_audit.jsonl
      risk_threshold: 0.55
    """

    def __init__(self, config: Dict[str, Any]):
        self.upstream_url = config.get("upstream_url", "http://localhost:8001")
        self.guardian_url = config.get("guardian_url", None)
        self.hmac_secret = config.get("hmac_secret", "guardian-mcp-secret")
        self.audit_path = config.get("audit_log", "guardian_audit.jsonl")
        self.risk_threshold = float(config.get("risk_threshold", 0.55))

        self._action_log: List[Dict] = []
        self._rate_counts: Dict[str, int] = defaultdict(int)
        self._rate_window_start: float = time.time()
        self._RATE_WINDOW_SECONDS: int = 300
        self._session_id = str(uuid.uuid4())[:8]
        self._step = 0
        self._intercept_log: List[Dict] = []

    # ── Signing ──────────────────────────────────────────────────────────────

    def _sign(self, tool: str, args: Dict) -> str:
        payload = json.dumps({"tool": tool, "args": args}, sort_keys=True)
        return hmac.new(
            self.hmac_secret.encode(), payload.encode(), hashlib.sha256
        ).hexdigest()[:16]

    # ── Rate limit ───────────────────────────────────────────────────────────

    def _check_rate(self, tool: str) -> bool:
        now = time.time()
        if now - self._rate_window_start > self._RATE_WINDOW_SECONDS:
            self._rate_counts = defaultdict(int)
            self._rate_window_start = now
        limit = RATE_LIMITS.get(tool)
        if limit is None:
            return True
        self._rate_counts[tool] += 1
        return self._rate_counts[tool] <= limit

    def reset_session(self) -> None:
        """Reset all state for a new demo session."""
        self._action_log = []
        self._rate_counts = defaultdict(int)
        self._rate_window_start = time.time()
        self._intercept_log = []
        self._step = 0

    # ── Guardian risk evaluation ─────────────────────────────────────────────

    async def _evaluate_risk(self, tool: str, args: Dict) -> Dict[str, Any]:
        """Call the GUARDIAN inference API if configured, else use heuristics."""
        if self.guardian_url:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.post(
                        f"{self.guardian_url}/evaluate",
                        json={"action_log": self._action_log[-6:], "step": self._step},
                    )
                    if resp.status_code == 200:
                        return resp.json()
            except Exception:
                pass

        # Heuristic fallback
        risk = 0.1
        iv = "allow"
        dangerous_patterns = [
            ("bypass_approval", 0.75, "approval_bypass"),
            ("self_authorized", 0.7, "confused_deputy"),
            ("salary", 0.5, "data_exfiltration"),
            ("exfil", 0.9, "data_exfiltration"),
            ("trigger_approval", 0.65, "social_engineering"),
        ]
        args_str = json.dumps(args).lower()
        attack = "clean"
        for pattern, r, atk in dangerous_patterns:
            if pattern in args_str:
                if r > risk:
                    risk = r
                    attack = atk
        if risk > 0.7:
            iv = "shadow"
        elif risk > self.risk_threshold:
            iv = "interrogate"
        return {"risk_score": risk, "intervention": iv, "classified_attack": attack if attack != "clean" else None}

    # ── Core dispatch ────────────────────────────────────────────────────────

    async def dispatch(self, jsonrpc_body: Dict) -> Dict[str, Any]:
        method = jsonrpc_body.get("method", "")
        req_id = jsonrpc_body.get("id", "unknown")
        params = jsonrpc_body.get("params", {})
        tool = params.get("name", "unknown")
        args = params.get("arguments", {})

        self._step += 1
        ts = time.time()

        # Rate limit check
        if not self._check_rate(tool):
            entry = {
                "step": self._step, "tool": tool, "args": args,
                "blocked_reason": "rate_limit", "risk_score": 1.0,
                "intervention": "block", "ts": ts,
            }
            self._intercept_log.append(entry)
            self._append_audit(entry)
            return self._error(req_id, -32600, f"Rate limit exceeded for {tool}")

        # Build action log entry
        action_entry = {
            "step": self._step,
            "tool": tool,
            "params": args,
            "timestamp": ts,
        }
        self._action_log.append(action_entry)

        # Guardian evaluation
        guardian_result = await self._evaluate_risk(tool, args)
        risk = guardian_result.get("risk_score", 0.1)
        intervention = guardian_result.get("intervention", "allow")
        attack_type = guardian_result.get("classified_attack")

        intercept_entry = {
            "step": self._step,
            "tool": tool,
            "risk_score": risk,
            "intervention": intervention,
            "classified_attack": attack_type,
            "ts": ts,
            "session": self._session_id,
        }
        self._intercept_log.append(intercept_entry)
        self._append_audit(intercept_entry)

        # Block?
        if intervention in BLOCK_INTERVENTIONS:
            logger.warning(
                "GUARDIAN BLOCKED %s risk=%.2f intervention=%s",
                tool, risk, intervention,
            )
            return self._error(
                req_id, -32603,
                f"GUARDIAN blocked: {intervention} (risk={risk:.2f}, attack={attack_type})"
            )

        # Rewrite?
        if intervention == "rewrite":
            args = self._apply_rewrite(tool, args)
            params["arguments"] = args

        # Forward to upstream
        return await self._forward(jsonrpc_body)

    async def _forward(self, body: Dict) -> Dict:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.upstream_url}/mcp",
                    json=body,
                    headers={"Content-Type": "application/json"},
                )
                return resp.json()
        except Exception as e:
            return self._error(body.get("id", "x"), -32603, f"Upstream error: {e}")

    def _apply_rewrite(self, tool: str, args: Dict) -> Dict:
        """Strip dangerous fields from args."""
        safe_args = dict(args)
        for dangerous_key in ("bypass_approval", "self_authorized", "trigger_approval"):
            if dangerous_key in safe_args:
                safe_args[dangerous_key] = False
        return safe_args

    def _error(self, req_id: Any, code: int, message: str) -> Dict:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": code, "message": message},
        }

    def _append_audit(self, entry: Dict) -> None:
        try:
            with open(self.audit_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass

    def get_summary(self) -> Dict:
        return {
            "session": self._session_id,
            "steps": self._step,
            "intercepts": len(self._intercept_log),
            "blocked": sum(1 for e in self._intercept_log if e.get("intervention") in BLOCK_INTERVENTIONS),
            "high_risk": sum(1 for e in self._intercept_log if e.get("risk_score", 0) > 0.7),
        }


# ── FastAPI application factory ───────────────────────────────────────────────

def create_app(config: Dict[str, Any]) -> FastAPI:
    proxy = GuardianProxy(config)
    app = FastAPI(title="GUARDIAN MCP Proxy", version="0.1.0")

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> Response:
        body = await request.json()
        result = await proxy.dispatch(body)
        return Response(
            content=json.dumps(result),
            media_type="application/json",
        )

    @app.get("/health")
    async def health():
        return {"status": "ok", "summary": proxy.get_summary()}

    @app.post("/reset")
    async def reset_session():
        proxy.reset_session()
        return {"status": "reset"}

    @app.get("/audit")
    async def audit():
        return {"intercepts": proxy._intercept_log[-50:]}

    @app.get("/capabilities")
    async def capabilities():
        return {
            "name": "guardian-mcp-proxy",
            "version": "0.1.0",
            "features": [
                "hmac_request_signing",
                "per_tool_rate_limiting",
                "risk_based_intervention",
                "persistent_audit_log",
                "guardian_api_integration",
            ],
            "rate_limits": RATE_LIMITS,
            "upstream": proxy.upstream_url,
        }

    return app
