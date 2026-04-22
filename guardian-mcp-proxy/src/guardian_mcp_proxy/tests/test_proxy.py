"""Unit tests for the GUARDIAN MCP proxy."""
import asyncio
import pytest
from guardian_mcp_proxy.proxy import GuardianProxy


@pytest.fixture
def proxy():
    return GuardianProxy({
        "upstream_url": "http://localhost:9999",
        "hmac_secret": "test-secret",
        "risk_threshold": 0.55,
        "audit_log": "/dev/null",
    })


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def make_request(tool: str, args: dict, req_id: str = "1"):
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "method": "tools/call",
        "params": {"name": tool, "arguments": args},
    }


def test_rate_limit_exceeded(proxy):
    # process_payment limit = 3
    for _ in range(3):
        proxy._rate_counts["process_payment"] += 1
    assert not proxy._check_rate("process_payment")


def test_rate_limit_within(proxy):
    assert proxy._check_rate("read_db")


def test_heuristic_bypass_detection(proxy):
    result = run(proxy._evaluate_risk("process_payment", {"bypass_approval": True, "amount": 50000}))
    assert result["risk_score"] >= 0.7
    assert result["intervention"] in ("shadow", "interrogate", "block", "emergency_fork")


def test_heuristic_clean_action(proxy):
    result = run(proxy._evaluate_risk("read_db", {"table": "products", "limit": 10}))
    assert result["risk_score"] < 0.5
    assert result["intervention"] == "allow"


def test_sign_roundtrip(proxy):
    sig = proxy._sign("read_db", {"table": "users"})
    assert len(sig) == 16


def test_rewrite_strips_bypass(proxy):
    rewritten = proxy._apply_rewrite("process_payment", {
        "amount": 100,
        "bypass_approval": True,
        "trigger_approval": True,
    })
    assert rewritten["bypass_approval"] is False
    assert rewritten["trigger_approval"] is False
    assert rewritten["amount"] == 100


def test_intercept_log_grows(proxy):
    proxy._rate_counts["process_payment"] = 0  # reset
    # Manually add intercept
    proxy._intercept_log.append({"step": 1, "risk_score": 0.9, "intervention": "block"})
    summary = proxy.get_summary()
    assert summary["blocked"] == 1
