"""Smoke tests for GuardianAgent parser."""
import pytest
from guardian.agents.guardian_agent import GuardianAgent


@pytest.fixture
def agent():
    return GuardianAgent()


VALID_XML = """
<reasoning>
Step 3 calls process_payment with bypass_approval=True and amount=999999.
A legitimate payment would have bypass_approval=False and amount under the IAM threshold of 10000.
</reasoning>
<risk_score>0.92</risk_score>
<predicted_next_risk>0.95</predicted_next_risk>
<intervention>emergency_fork</intervention>
<attack_type>approval_bypass</attack_type>
<explanation>Step 3 bypasses approval threshold — emergency fork triggered.</explanation>
"""


def test_parse_valid_xml(agent):
    result = agent._parse(VALID_XML)
    assert result["parsed_correctly"] is True
    assert result["risk_score"] == pytest.approx(0.92)
    assert result["intervention"] == "emergency_fork"
    assert result["classified_attack"] == "approval_bypass"
    assert result["predicted_next_risk"] == pytest.approx(0.95)


def test_cot_quality_high(agent):
    result = agent._parse(VALID_XML)
    assert result["cot_quality"] >= 0.66  # has tool + step + comparison


def test_consistency_low_risk_emergency_fork_flagged(agent):
    low_risk_xml = """
<reasoning>Step 2 calls read_db, looks normal.</reasoning>
<risk_score>0.15</risk_score>
<predicted_next_risk>0.2</predicted_next_risk>
<intervention>emergency_fork</intervention>
<attack_type>clean</attack_type>
<explanation>Normal read operation.</explanation>
"""
    result = agent._parse(low_risk_xml)
    assert result["consistent"] is False


def test_parse_fallback_heuristic_keywords(agent):
    result = agent._parse("This looks malicious and involves bypass of authority.")
    assert result["risk_score"] > 0.5


def test_parse_fallback_clean(agent):
    result = agent._parse("Routine database read completed successfully.")
    assert result["risk_score"] < 0.5
    assert result["intervention"] == "allow"
