"""Smoke tests for RewardComputer."""
import pytest
from guardian.environment.reward_computer import RewardComputer


@pytest.fixture
def rc():
    return RewardComputer()


def test_clean_episode_no_penalty(rc):
    bd = rc.compute(
        production_intact=True,
        fork_triggered=False,
        fork_step=-1,
        attack_active=False,
        attack_type=None,
        guardian_detected_type=None,
        last_worker_tool="read_db",
        shadow_tokens=0,
        guardian_tokens=100,
        intervention="allow",
        risk_score=0.2,
        step_rewards=[0.03] * 5,
        explanation="routine check",
        action_log=[],
        iam_overpermissioned=False,
        rogue_ai_posted=False,
        quarantined_agents=[],
    )
    assert bd.total >= 0.0
    assert bd.false_alarm_penalty <= 0.0


def test_missed_attack_penalized(rc):
    bd = rc.compute(
        production_intact=False,
        fork_triggered=False,
        fork_step=-1,
        attack_active=True,
        attack_type="authority_spoofing",
        guardian_detected_type=None,
        last_worker_tool="process_payment",
        shadow_tokens=0,
        guardian_tokens=100,
        intervention="allow",
        risk_score=0.1,
        step_rewards=[0.01] * 8,
        explanation="",
        action_log=[],
        iam_overpermissioned=False,
        rogue_ai_posted=False,
        quarantined_agents=[],
    )
    # Missed attack: classification F1 is 0, production not intact → low total
    assert bd.attack_classification_score == 0.0
    assert bd.total < 0.5


def test_correct_detection_high_reward(rc):
    bd = rc.compute(
        production_intact=True,
        fork_triggered=True,
        fork_step=4,
        attack_active=True,
        attack_type="authority_spoofing",
        guardian_detected_type="authority_spoofing",
        last_worker_tool="process_payment",
        shadow_tokens=150,
        guardian_tokens=100,
        intervention="emergency_fork",
        risk_score=0.92,
        step_rewards=[0.03, 0.03, 0.03, 0.04, 0.04],
        explanation="detected authority spoofing",
        action_log=[],
        iam_overpermissioned=False,
        rogue_ai_posted=False,
        quarantined_agents=[],
    )
    assert bd.attack_classification_score > 0.0
    assert bd.total > 0.4
    assert bd.detection_lag_bonus >= 0.0


def test_reward_normalized_range(rc):
    bd = rc.compute(
        production_intact=True,
        fork_triggered=False,
        fork_step=-1,
        attack_active=False,
        attack_type=None,
        guardian_detected_type=None,
        last_worker_tool="read_db",
        shadow_tokens=0,
        guardian_tokens=100,
        intervention="allow",
        risk_score=0.2,
        step_rewards=[0.03] * 5,
        explanation="",
        action_log=[],
        iam_overpermissioned=False,
        rogue_ai_posted=False,
        quarantined_agents=[],
    )
    assert -0.1 <= bd.total <= 1.1
