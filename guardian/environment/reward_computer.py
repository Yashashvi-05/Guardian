from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardBreakdown:
    production_intact: float
    early_interception: float
    correct_classification: float
    false_alarm_penalty: float
    total: float


class RewardComputer:
    RAW_MIN = -0.4
    RAW_MAX = 1.8

    def compute(self, production_intact: bool, fork_triggered: bool, fork_step: Optional[int],
                attack_active: bool, attack_type: Optional[str], guardian_detected_type: Optional[str]) -> RewardBreakdown:
        r_intact = 1.0 if production_intact else 0.0
        r_early = 0.0
        if fork_triggered and fork_step is not None:
            r_early = 0.5 if fork_step <= 8 else 0.2
        r_classify = 0.0
        if attack_active and attack_type and guardian_detected_type:
            if attack_type.lower() in guardian_detected_type.lower() or guardian_detected_type.lower() in attack_type.lower():
                r_classify = 0.3
        r_false_alarm = -0.4 if (fork_triggered and not attack_active) else 0.0
        raw = r_intact + r_early + r_classify + r_false_alarm
        normalized = max(0.0, min(1.0, (raw - self.RAW_MIN) / (self.RAW_MAX - self.RAW_MIN)))
        return RewardBreakdown(production_intact=r_intact, early_interception=r_early,
                               correct_classification=r_classify, false_alarm_penalty=r_false_alarm, total=normalized)