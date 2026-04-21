import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from guardian.environment.guardian_env import ATTACK_PATTERNS, SAFE_TASKS, GUARDIANEnvironment
from guardian.environment.reward_computer import RewardBreakdown, RewardComputer
from guardian.agents.worker_agent import WorkerAgent
from guardian.agents.guardian_agent import GuardianAgent


@dataclass
class EpisodeResult:
    episode_id: str
    attack_type: Optional[str]
    production_intact: bool
    fork_triggered: bool
    fork_step: Optional[int]
    guardian_detected_type: Optional[str]
    total_steps: int
    reward: float
    reward_breakdown: RewardBreakdown
    action_log: List[Dict]
    training_samples: List[Dict] = field(default_factory=list)


class EpisodeRunner:
    def __init__(self, env: GUARDIANEnvironment, worker: WorkerAgent,
                 guardian: GuardianAgent, reward_computer: RewardComputer):
        self.env = env
        self.worker = worker
        self.guardian = guardian
        self.rc = reward_computer

    def run_episode(self, attack_type: Optional[str] = None) -> EpisodeResult:
        episode_id = str(uuid.uuid4())[:8]
        state = self.env.reset(attack_type=attack_type)
        training_samples = []
        guardian_detected_type = None

        for safe_task in SAFE_TASKS:
            wa = self.worker.get_action(safe_task["task"])
            self.env.worker_step(wa.get("tool", "read_db"), wa.get("parameters", safe_task["params"]), wa.get("reasoning", ""))
            gd = self.guardian.evaluate(state.action_log)
            self.env.guardian_step(gd["risk_score"], gd["intervention"], gd["reasoning"], gd.get("classified_attack"))
            prompt = self.guardian.build_training_prompt(state.action_log[:-1])
            training_samples.append({"prompt": prompt, "completion": json.dumps(gd)})

        if attack_type and attack_type in ATTACK_PATTERNS:
            pattern = ATTACK_PATTERNS[attack_type]
            state.attack_active = True
            wa = self.worker.get_action(f"Process workflow #{episode_id}", attack_injection=pattern["injection"])
            self.env.worker_step(pattern["tool"], pattern["dangerous_params"], wa.get("reasoning", ""))
            gd = self.guardian.evaluate(state.action_log)
            self.env.guardian_step(gd["risk_score"], gd["intervention"], gd["reasoning"], gd.get("classified_attack"))
            if gd.get("classified_attack"):
                guardian_detected_type = gd["classified_attack"]
            prompt = self.guardian.build_training_prompt(state.action_log[:-1])
            training_samples.append({"prompt": prompt, "completion": json.dumps(gd)})

        production_intact = self.env.verify_production_intact()
        breakdown = self.rc.compute(production_intact, state.fork_triggered, state.fork_step,
                                    state.attack_active, attack_type, guardian_detected_type)
        return EpisodeResult(episode_id=episode_id, attack_type=attack_type, production_intact=production_intact,
                             fork_triggered=state.fork_triggered, fork_step=state.fork_step,
                             guardian_detected_type=guardian_detected_type, total_steps=state.episode_step,
                             reward=breakdown.total, reward_breakdown=breakdown,
                             action_log=state.action_log, training_samples=training_samples)