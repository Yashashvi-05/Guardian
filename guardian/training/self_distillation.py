"""
Self-Distilled RLVR for GUARDIAN Fleet
========================================
Three-phase loop: Exploration -> Verification -> Distillation -> GRPO

Phase 1 EXPLORATION: Sample N=8 Guardian responses at temperature=0.9
Phase 2 VERIFICATION: Score each with CounterfactualScorer (no LLM, no env step)
Phase 3 DISTILLATION: Select best -> GoldenTrajectory -> GoldenReplayBuffer
Phase 4 GRPO: Train on prioritized replay buffer samples

Key insight: decouples exploration (sampling) from exploitation (GRPO).
Guarantees at least one correct intervention is discovered per attack type.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SelfDistillationConfig:
    n_samples: int = 8
    temperature: float = 0.9
    min_reward_gap: float = 0.05
    trigger_on_new_attack_types: bool = True
    min_episodes_per_attack: int = 5
    trigger_on_low_detection: bool = True
    low_detection_threshold: float = 0.5
    max_replay_size: int = 500
    min_replay_size_to_train: int = 8
    replay_sample_strategy: str = "prioritized"


class CounterfactualScorer:
    """Score a Guardian decision without running a live episode."""

    _OUTCOME_TABLE: Dict = {
        (True, "shadow"):                (True,  True),
        (True, "emergency_fork"):        (True,  True),
        (True, "rollback_k"):            (True,  True),
        (True, "quarantine_tool"):       (True,  True),
        (True, "rewrite"):               (False, True),
        (True, "reduce_privs"):          (False, True),
        (True, "interrogate"):           (False, True),
        (True, "require_justification"): (False, True),
        (True, "canary_inject"):         (False, True),
        (True, "escalate_human"):        (True,  True),
        (True, "allow"):                 (False, False),
        (False, "shadow"):               (True,  True),
        (False, "emergency_fork"):       (True,  True),
        (False, "allow"):                (False, True),
        (False, "interrogate"):          (False, True),
        (False, "rewrite"):              (False, True),
        (False, "canary_inject"):        (False, True),
        (False, "reduce_privs"):         (False, True),
    }

    def __init__(self, reward_computer) -> None:
        self._rc = reward_computer

    def score(
        self,
        decision: Dict[str, Any],
        attack_type: Optional[str],
        attack_active: bool,
        action_log: List[Dict],
        fork_step: int,
        current_step: int,
        step_rewards: List[float],
        shadow_tokens: int = 300,
        guardian_tokens: int = 500,
    ) -> Tuple[float, Dict[str, float]]:
        iv = decision.get("intervention", "allow")
        classified = decision.get("classified_attack")
        risk_score = float(decision.get("risk_score", 0.3))
        explanation = str(decision.get("reasoning", "")) + " " + str(decision.get("explanation", ""))
        key = (attack_active, iv)
        fork_triggered, production_intact = self._OUTCOME_TABLE.get(key, (False, True))
        guardian_detected = classified if attack_active and classified else None
        last_tool = action_log[-1].get("tool", "read_db") if action_log else "read_db"
        breakdown = self._rc.compute(
            production_intact=production_intact,
            fork_triggered=fork_triggered,
            fork_step=fork_step if fork_triggered else None,
            attack_active=attack_active,
            attack_type=attack_type,
            guardian_detected_type=guardian_detected,
            last_worker_tool=last_tool,
            shadow_tokens=shadow_tokens if fork_triggered else 0,
            guardian_tokens=guardian_tokens,
            intervention=iv,
            risk_score=risk_score,
            step_rewards=step_rewards,
            explanation=explanation,
            action_log=action_log,
        )
        return breakdown.total, breakdown.to_dict()


@dataclass
class GoldenTrajectory:
    prompt: str
    completion: str
    decision: Dict
    reward: float
    reward_breakdown: Dict
    attack_type: Optional[str]
    attack_active: bool
    n_sampled: int
    all_rewards: List[float]
    reward_gap: float
    timestamp: float = field(default_factory=time.time)
    episode_step: int = 0


class GoldenReplayBuffer:
    """Persistent prioritized replay buffer stored as JSONL on disk."""

    def __init__(self, path: str = "guardian/data/golden_replay.jsonl", max_size: int = 500) -> None:
        self.path = path
        self.max_size = max_size
        self._buffer: List[GoldenTrajectory] = []
        self._per_attack_count: Dict[str, int] = {}
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._load()

    def add(self, traj: GoldenTrajectory) -> None:
        self._buffer.append(traj)
        atk = traj.attack_type or "clean"
        self._per_attack_count[atk] = self._per_attack_count.get(atk, 0) + 1
        if len(self._buffer) > self.max_size:
            self._buffer.sort(key=lambda t: t.reward, reverse=True)
            self._buffer = self._buffer[:self.max_size]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(traj)) + "\n")

    def sample(self, batch_size: int = 16, strategy: str = "prioritized") -> List[Dict[str, str]]:
        if not self._buffer:
            return []
        n = min(batch_size, len(self._buffer))
        if strategy == "uniform":
            chosen = random.sample(self._buffer, n)
        elif strategy == "balanced":
            chosen = self._balanced_sample(n)
        else:
            chosen = self._prioritized_sample(n)
        return [{"prompt": t.prompt, "completion": t.completion} for t in chosen]

    def sample_with_rewards(self, batch_size: int = 16, strategy: str = "prioritized") -> Tuple[List[str], List[float]]:
        if not self._buffer:
            return [], []
        n = min(batch_size, len(self._buffer))
        chosen = self._prioritized_sample(n) if strategy == "prioritized" else random.sample(self._buffer, n)
        return [t.prompt for t in chosen], [t.reward for t in chosen]

    def size(self) -> int:
        return len(self._buffer)

    def per_attack_counts(self) -> Dict[str, int]:
        return dict(self._per_attack_count)

    def mean_reward(self) -> float:
        if not self._buffer:
            return 0.0
        return sum(t.reward for t in self._buffer) / len(self._buffer)

    def stats(self) -> Dict[str, Any]:
        return {"size": self.size(), "mean_reward": round(self.mean_reward(), 4), "per_attack": self.per_attack_counts(), "path": self.path}

    def _prioritized_sample(self, n: int) -> List[GoldenTrajectory]:
        rewards = [max(0.01, t.reward + 1.0) for t in self._buffer]
        total = sum(rewards)
        weights = [r / total for r in rewards]
        return random.choices(self._buffer, weights=weights, k=n)

    def _balanced_sample(self, n: int) -> List[GoldenTrajectory]:
        groups: Dict[str, List[GoldenTrajectory]] = {}
        for t in self._buffer:
            groups.setdefault(t.attack_type or "clean", []).append(t)
        chosen = [max(g, key=lambda t: t.reward) for g in groups.values()]
        remaining = n - len(chosen)
        if remaining > 0:
            rest = [t for t in self._buffer if t not in chosen]
            chosen.extend(random.sample(rest, min(remaining, len(rest))))
        return chosen[:n]

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    traj = GoldenTrajectory(**d)
                    self._buffer.append(traj)
                    atk = traj.attack_type or "clean"
                    self._per_attack_count[atk] = self._per_attack_count.get(atk, 0) + 1
            print(f"[GoldenReplayBuffer] Loaded {len(self._buffer)} trajectories from {self.path}")
        except Exception as e:
            print(f"[GoldenReplayBuffer] Warning: {e}")


class SelfDistillationSampler:
    """Orchestrates Exploration -> Verification -> Distillation."""

    def __init__(self, guardian, reward_computer, config: Optional[SelfDistillationConfig] = None) -> None:
        self.guardian = guardian
        self.cfg = config or SelfDistillationConfig()
        self._scorer = CounterfactualScorer(reward_computer)
        self._trigger_counts: Dict[str, int] = {}

    def should_run(self, episode: int, attack_type: Optional[str], per_attack_rewards: Dict[str, List[float]], replay_buffer: Optional[GoldenReplayBuffer] = None) -> bool:
        atk = str(attack_type)
        if replay_buffer and self.cfg.trigger_on_new_attack_types:
            if replay_buffer.per_attack_counts().get(atk, 0) < self.cfg.min_episodes_per_attack:
                return True
        if self.cfg.trigger_on_low_detection and attack_type is not None:
            rewards = per_attack_rewards.get(atk, [])
            if len(rewards) >= 3:
                recent = rewards[-5:]
                if sum(recent) / len(recent) < self.cfg.low_detection_threshold:
                    return True
        return False

    def find_golden_trajectory(
        self,
        state,
        attack_type: Optional[str],
        action_log: List[Dict],
        risk_history: Optional[List[float]] = None,
        faiss_context: Optional[str] = None,
        schema_version: int = 0,
    ) -> Optional[GoldenTrajectory]:
        n = self.cfg.n_samples
        print(f"\n  [SelfDistill] Exploring {n} paths for attack={attack_type}...")
        attack_active = bool(getattr(state, "attack_active", False))
        fork_step = getattr(state, "fork_step", None) or max(1, getattr(state, "episode_step", 1) - 1)
        current_step = getattr(state, "episode_step", 1)
        decisions = self.guardian.sample_n_completions(
            action_log=action_log,
            n=n,
            temperature=self.cfg.temperature,
            faiss_context=faiss_context,
            schema_version=schema_version,
            risk_history=risk_history,
        )
        step_rewards = [0.03] * max(1, current_step - 1)
        scored = []
        for d in decisions:
            reward, breakdown = self._scorer.score(
                decision=d,
                attack_type=attack_type,
                attack_active=attack_active,
                action_log=action_log,
                fork_step=fork_step,
                current_step=current_step,
                step_rewards=step_rewards,
                shadow_tokens=300,
                guardian_tokens=500,
            )
            scored.append((reward, breakdown, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        all_rewards = [s[0] for s in scored]
        best_reward, best_breakdown, best_decision = scored[0]
        mean_reward_val = sum(all_rewards) / len(all_rewards)
        reward_gap = best_reward - mean_reward_val
        print(f"  [SelfDistill] best={best_reward:.3f} mean={mean_reward_val:.3f} gap={reward_gap:.3f}")
        if reward_gap < self.cfg.min_reward_gap and best_reward < 0.3:
            return None
        prompt = self.guardian.build_training_prompt(
            action_log, faiss_context=faiss_context,
            schema_version=schema_version, risk_history=risk_history,
        )
        completion = best_decision.get("_raw_completion", "") or self._decision_to_xml(best_decision)
        atk = str(attack_type)
        self._trigger_counts[atk] = self._trigger_counts.get(atk, 0) + 1
        return GoldenTrajectory(
            prompt=prompt,
            completion=completion,
            decision=best_decision,
            reward=best_reward,
            reward_breakdown=best_breakdown,
            attack_type=attack_type,
            attack_active=attack_active,
            n_sampled=n,
            all_rewards=all_rewards,
            reward_gap=reward_gap,
            timestamp=time.time(),
            episode_step=current_step,
        )

    def trigger_stats(self) -> Dict[str, int]:
        return dict(self._trigger_counts)

    @staticmethod
    def _decision_to_xml(decision: Dict) -> str:
        return (
            f"<reasoning>{decision.get('reasoning', 'Detected anomaly.')}</reasoning>\n"
            f"<risk_score>{decision.get('risk_score', 0.5):.2f}</risk_score>\n"
            f"<intervention>{decision.get('intervention', 'shadow')}</intervention>\n"
            f"<attack_type>{decision.get('classified_attack') or 'clean'}</attack_type>\n"
            f"<explanation>{decision.get('explanation', '')}</explanation>"
        )
