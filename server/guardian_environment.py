"""
GUARDIAN Fleet — OpenEnv Server-Side Environment
==================================================
GuardianOpenEnvEnvironment wraps the existing GUARDIANGymEnv (Gymnasium env)
as an OpenEnv-compatible server-side Environment.

This is the bridge between:
  guardian/environment/openenv_wrapper.py  (Gymnasium)
      ↓  wrapped by
  server/guardian_environment.py           (OpenEnv server protocol)
      ↓  served by
  server/app.py                            (FastAPI / WebSocket)
      ↓  consumed by
  client.py                                (async GuardianEnv client)

Entry point referenced in openenv.yaml:
    entry_point: server.guardian_environment:GuardianOpenEnvEnvironment
"""

from __future__ import annotations

import os
import sys
import uuid
from typing import Any, Dict, Optional, Tuple

# ── Ensure repo root is ALWAYS importable, regardless of working directory ──
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import GuardianAction, GuardianObservation, GuardianState, StepResult

# ── Import the existing Gymnasium environment ────────────────────────────────
try:
    from guardian.environment.openenv_wrapper import GUARDIANGymEnv
    from guardian.environment.reward_computer import RewardComputer
except ImportError as _err:
    raise ImportError(
        f"Cannot import guardian inner package: {_err}\n"
        "Make sure you have run `pip install -e .` from the repo root, "
        "or that the repo root is in PYTHONPATH."
    ) from _err


class GuardianOpenEnvEnvironment:
    """
    OpenEnv-compatible wrapper around GUARDIANGymEnv.

    Implements the three OpenEnv server methods:
        reset(options)            → GuardianObservation
        step(action)              → (GuardianObservation, reward, done, info)
        state()                   → GuardianState

    Each call maps as follows:
        GuardianAction.to_gym_action()  →  GUARDIANGymEnv.step(gym_action)
        Gymnasium obs dict              →  GuardianObservation
        Gymnasium info dict             →  merged into StepResult.info

    The environment tracks one episode at a time (stateful).
    A new WebSocket connection gets its own GuardianOpenEnvEnvironment instance
    (see server/app.py) so sessions are fully isolated.
    """

    def __init__(
        self,
        difficulty: int = 1,
        attack_pool: Optional[list] = None,
    ) -> None:
        self._gym = GUARDIANGymEnv(difficulty=difficulty, attack_pool=attack_pool)
        self._episode_id: str = str(uuid.uuid4())[:8]
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._done: bool = False
        self._last_breakdown: Dict[str, float] = {}
        # Auto-reset the gym so _risk_history and all internal state is
        # properly initialised even if the client sends step() without reset().
        self._gym.reset()

    # ── Core OpenEnv API ────────────────────────────────────────────────────

    def reset(self, options: Optional[Dict] = None) -> GuardianObservation:
        """
        Start a new episode.

        options (dict, optional):
            seed (int) — RNG seed for reproducibility
        Returns:
            GuardianObservation — initial telemetry (no attack yet)
        """
        self._episode_id = str(uuid.uuid4())[:8]
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._done = False
        self._last_breakdown = {}

        seed: Optional[int] = None
        if options and isinstance(options, dict):
            seed = options.get("seed")

        obs_dict, _info = self._gym.reset(seed=seed)
        return self._gym_obs_to_model(obs_dict)

    def step(
        self,
        action: GuardianAction,
    ) -> Tuple[GuardianObservation, float, bool, Dict]:
        """
        Execute one Guardian decision.

        Internally:
          1. Converts GuardianAction → Gymnasium dict action
          2. Runs the next Worker step (inside GUARDIANGymEnv.step)
          3. Applies the Guardian decision
          4. Computes the 11-component reward on episode end
          5. Returns (GuardianObservation, reward, done, info)

        Args:
            action: GuardianAction from the client / LLM

        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            # Auto-reset so callers don't get stuck
            obs = self.reset()
            return obs, 0.0, False, {"auto_reset": True}

        # Convert to Gymnasium action dict
        gym_action = action.to_gym_action(
            self._gym.INTERVENTIONS,
            self._gym.ATTACK_TYPES,
        )

        obs_dict, reward, terminated, truncated, info = self._gym.step(gym_action)

        self._step_count += 1
        self._cumulative_reward += reward
        self._done = terminated or truncated

        # Build typed observation
        obs = self._gym_obs_to_model(obs_dict, info=info)

        # On episode end, attach reward breakdown and hash audit
        if self._done:
            # GUARDIANGymEnv doesn't expose breakdown via info yet,
            # so we read it from the last state directly
            obs.hash_chain_report = self._gym.env.get_hash_chain_report()
            obs.production_intact = self._gym.env.verify_production_intact()
            obs.fork_triggered = self._gym.env.state.fork_triggered

        return obs, float(reward), self._done, _safe_json(info)

    def state(self) -> GuardianState:
        """
        Return current episode metadata (no environment state is changed).

        Returns:
            GuardianState with episode_id, step_count, difficulty, attack_active,
            cumulative_reward
        """
        return GuardianState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            difficulty=self._gym.difficulty_manager.current_level,
            attack_active=bool(self._gym.env.state.attack_active),
            cumulative_reward=round(self._cumulative_reward, 6),
        )

    # ── Internal helpers ────────────────────────────────────────────────────

    def _gym_obs_to_model(
        self,
        obs_dict: Dict[str, Any],
        info: Optional[Dict] = None,
    ) -> GuardianObservation:
        """
        Convert the raw Gymnasium observation dict to a typed GuardianObservation.

        The Gymnasium obs dict has:
            action_log_json (str)
            current_step    (np.ndarray shape (1,))
            attack_active   (int 0/1)
            difficulty      (int, 0-indexed)
            schema_version  (int)
            risk_history    (np.ndarray shape (10,))
        """
        # risk_history may be a numpy array
        risk_history = obs_dict.get("risk_history", [])
        if hasattr(risk_history, "tolist"):
            risk_history = risk_history.tolist()

        # current_step may be a numpy array of shape (1,)
        current_step_raw = obs_dict.get("current_step", [0])
        if hasattr(current_step_raw, "__iter__") and not isinstance(
            current_step_raw, (str, bytes)
        ):
            current_step = int(list(current_step_raw)[0])
        else:
            current_step = int(current_step_raw)

        # production_intact / fork_triggered from info (if available) or gym state
        if info:
            production_intact = bool(info.get("production_intact", True))
            fork_triggered = bool(info.get("fork_triggered", False))
        else:
            production_intact = self._gym.env.verify_production_intact()
            fork_triggered = self._gym.env.state.fork_triggered

        # difficulty in Gymnasium obs is 0-indexed; expose as 1-indexed
        difficulty_raw = obs_dict.get("difficulty", 0)
        difficulty = int(difficulty_raw) + 1

        return GuardianObservation(
            action_log_json=obs_dict.get("action_log_json", "[]"),
            multi_app_log_json=obs_dict.get("multi_app_log_json", "[]"),
            current_step=current_step,
            attack_active=bool(obs_dict.get("attack_active", 0)),
            production_intact=production_intact,
            fork_triggered=fork_triggered,
            difficulty=difficulty,
            schema_version=int(obs_dict.get("schema_version", 0)),
            risk_history=risk_history,
            reward_breakdown={},
            hash_chain_report="",
            rogue_ai_posted=bool(obs_dict.get("rogue_ai_posted", 0)),
            iam_overpermissioned=bool(obs_dict.get("iam_overpermissioned", 0)),
        )


# ── JSON-safe info dict ──────────────────────────────────────────────────────

def _safe_json(info: Dict) -> Dict:
    """Strip non-serialisable values from Gymnasium info dict."""
    safe: Dict[str, Any] = {}
    for k, v in info.items():
        try:
            import json
            json.dumps(v)
            safe[k] = v
        except (TypeError, ValueError):
            safe[k] = str(v)
    return safe
