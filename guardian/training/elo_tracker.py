"""
ELO Attack Rating System
=========================
Competitive ELO leaderboard for attack types vs Guardian.

Guardian blocks attack -> attack ELO drops
Curriculum agent generates harder variant -> attack ELO rises
Guardian detects correctly -> Guardian ELO rises

Saved to guardian/data/elo_ratings.json after each update.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Optional


K_FACTOR = 32
INITIAL_ELO = 1000


class ELOTracker:
    """Tracks ELO ratings for Guardian vs each attack type."""

    def __init__(self, path: str = "guardian/data/elo_ratings.json") -> None:
        self.path = path
        self._attack_elos: Dict[str, float] = {}
        self._guardian_elo: float = INITIAL_ELO
        self._history: list = []
        self._load()

    def get_attack_elo(self, attack_type: str) -> float:
        return self._attack_elos.get(attack_type, INITIAL_ELO)

    def get_guardian_elo(self) -> float:
        return self._guardian_elo

    def update(self, attack_type: Optional[str], guardian_detected: bool, reward: float) -> Dict:
        """Update ELO ratings after one episode."""
        if attack_type is None:
            return {}

        atk_elo = self.get_attack_elo(attack_type)
        guardian_elo = self._guardian_elo

        expected_guardian = 1.0 / (1.0 + 10.0 ** ((atk_elo - guardian_elo) / 400.0))
        expected_attack = 1.0 - expected_guardian

        actual_guardian = 1.0 if guardian_detected else 0.0
        actual_attack = 1.0 - actual_guardian

        new_guardian = guardian_elo + K_FACTOR * (actual_guardian - expected_guardian)
        new_attack = atk_elo + K_FACTOR * (actual_attack - expected_attack)

        self._guardian_elo = new_guardian
        self._attack_elos[attack_type] = new_attack

        entry = {
            "attack_type": attack_type,
            "guardian_detected": guardian_detected,
            "reward": reward,
            "guardian_elo_before": round(guardian_elo, 1),
            "guardian_elo_after": round(new_guardian, 1),
            "attack_elo_before": round(atk_elo, 1),
            "attack_elo_after": round(new_attack, 1),
        }
        self._history.append(entry)
        self._save()
        return entry

    def leaderboard(self) -> list:
        """Return attacks sorted by ELO descending (hardest first)."""
        rows = []
        for atk, elo in sorted(self._attack_elos.items(), key=lambda x: x[1], reverse=True):
            trend = "up" if self._recent_trend(atk) > 0 else ("down" if self._recent_trend(atk) < 0 else "flat")
            rows.append({"attack_type": atk, "elo": round(elo, 1), "trend": trend})
        return rows

    def _recent_trend(self, attack_type: str) -> float:
        recent = [e for e in self._history[-20:] if e["attack_type"] == attack_type]
        if len(recent) < 2:
            return 0.0
        return recent[-1]["attack_elo_after"] - recent[0]["attack_elo_before"]

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({
                "guardian_elo": round(self._guardian_elo, 1),
                "attack_elos": {k: round(v, 1) for k, v in self._attack_elos.items()},
                "leaderboard": self.leaderboard(),
            }, f, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._guardian_elo = float(d.get("guardian_elo", INITIAL_ELO))
            self._attack_elos = {k: float(v) for k, v in d.get("attack_elos", {}).items()}
        except Exception:
            pass
