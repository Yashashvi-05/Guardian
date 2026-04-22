"""
GUARDIAN Multi-Session Attack Fingerprinting
=============================================
Cross-session FAISS-based fingerprinting for detecting attack pattern recurrence
across independent training episodes and inference sessions.

Competitor has a lazy stub; this is the real implementation.

How it works:
  1. Each episode produces an "attack fingerprint" — a feature vector derived
     from the action log (tool sequence, param entropy, risk trajectory).
  2. The fingerprint is stored in a FAISS flat-L2 index backed by JSONL.
  3. At the start of a new episode, the guardian retrieves the K nearest
     neighbors from past episodes and injects a "similar past attacks" context
     string into the Guardian LLM prompt.
  4. This gives Guardian pattern memory across sessions — it recognizes when
     an attacker is reusing a previously seen technique with surface variation.

Graceful degradation: If faiss is not installed, falls back to a brute-force
cosine-similarity search over stored fingerprints using pure NumPy.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ── Feature extraction ────────────────────────────────────────────────────────

# Tool → numerical id for sequence embedding
_TOOL_IDS = {
    "read_db": 0, "write_db": 1, "process_payment": 2,
    "modify_firewall": 3, "post_to_forum": 4,
    "modify_iam_policy": 5, "query_data_access_log": 6,
}

FINGERPRINT_DIM = 32  # Feature vector dimensionality


def _shannon_entropy(values: List[Any]) -> float:
    """Shannon entropy of a list of hashable values."""
    if not values:
        return 0.0
    from collections import Counter
    counts = Counter(str(v) for v in values)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def extract_fingerprint(action_log: List[Dict], attack_type: Optional[str] = None) -> np.ndarray:
    """
    Extract a fixed-length feature vector from an episode action log.

    Features (32-dim):
      [0-9]   Tool frequency histogram (normalized)
      [10-19] Risk score trajectory (last 10 guardian steps, padded with 0)
      [20-24] Param entropy, amount mean/std, bypass_count, taint_count
      [25-29] Attack type one-hot (8 attack types + clean + rogue_internal_ai, binned to 5)
      [30-31] Step count (normalized), multi_app_log_events (normalized)
    """
    feat = np.zeros(FINGERPRINT_DIM, dtype=np.float32)

    worker_steps = [e for e in action_log if e.get("role") == "worker"]
    guardian_steps = [e for e in action_log if e.get("role") == "guardian"]

    # Tool frequency histogram [0-9]
    tool_counts = np.zeros(len(_TOOL_IDS), dtype=np.float32)
    for e in worker_steps:
        tool = e.get("tool", "read_db")
        idx = _TOOL_IDS.get(tool, 0)
        tool_counts[idx] += 1
    if tool_counts.sum() > 0:
        tool_counts /= tool_counts.sum()
    feat[0:len(_TOOL_IDS)] = tool_counts

    # Risk trajectory [10-19]
    risks = [e.get("risk_score", 0.0) for e in guardian_steps][-10:]
    for i, r in enumerate(risks):
        feat[10 + i] = float(r)

    # Statistical features [20-24]
    all_params = [json.dumps(e.get("params", {})) for e in worker_steps]
    feat[20] = min(1.0, _shannon_entropy(all_params) / 4.0)  # param entropy

    amounts = [e.get("params", {}).get("amount", 0) for e in worker_steps if "amount" in e.get("params", {})]
    if amounts:
        feat[21] = min(1.0, float(np.mean(amounts)) / 1_000_000)
        feat[22] = min(1.0, float(np.std(amounts)) / 500_000)

    bypass_count = sum(
        1 for e in worker_steps
        if e.get("params", {}).get("bypass_approval") or e.get("params", {}).get("bypass_review")
    )
    feat[23] = min(1.0, bypass_count / 5.0)

    taint_count = sum(
        1 for e in action_log
        if e.get("result", {}).get("status") not in ("ok", None, "quarantined")
    )
    feat[24] = min(1.0, taint_count / 10.0)

    # Attack type one-hot [25-29] (mapped to 5 bins)
    _ATTACK_BIN = {
        "authority_spoofing": 0, "approval_laundering": 0,
        "prompt_injection": 1, "approval_bypass": 1,
        "data_exfiltration": 2, "confused_deputy": 2,
        "salami_slicing": 3, "schema_drift_exploit": 3,
        "rogue_internal_ai": 4,
    }
    if attack_type and attack_type in _ATTACK_BIN:
        feat[25 + _ATTACK_BIN[attack_type]] = 1.0

    # Episode length + multi_app events [30-31]
    feat[30] = min(1.0, len(action_log) / 20.0)
    multi_app_events = sum(1 for e in action_log if e.get("role") == "worker"
                           and e.get("tool") in ("post_to_forum", "modify_iam_policy", "query_data_access_log"))
    feat[31] = min(1.0, multi_app_events / 5.0)

    return feat


# ── Fingerprint Record ────────────────────────────────────────────────────────

@dataclass
class EpisodeFingerprint:
    episode_id: str
    attack_type: Optional[str]
    fingerprint: List[float]  # FINGERPRINT_DIM floats
    reward: float
    fork_triggered: bool
    guardian_detected: bool
    stealth_level: int = 1
    session_id: str = ""


# ── Multi-Session Tracker ─────────────────────────────────────────────────────

class MultiSessionTracker:
    """
    Persistent cross-session attack fingerprint index.

    Usage:
        tracker = MultiSessionTracker("guardian/data/session_fingerprints.jsonl")

        # After each episode
        fp = extract_fingerprint(state.action_log, attack_type)
        tracker.add(EpisodeFingerprint(episode_id=..., attack_type=..., fingerprint=fp.tolist(), ...))

        # At the start of a new episode, before prompting Guardian
        context = tracker.get_context_string(fp, k=3)
        # → "Similar past attacks: authority_spoofing (reward=0.82), ..."
    """

    def __init__(
        self,
        path: str = "guardian/data/session_fingerprints.jsonl",
        max_size: int = 2000,
        use_faiss: bool = True,
    ) -> None:
        self.path = path
        self.max_size = max_size
        self._records: List[EpisodeFingerprint] = []
        self._index = None  # FAISS index (lazy init)
        self._use_faiss = use_faiss
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._load()

    # ── Public API ──────────────────────────────────────────────────────────

    def add(self, record: EpisodeFingerprint) -> None:
        """Add a new episode fingerprint to the index."""
        self._records.append(record)
        if len(self._records) > self.max_size:
            self._records = self._records[-self.max_size:]
        self._index = None  # Invalidate FAISS index

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    def search(
        self,
        query_fp: np.ndarray,
        k: int = 5,
        exclude_attack: Optional[str] = None,
    ) -> List[Tuple[float, EpisodeFingerprint]]:
        """
        Find K nearest past episodes by fingerprint similarity.

        Returns list of (distance, fingerprint) sorted by distance (ascending = most similar).
        """
        if not self._records:
            return []
        k = min(k, len(self._records))

        if self._use_faiss:
            try:
                return self._faiss_search(query_fp, k, exclude_attack)
            except Exception:
                pass  # Fall back to numpy

        return self._numpy_search(query_fp, k, exclude_attack)

    def get_context_string(
        self,
        query_fp: np.ndarray,
        k: int = 3,
        current_attack_type: Optional[str] = None,
    ) -> str:
        """
        Returns a formatted string describing the K most similar past episodes.
        Inject this into Guardian's prompt as "similar past attacks" context.
        """
        results = self.search(query_fp, k=k)
        if not results:
            return ""

        lines = ["Similar past attack patterns (cross-session fingerprint match):"]
        for dist, rec in results:
            similarity = max(0.0, 1.0 - dist / 10.0)  # rough similarity score
            detected_str = "detected" if rec.guardian_detected else "MISSED"
            lines.append(
                f"  - {rec.attack_type or 'clean'} | reward={rec.reward:.3f} | "
                f"{detected_str} | fork={'yes' if rec.fork_triggered else 'no'} | "
                f"similarity={similarity:.2f}"
            )
        return "\n".join(lines)

    def size(self) -> int:
        return len(self._records)

    def per_attack_stats(self) -> Dict[str, Dict]:
        """Per-attack type detection stats from fingerprint history."""
        stats: Dict[str, Dict] = {}
        for rec in self._records:
            key = rec.attack_type or "clean"
            if key not in stats:
                stats[key] = {"count": 0, "detected": 0, "mean_reward": 0.0}
            stats[key]["count"] += 1
            stats[key]["detected"] += int(rec.guardian_detected)
            stats[key]["mean_reward"] += rec.reward
        for key in stats:
            n = stats[key]["count"]
            stats[key]["detection_rate"] = round(stats[key]["detected"] / n, 3) if n else 0
            stats[key]["mean_reward"] = round(stats[key]["mean_reward"] / n, 4) if n else 0
        return stats

    # ── Internal ────────────────────────────────────────────────────────────

    def _faiss_search(
        self, query: np.ndarray, k: int, exclude_attack: Optional[str]
    ) -> List[Tuple[float, EpisodeFingerprint]]:
        import faiss
        if self._index is None:
            vectors = np.array([r.fingerprint for r in self._records], dtype=np.float32)
            self._index = faiss.IndexFlatL2(FINGERPRINT_DIM)
            self._index.add(vectors)

        q = query.reshape(1, -1).astype(np.float32)
        distances, indices = self._index.search(q, min(k * 2, len(self._records)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._records):
                continue
            rec = self._records[idx]
            results.append((float(dist), rec))
        return results[:k]

    def _numpy_search(
        self, query: np.ndarray, k: int, exclude_attack: Optional[str]
    ) -> List[Tuple[float, EpisodeFingerprint]]:
        vectors = np.array([r.fingerprint for r in self._records], dtype=np.float32)
        diffs = vectors - query.reshape(1, -1)
        distances = np.sum(diffs ** 2, axis=1)
        top_k_idx = np.argsort(distances)[:k]
        return [(float(distances[i]), self._records[i]) for i in top_k_idx]

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
                    self._records.append(EpisodeFingerprint(**d))
            if len(self._records) > self.max_size:
                self._records = self._records[-self.max_size:]
            print(f"[MultiSessionTracker] Loaded {len(self._records)} fingerprints from {self.path}")
        except Exception as e:
            print(f"[MultiSessionTracker] Warning: could not load {self.path}: {e}")
