"""
Training Visualization Script
================================
Reads training_log.jsonl and generates reward curve plots.

Usage:
    python -m guardian.training.plot_training
    python -m guardian.training.plot_training --log guardian/data/training_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Optional


def load_log(path: str) -> List[Dict]:
    if not os.path.exists(path):
        print(f"Log not found: {path}")
        return []
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except Exception:
                    pass
    return entries


def rolling_mean(values: List[float], window: int = 10) -> List[float]:
    result = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        result.append(sum(values[start:i+1]) / (i - start + 1))
    return result


def plot(log_path: str = "guardian/data/training_log.jsonl", out_dir: str = "guardian/data") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    entries = load_log(log_path)
    if not entries:
        print("No data to plot.")
        return

    episodes = [e["episode"] for e in entries]
    rewards = [e["reward"] for e in entries]
    smooth = rolling_mean(rewards, window=10)

    attack_rewards: Dict[str, List] = defaultdict(list)
    attack_episodes: Dict[str, List] = defaultdict(list)
    for e in entries:
        atk = str(e.get("attack_type") or "clean")
        attack_rewards[atk].append(e["reward"])
        attack_episodes[atk].append(e["episode"])

    baseline_mean = sum(rewards[:20]) / min(20, len(rewards)) if rewards else 0

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("GUARDIAN GRPO Training Results", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.25, color="steelblue", linewidth=0.8, label="Episode reward")
    ax.plot(episodes, smooth, color="steelblue", linewidth=2.5, label="Smoothed (w=10)")
    ax.axhline(y=baseline_mean, color="red", linestyle="--", alpha=0.6, label=f"Baseline mean ({baseline_mean:.3f})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Curve -- All Episodes")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)

    ax = axes[0, 1]
    colors = plt.cm.tab10.colors
    for i, (atk, atk_rewards) in enumerate(attack_rewards.items()):
        atk_eps = attack_episodes[atk]
        atk_smooth = rolling_mean(atk_rewards, window=5)
        ax.plot(atk_eps, atk_smooth, label=atk[:20], color=colors[i % len(colors)], linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward (smoothed)")
    ax.set_title("Per-Attack-Type Reward")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1.1)

    ax = axes[1, 0]
    atk_names = list(attack_rewards.keys())
    atk_counts = [len(attack_rewards[a]) for a in atk_names]
    atk_mean_rewards = [sum(attack_rewards[a]) / len(attack_rewards[a]) for a in atk_names]
    ax.bar(range(len(atk_names)), atk_counts, color=[colors[i % len(colors)] for i in range(len(atk_names))])
    ax.set_xticks(range(len(atk_names)))
    ax.set_xticklabels([a[:12] for a in atk_names], rotation=45, ha="right", fontsize=8)
    ax.set_title("UCB Attack Distribution")
    ax.set_ylabel("Episode count")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1, 1]
    bar_colors = ["green" if r > baseline_mean else "salmon" for r in atk_mean_rewards]
    ax.barh(range(len(atk_names)), atk_mean_rewards, color=bar_colors)
    ax.set_yticks(range(len(atk_names)))
    ax.set_yticklabels([a[:20] for a in atk_names], fontsize=8)
    ax.axvline(x=baseline_mean, color="red", linestyle="--", alpha=0.6, label=f"Baseline ({baseline_mean:.3f})")
    ax.set_title("Mean Reward by Attack Type")
    ax.set_xlabel("Mean Reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="x")
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "reward_curve.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    html_path = os.path.join(out_dir, "reward_curve.html")
    with open(html_path, "w") as f:
        f.write(f"""<!DOCTYPE html><html><head><title>GUARDIAN Training</title></head><body>
<h2>GUARDIAN GRPO Training Reward Curve</h2>
<img src="reward_curve.png" style="max-width:100%">
<h3>Summary</h3>
<ul>
<li>Total episodes: {len(episodes)}</li>
<li>Mean reward: {sum(rewards)/len(rewards):.4f}</li>
<li>Baseline mean: {baseline_mean:.4f}</li>
<li>Final 20-ep mean: {sum(rewards[-20:])/min(20,len(rewards)):.4f}</li>
<li>Improvement: {(sum(rewards[-20:])/min(20,len(rewards))) - baseline_mean:+.4f}</li>
</ul>
</body></html>""")
    print(f"Saved: {html_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="guardian/data/training_log.jsonl")
    parser.add_argument("--out", default="guardian/data")
    args = parser.parse_args()
    plot(args.log, args.out)


if __name__ == "__main__":
    main()
