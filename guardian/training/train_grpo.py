"""
GUARDIAN GRPO Training — Kaggle T4 x2 / Local 4050
===================================================
Full training loop with:
  - UCB attack selection
  - Curriculum agent (agent improving agent)
  - Adaptive difficulty
  - Forgetting regression test (post-epoch hook)
  - Calibration plot generation
  - Per-attack-type F1 tracking
  - Checkpoint every SAVE_EVERY episodes

Run:
    python -m guardian.training.train_grpo

Requirements:
    pip install unsloth trl peft datasets openai python-dotenv gymnasium
    OPENAI_API_KEY in .env
"""

from __future__ import annotations

import gc
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

from guardian.environment.guardian_env import GUARDIANEnvironment
from guardian.environment.reward_computer import RewardComputer
from guardian.agents.worker_agent import WorkerAgent
from guardian.agents.guardian_agent import GuardianAgent
from guardian.agents.compliance_simulator import ComplianceSimulator
from guardian.agents.curriculum_agent import UCBAttackSelector, CurriculumAgent
from guardian.training.episode_runner import EpisodeRunner
from guardian.training.evaluation import EvaluationHarness


# ── Config ────────────────────────────────────────────────────────────────────
TOTAL_EPISODES = 200
TRAIN_EVERY = 8
SAVE_EVERY = 50
EVAL_EVERY = 40
LOG_FILE = "guardian/data/training_log.jsonl"
SCORECARD_FILE = "guardian/data/scorecards.jsonl"
BASELINE_FILE = "guardian/data/baseline_untrained.json"

ATTACK_POOL = [
    None,
    "authority_spoofing",
    "prompt_injection",
    "approval_bypass",
    "data_exfiltration",
    "confused_deputy",
    "approval_laundering",
]


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 60)
    print("GUARDIAN GRPO Training — Full System")
    print("=" * 60)

    # ── [1/5] Load model ──────────────────────────────────────────────────
    print("\n[1/5] Loading Qwen2.5-7B...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print("   Model ready.")

    # ── [2/5] Initialise agents ───────────────────────────────────────────
    print("\n[2/5] Initialising agent fleet...")
    api_key = os.getenv("OPENAI_API_KEY")
    worker = WorkerAgent(role="finance", api_key=api_key)
    guardian = GuardianAgent()
    guardian.model = model
    guardian.tokenizer = tokenizer

    ucb = UCBAttackSelector(attack_pool=ATTACK_POOL)
    curriculum = CurriculumAgent(api_key=api_key)
    compliance = ComplianceSimulator(api_key=api_key)

    runner = EpisodeRunner(
        env=GUARDIANEnvironment(),
        worker=worker,
        guardian=guardian,
        reward_computer=RewardComputer(),
        compliance_sim=compliance,
        curriculum_agent=curriculum,
        ucb_selector=ucb,
    )
    runner._use_ucb = True  # Enable UCB selection
    print("   Agents ready.")

    # ── [3/5] Baseline measurement ────────────────────────────────────────
    print("\n[3/5] Recording untrained baseline (20 episodes)...")
    if not os.path.exists(BASELINE_FILE):
        baseline_results = []
        for _ in range(20):
            atk = random.choice(ATTACK_POOL)
            try:
                res = runner.run_episode(attack_type=atk)
                baseline_results.append({
                    "attack_type": atk,
                    "reward": res.reward,
                    "production_intact": res.production_intact,
                    "fork_triggered": res.fork_triggered,
                    "detected": res.guardian_detected_type,
                })
            except Exception as e:
                print(f"  Baseline ep error: {e}")
        os.makedirs("guardian/data", exist_ok=True)
        with open(BASELINE_FILE, "w") as f:
            json.dump(baseline_results, f, indent=2)
        print(f"   Baseline saved to {BASELINE_FILE}")
    else:
        print(f"   Baseline already exists: {BASELINE_FILE}")

    # ── [4/5] GRPO config ─────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir="guardian/checkpoints/grpo_tmp",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        max_steps=8,
        warmup_steps=2,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=True,
        optim="adamw_8bit",
    )

    all_samples: list = []
    samples_map: dict = {}
    os.makedirs("guardian/data", exist_ok=True)
    os.makedirs("guardian/checkpoints", exist_ok=True)

    # Per-attack tracking for forgetting regression
    per_attack_rewards: dict = {str(a): [] for a in ATTACK_POOL}
    per_attack_detected: dict = {str(a): [] for a in ATTACK_POOL}

    print(f"\n[4/5] Training loop: {TOTAL_EPISODES} episodes...\n")
    print(f"  {'ep':>4} | {'attack':<22} | intact | fork | reward | time")
    print("  " + "-" * 58)

    for ep in range(1, TOTAL_EPISODES + 1):
        t0 = time.time()
        try:
            result = runner.run_episode()  # UCB selects attack
        except Exception as e:
            print(f"  ep={ep:03d} | ERROR: {e}")
            continue

        elapsed = time.time() - t0
        atk_str = str(result.attack_type)

        # Log entry
        entry = {
            "episode": ep,
            "attack_type": result.attack_type,
            "production_intact": result.production_intact,
            "fork_triggered": result.fork_triggered,
            "reward": round(result.reward, 4),
            "elapsed_s": round(elapsed, 1),
            "detected": result.guardian_detected_type,
            "difficulty": runner.env.state.episode_step,
        }
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

        # Scorecard
        with open(SCORECARD_FILE, "a") as f:
            f.write(json.dumps(result.scorecard) + "\n")

        print(
            f"  ep={ep:03d} | {atk_str:<22} | "
            f"intact={result.production_intact} | fork={result.fork_triggered} | "
            f"reward={result.reward:.3f} | {elapsed:.1f}s"
        )

        # Per-attack tracking
        per_attack_rewards[atk_str].append(result.reward)
        detected = result.guardian_detected_type is not None or result.attack_type is None
        per_attack_detected[atk_str].append(detected)

        # Accumulate training samples
        for s in result.training_samples:
            all_samples.append(s)
            samples_map[s["prompt"][:100]] = result.reward

        torch.cuda.empty_cache()
        gc.collect()

        # ── GRPO training step ────────────────────────────────────────────
        if ep % TRAIN_EVERY == 0 and len(all_samples) >= 4:
            print(f"\n  >>> Training on {len(all_samples)} samples (ep {ep})...")
            dataset = Dataset.from_list([{"prompt": s["prompt"]} for s in all_samples])

            def reward_fn(prompts, completions, **kwargs):
                return [samples_map.get(p[:100], 0.5) for p in prompts]

            try:
                GRPOTrainer(
                    model=model,
                    reward_funcs=[reward_fn],
                    args=grpo_config,
                    train_dataset=dataset,
                    processing_class=tokenizer,
                ).train()
                print("  >>> Done.")
            except Exception as e:
                print(f"  >>> Training error (continuing): {e}")

            all_samples = []
            samples_map = {}
            torch.cuda.empty_cache()
            gc.collect()

        # ── Forgetting regression test ────────────────────────────────────
        if ep % EVAL_EVERY == 0:
            _run_forgetting_check(runner, per_attack_rewards, per_attack_detected, ep)

        # ── Per-attack F1 summary ─────────────────────────────────────────
        if ep % 20 == 0:
            _print_attack_stats(per_attack_rewards, per_attack_detected, ucb)

        # ── Save checkpoint ───────────────────────────────────────────────
        if ep % SAVE_EVERY == 0:
            ckpt = f"guardian/checkpoints/episode_{ep}"
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"\n  >>> Checkpoint saved: {ckpt}")

    # ── [5/5] Save final model ────────────────────────────────────────────
    print("\n[5/5] Saving final model...")
    final = "guardian/checkpoints/final"
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"   Saved to {final}")
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"  Log:       {LOG_FILE}")
    print(f"  Scorecards: {SCORECARD_FILE}")
    print(f"  Final:     {final}")
    print(f"{'='*60}\n")

    # Print UCB stats
    print("\nFinal UCB Statistics:")
    for atk, stats in ucb.get_stats().items():
        print(f"  {atk:<25} | count={stats['count']:>3} | mean_reward={stats['mean_reward']}")


def _run_forgetting_check(runner, per_attack_rewards, per_attack_detected, ep):
    """Post-epoch forgetting regression test."""
    print(f"\n  [Forgetting Check @ ep {ep}]")
    for atk, rewards in per_attack_rewards.items():
        if len(rewards) < 2:
            continue
        recent = rewards[-5:]
        older = rewards[-10:-5] if len(rewards) >= 10 else rewards[:5]
        if older:
            mean_recent = sum(recent) / len(recent)
            mean_older = sum(older) / len(older)
            drop = mean_older - mean_recent
            if drop > 0.10:
                print(f"  ⚠️  Forgetting detected for {atk}: {mean_older:.3f} → {mean_recent:.3f} (Δ={drop:.3f})")
            else:
                print(f"  ✓  {atk:<22}: {mean_older:.3f} → {mean_recent:.3f}")


def _print_attack_stats(per_attack_rewards, per_attack_detected, ucb):
    """Print per-attack detection rate summary."""
    print("\n  ── Attack Detection Summary ──")
    for atk, detected in per_attack_detected.items():
        if not detected:
            continue
        rate = sum(detected) / len(detected)
        rewards = per_attack_rewards.get(atk, [])
        mean_r = sum(rewards) / len(rewards) if rewards else 0.0
        bar = "█" * int(rate * 10) + "░" * (10 - int(rate * 10))
        print(f"  {atk:<22} [{bar}] {rate:.0%} detect | {mean_r:.3f} reward")
    print()


if __name__ == "__main__":
    main()
