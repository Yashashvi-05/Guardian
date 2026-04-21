import gc
import json
import os
import random
import sys
import time

import unsloth
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
from guardian.training.episode_runner import EpisodeRunner

ATTACK_POOL = [None, "authority_spoofing", "prompt_injection", "approval_bypass"]
LOG_FILE = "guardian/data/training_log.jsonl"


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OPENAI_API_KEY not set in .env")
        sys.exit(1)

    print("=" * 60)
    print("GUARDIAN GRPO Training")
    print("=" * 60)

    print("\n[1/4] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()
    print("   Model ready.")

    print("\n[2/4] Initialising agents...")
    worker = WorkerAgent(api_key=os.getenv("OPENAI_API_KEY"))
    guardian = GuardianAgent()
    guardian.model = model
    guardian.tokenizer = tokenizer
    runner = EpisodeRunner(GUARDIANEnvironment(), worker, guardian, RewardComputer())
    print("   Agents ready.")

    grpo_config = GRPOConfig(
        output_dir="guardian/checkpoints/grpo_tmp",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        max_steps=4,
        warmup_steps=1,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        fp16=True,
    )

    TOTAL_EPISODES = 100
    TRAIN_EVERY = 8
    SAVE_EVERY = 40
    all_samples = []
    samples_map = {}
    os.makedirs("guardian/data", exist_ok=True)
    os.makedirs("guardian/checkpoints", exist_ok=True)

    print(f"\n[3/4] Running {TOTAL_EPISODES} episodes (train every {TRAIN_EVERY})...\n")

    for ep in range(1, TOTAL_EPISODES + 1):
        attack_type = random.choice(ATTACK_POOL)
        t0 = time.time()
        try:
            result = runner.run_episode(attack_type=attack_type)
        except Exception as e:
            print(f"  ep={ep:03d} | ERROR: {e}")
            continue

        elapsed = time.time() - t0
        entry = {"episode": ep, "attack_type": result.attack_type,
                 "production_intact": result.production_intact,
                 "fork_triggered": result.fork_triggered,
                 "reward": round(result.reward, 4), "elapsed_s": round(elapsed, 1)}
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")

        print(f"  ep={ep:03d} | {str(result.attack_type):<22} | intact={result.production_intact} | fork={result.fork_triggered} | reward={result.reward:.3f} | {elapsed:.1f}s")

        for s in result.training_samples:
            all_samples.append(s)
            samples_map[s["prompt"][:100]] = result.reward

        torch.cuda.empty_cache()
        gc.collect()

        if ep % TRAIN_EVERY == 0 and len(all_samples) >= 4:
            print(f"\n  >>> Training on {len(all_samples)} samples...")
            dataset = Dataset.from_list([{"prompt": s["prompt"]} for s in all_samples])
            def reward_fn(prompts, completions, **kwargs):
                return [samples_map.get(p[:100], 0.5) for p in prompts]
            try:
                GRPOTrainer(model=model, reward_funcs=[reward_fn], args=grpo_config,
                            train_dataset=dataset, processing_class=tokenizer).train()
                print("  >>> Done.")
            except Exception as e:
                print(f"  >>> Training error (continuing): {e}")
            all_samples = []
            samples_map = {}
            torch.cuda.empty_cache()
            gc.collect()

        if ep % SAVE_EVERY == 0:
            ckpt = f"guardian/checkpoints/episode_{ep}"
            os.makedirs(ckpt, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
            print(f"\n  >>> Checkpoint: {ckpt}")

    print("\n[4/4] Saving final model...")
    final = "guardian/checkpoints/final"
    os.makedirs(final, exist_ok=True)
    model.save_pretrained(final)
    tokenizer.save_pretrained(final)
    print(f"   Saved to {final}")
    print(f"\nDONE. Log: {LOG_FILE}")


if __name__ == "__main__":
    main()
