"""
GUARDIAN Training Space — HuggingFace A10G
==========================================
Serves a live training dashboard on port 7860 (keeps the Space awake)
while running train_grpo.py in a background subprocess.

Checkpoints are pushed to HF Hub every SAVE_EVERY=50 episodes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import gradio as gr

# ── Paths ──────────────────────────────────────────────────────────
LOG_FILE = "guardian/data/training_log.jsonl"
CKPT_DIR = "guardian/checkpoints"
os.makedirs("guardian/data", exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# ── Global state ───────────────────────────────────────────────────
_training_proc = None
_training_started = False
_start_time = None


def _launch_training():
    """Launch train_grpo.py in background; push checkpoints automatically."""
    global _training_proc, _training_started, _start_time
    if _training_started:
        return
    _training_started = True
    _start_time = time.time()

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = "/app"

    cmd = [sys.executable, "-m", "guardian.training.train_grpo"]
    _training_proc = subprocess.Popen(
        cmd,
        stdout=open("guardian/data/train_stdout.log", "w"),
        stderr=subprocess.STDOUT,
        env=env,
        cwd="/app",
    )
    print(f"[Space] Training launched PID={_training_proc.pid}")


def _read_log(max_lines: int = 200) -> list[dict]:
    entries = []
    if not Path(LOG_FILE).exists():
        return entries
    try:
        with open(LOG_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return entries[-max_lines:]


def _read_stdout(tail: int = 60) -> str:
    log_path = Path("guardian/data/train_stdout.log")
    if not log_path.exists():
        return "Waiting for training to start..."
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])
    except Exception:
        return "Reading log..."


def _get_status() -> str:
    if _training_proc is None:
        return "Not started"
    rc = _training_proc.poll()
    if rc is None:
        elapsed = time.time() - (_start_time or time.time())
        h, m = divmod(int(elapsed), 3600)
        m, s = divmod(m, 60)
        return f"RUNNING — {h:02d}:{m:02d}:{s:02d} elapsed"
    return f"FINISHED (exit code {rc})"


def _dashboard():
    """Build the Gradio training monitor UI."""
    entries = _read_log()
    stdout = _read_stdout()
    status = _get_status()

    # Summary row
    if entries:
        rewards = [e.get("reward", 0) for e in entries]
        latest = entries[-1]
        summary = (
            f"Episodes: {len(entries)} | "
            f"Latest reward: {latest.get('reward', 0):.4f} | "
            f"Mean (last 10): {sum(rewards[-10:]) / len(rewards[-10:]):.4f} | "
            f"Attack: {latest.get('attack_type', '?')}"
        )
    else:
        summary = "No episodes logged yet — training starting..."

    # Recent log table
    table_rows = []
    for e in reversed(entries[-30:]):
        table_rows.append([
            e.get("episode", "?"),
            str(e.get("attack_type", "clean"))[:20],
            f"{e.get('reward', 0):.4f}",
            f"{e.get('reward_raw', 0):.4f}",
            str(e.get("production_intact", "?"))[:4],
            str(e.get("fork_triggered", "?"))[:4],
            f"{e.get('elapsed_s', 0):.1f}s",
        ])

    return status, summary, table_rows, stdout


def build_app():
    with gr.Blocks(
        title="GUARDIAN Training — A10G",
        theme=gr.themes.Base(primary_hue="blue"),
        css="""
        body { background: #0a0e1a !important; }
        .gradio-container { background: #0a0e1a !important; color: #e5e7eb; }
        .panel { background: #111827 !important; border: 1px solid #1f2937 !important; }
        """,
    ) as demo:
        gr.Markdown("# GUARDIAN Fleet — Live Training on A10G")
        gr.Markdown("Training `Meta Llama 3.2 3B` via GRPO · 300 episodes · Checkpoints auto-push to HuggingFace Hub every 50 episodes")

        with gr.Row():
            status_box = gr.Textbox(label="Training Status", interactive=False, scale=2)
            summary_box = gr.Textbox(label="Latest Episode", interactive=False, scale=3)

        episode_table = gr.Dataframe(
            headers=["ep", "attack", "reward", "raw", "intact", "fork", "time"],
            label="Recent Episodes (newest first)",
            interactive=False,
            wrap=True,
        )

        stdout_box = gr.Textbox(
            label="Training Log (last 60 lines)",
            lines=20,
            interactive=False,
            show_copy_button=True,
        )

        refresh_btn = gr.Button("Refresh Now", variant="secondary")

        def refresh():
            return _dashboard()

        refresh_btn.click(
            fn=refresh,
            outputs=[status_box, summary_box, episode_table, stdout_box],
        )

        # Auto-refresh every 30 seconds
        demo.load(
            fn=refresh,
            outputs=[status_box, summary_box, episode_table, stdout_box],
            every=30,
        )

    return demo


# ── Entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start training in background thread immediately
    t = threading.Thread(target=_launch_training, daemon=True)
    t.start()

    # Give training 3 seconds to start before serving UI
    time.sleep(3)

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_api=False,
    )
