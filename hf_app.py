"""
HuggingFace Space entry point for GUARDIAN SOC War Room.
Starts the FastAPI server on port 7860 and embeds it in a Gradio iframe.
Deploy this as a Gradio SDK Space.
"""
import subprocess
import sys
import os

# Start FastAPI server in background on port 7860
subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "server.app:app",
     "--host", "0.0.0.0", "--port", "7860"],
    env={**os.environ, "PORT": "7860"},
)

import time
time.sleep(3)  # Wait for FastAPI to start

import gradio as gr

with gr.Blocks(title="GUARDIAN SOC War Room") as demo:
    gr.HTML("""
    <div style="text-align:center; padding:10px 0; border-bottom:1px solid #1f2937; margin-bottom:10px;">
        <h2 style="color:#60a5fa; font-family:monospace;">GUARDIAN Fleet — AI Security Oversight</h2>
        <p style="color:#6b7280; font-size:0.85rem;">
            Real-time AI agent security via GRPO + Human-in-the-Loop
        </p>
    </div>
    <iframe
        src="http://localhost:7860/web"
        style="width:100%; height:850px; border:none; border-radius:8px;"
    ></iframe>
    """)

demo.launch(server_port=7860, share=False, show_api=False)
