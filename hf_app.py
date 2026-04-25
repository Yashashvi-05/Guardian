"""
HuggingFace Space entry point — GUARDIAN SOC War Room (Demo Space).

NOTE: This file is used by the demo SOC War Room Space (sdk: docker).
The training space uses training_space.py as the Docker CMD instead.

This script starts the FastAPI server directly — no Gradio wrapper needed
because the SOC War Room UI is already embedded in server/app.py at /web.
"""
import os
import sys
import subprocess
import time

PORT = int(os.getenv("PORT", 7860))

# Start FastAPI server — SOC War Room available at /web
proc = subprocess.Popen(
    [
        sys.executable, "-m", "uvicorn", "server.app:app",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--log-level", "info",
    ],
    env={**os.environ, "PYTHONPATH": "/app"},
)

print(f"[hf_app] FastAPI server started PID={proc.pid} on port {PORT}")
print(f"[hf_app] SOC War Room: http://0.0.0.0:{PORT}/web")

# Keep process alive — exit when FastAPI exits
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    proc.wait()
