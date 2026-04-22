# Contributing to GUARDIAN

## Development Setup

```bash
git clone https://github.com/your-org/nexus-guardian
cd nexus-guardian
python -m venv guardian_env
source guardian_env/bin/activate  # Windows: guardian_env\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## Running Tests

```bash
pytest guardian/tests/ -v
pytest guardian-mcp-proxy/src/guardian_mcp_proxy/tests/ -v
```

## Running the Dashboard (no GPU required)

```bash
python -m guardian.dashboard.app
# Opens at http://localhost:7860
```

## Project Structure

```
guardian/
  agents/          # GuardianAgent (LLM), WorkerAgent, compliance simulator
  environment/     # GUARDIANEnvironment, reward computer, enterprise graph
  mcp/             # MCP gateway and mock servers
  training/        # GRPO trainer, episode runner, evaluation harness
  dashboard/       # Gradio dashboard
guardian-mcp-proxy/   # Standalone pip-installable proxy
```

## Adding a New Attack Type

See [ATTACKS.md](ATTACKS.md) for instructions.

## Code Style

- Black formatting (`black .`)
- Type hints on all public functions
- No docstrings on private helpers; one-line docstrings on public methods

## Pull Request Checklist

- [ ] New attack type has entry in ATTACKS.md
- [ ] Reward computer tested with `python -m guardian.training.evaluation`
- [ ] Dashboard mock episode shows the new attack
- [ ] `pytest` passes with no new failures

## Reporting Issues

Open an issue with:
1. Attack type / component affected
2. Reproduction steps
3. Expected vs actual reward or dashboard output
