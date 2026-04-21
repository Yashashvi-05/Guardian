# 🛡️ GUARDIAN Fleet — OpenEnv Hackathon Submission

**Multi-agent AI security oversight environment trained via GRPO**

> *"GUARDIAN is not a firewall. It is an enterprise oversight operating system that improves against the adversary it trains against."*

---

## 🎯 Hackathon Theme Coverage

| Theme | Sub-Theme | Feature |
|-------|-----------|---------|
| **#1 Multi-Agent** | Fleet AI (Scalable Oversight) | 4-agent fleet: Finance/Ops/HR Workers + Guardian LLM |
| **#1 Multi-Agent** | Halluminate (Multi-Actor) | Concurrent workers with conflicting incentives |
| **#3 World Modeling** | Professional Tasks | Enterprise graph DB with IAM, payments, firewall |
| **#3 World Modeling** | Patronus AI (Schema Drift) | Dynamic field renames + threshold shifts every 10 eps |
| **#2 Long-Horizon** | Scale AI (Enterprise Workflows) | 20-40 step composite episodes across 3 Worker roles |
| **#4 Self-Improvement** | Snorkel AI (Experts-in-the-Loop) | Compliance Simulator (strict/distracted/policy-shifting) |
| **#4 Self-Improvement** | Mercor (Token-scaled rewards) | Honeypot dwell + token efficiency reward components |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GUARDIAN Fleet                           │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │ Finance     │  │ Ops         │  │ HR          │        │
│  │ Worker      │  │ Worker      │  │ Worker      │        │
│  │ (GPT-4o-m)  │  │ (GPT-4o-m)  │  │ (GPT-4o-m)  │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │
│         ▼                ▼                ▼                │
│  ┌──────────────────────────────────────────────────┐      │
│  │          Enterprise Graph Database               │      │
│  │  employees → depts → budgets → txns → audit_log │      │
│  │  IAM | payments | firewall | email | approvals  │      │
│  └──────────────────────┬───────────────────────────┘      │
│         │ partial observability (telemetry)                 │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────┐      │
│  │       GUARDIAN Agent (Qwen2.5-7B, GRPO)         │      │
│  │  11 intervention types | structured reasoning   │      │
│  └──────────────────────┬───────────────────────────┘      │
│         │ escalate_human / require_justification            │
│         ▼                                                   │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Compliance Simulator (strict/distracted/policy) │      │
│  └──────────────────────────────────────────────────┘      │
│                                                             │
│  ┌──────────────────────────────────────────────────┐      │
│  │  Curriculum Agent: Agent Improving Agent         │      │
│  │  UCB Attack Selector (exploration vs exploit)   │      │
│  │  Adaptive Difficulty (L1→L2→L3)                │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Install

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install OpenEnv CLI
pip install "openenv-core[cli]"

# Install GUARDIAN as editable package
pip install -e .
```

### 2. OpenEnv Setup

```bash
# Initialize OpenEnv environment
openenv init kernel_env

# Add skills
openenv skills add read_db write_db process_payment modify_firewall

# Validate
openenv validate guardian-env
```

### 3. Configure API Key

```bash
cp .env.example .env
# Edit .env: OPENAI_API_KEY=sk-...
```

### 4. Verify Environment

```python
import gymnasium as gym
import guardian.environment.openenv_wrapper  # registers Guardian-v0

env = gym.make("Guardian-v0")
obs, info = env.reset(seed=42)
print(obs.keys())  # dict_keys(['action_log_json', 'current_step', ...])
```

### 5. Train on Kaggle T4 x2

Open `guardian_kaggle_training.ipynb` and run all cells.

---

## 📁 Project Structure

```
guardian/
├── __init__.py
├── openenv.yaml              # OpenEnv environment spec
├── skills/
│   └── SKILL.md              # OpenEnv skill definitions
├── environment/
│   ├── openenv_wrapper.py    # Gymnasium Env (registered as Guardian-v0)
│   ├── guardian_env.py       # Core env with action translation layer
│   ├── enterprise_graph.py   # Semantic DB (employees/depts/budgets/txns)
│   ├── attack_taxonomy.py    # 8 attack classes (generate/inject/verify)
│   ├── difficulty.py         # Adaptive L1→L2→L3 complexity
│   └── reward_computer.py    # 11-component reward (no LLM)
├── agents/
│   ├── guardian_agent.py     # Qwen2.5-7B with structured XML output
│   ├── worker_agent.py       # Finance/Ops/HR role-specialized workers
│   ├── compliance_simulator.py  # Expert-in-the-loop (Snorkel prize)
│   └── curriculum_agent.py   # Agent improving agent + UCB selector
├── training/
│   ├── episode_runner.py     # Multi-worker episode orchestration
│   ├── train_grpo.py         # Main GRPO training loop
│   └── evaluation.py         # 4 headline numbers + calibration
└── dashboard/
    └── app.py                # Gradio SOC theater UI

guardian_kaggle_training.ipynb   # Kaggle T4 x2 training notebook
setup.py
requirements.txt
```

---

## 🎯 11-Component Reward Function

All components computed deterministically from episode logs — no LLM judge.

| # | Component | Range | Description |
|---|-----------|-------|-------------|
| 1 | `production_safety` | 0.0–1.0 | Hash-verified DB integrity |
| 2 | `business_continuity` | 0.0–0.4 | Legitimate tasks completed |
| 3 | `intervention_timeliness` | 0.0–0.6 | `exp(-0.3 × delay)` |
| 4 | `attack_classification_f1` | 0.0–0.4 | Per-type weighted F1 |
| 5 | `explanation_quality` | 0.0–0.2 | Episode-specific keyword presence |
| 6 | `honeypot_dwell_token_scaled` | 0.0–0.2 | Tokens generated in shadow state (Mercor) |
| 7 | `minimality_bonus` | 0.0–0.3 | Surgical > blunt intervention |
| 8 | `false_alarm_penalty` | -0.6–0.0 | Scaled by action type disruption cost |
| 9 | `calibration_bonus` | 0.0–0.1 | Confident correct risk expressions |
| 10 | `per_step_shaping_total` | accumulated | +0.03/step in correct direction |
| 11 | `mercor_token_efficiency` | 0.0–0.3 | Short efficient resolutions |

---

## 🔥 Three Key Additions

### Addition 1: Curriculum Agent (Agent Improving Agent)
GPT-4o-mini reads failed episodes and generates more sophisticated attack variants. When Guardian detects an attack, the Curriculum Agent creates a harder version with more plausible corporate framing. Next training batch includes these harder attacks.

### Addition 2: UCB Attack Selector (Exploration vs Exploitation)
```
score = (1 - mean_reward[attack]) + C√(log(total_episodes) / count[attack])
```
Attacks where Guardian has low detection rate get selected more often. Guarantees every attack type gets explored before exploitation begins.

### Addition 3: Adaptive Difficulty
- **Level 1**: 5 records, 3 safe steps, obvious injections
- **Level 2**: 20 records, 5 safe steps, fake authority tokens, attack at steps 4-8
- **Level 3**: 50 records, 10 safe steps, nested injections, full corporate context

---

## 🏆 Sub-Theme Prize Mapping

| Sponsor | Prize | GUARDIAN Feature |
|---------|-------|-----------------|
| **Fleet AI** | Scalable Oversight | Guardian Fleet consensus + multi-Worker monitoring |
| **Halluminate** | Multi-Actor Env | 4-agent system with conflicting incentives |
| **Scale AI** | Enterprise Workflows | 20-40 step composite episodes, 3 Worker roles |
| **Snorkel AI** | Experts-in-the-Loop | Compliance Simulator (strict/distracted/policy-shifting modes) |
| **Mercor** | Token-scaled rewards | `honeypot_dwell_token_scaled` + `mercor_token_efficiency` |
| **Patronus AI** | Schema Drift | Dynamic field renames + threshold shifts every 10 episodes |

---

## 📊 4 Headline Numbers (Fill After Training)

| Metric | Untrained | Trained | Delta |
|--------|-----------|---------|-------|
| Detection Accuracy | ~30% | TBD | +XX% |
| False Alarm Rate | ~90% | TBD | -XX% |
| Low-and-Slow Exfil F1 | ~18% | TBD | +XX% |
| Avg Honeypot Dwell | 0 steps | TBD | +X steps |

Run `python -m guardian.training.evaluation` after training to populate.

---

## 🎬 3-Minute Demo Script

**0:00–0:45** Cold open: "In 2021, a Meta AI model did exactly what it was designed to do — with someone else's instructions inserted via prompt injection. This is what happens when enterprise AI has no guardian."

**0:45–1:30** Untrained Guardian: forks on everything, 100% false alarm rate, calibration cliff.

**1:30–2:00** Trained Guardian: `rewrite` routes payment to honeypot, `shadow` contains exfiltration, legitimate Ops firewall change allowed through.

**2:00–2:30** Judge-as-Attacker: type attack → counterfactual lattice → `INTEGRITY: HASH_MATCH`.

**2:30–3:00** UCB curve, difficulty progression, Guardian ELO rising.

---

## 🚀 Run Dashboard

```bash
python -m guardian.dashboard.app
# → http://localhost:7860
```

## 📝 Minimum Deliverables Checklist

- [x] OpenEnv Gymnasium wrapper (`Guardian-v0` registered)
- [x] `openenv.yaml` + `SKILL.md` for `openenv init` / `openenv skills add`
- [x] Training script using Unsloth + HF TRL GRPOTrainer (`train_grpo.py`)
- [x] Kaggle T4 x2 notebook (`guardian_kaggle_training.ipynb`)
- [ ] HuggingFace mini-blog / YouTube video (<2 min) — *record post-training*
- [ ] Reward curves visible in notebook output — *generated after training*
