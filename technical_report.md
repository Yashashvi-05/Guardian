# GUARDIAN: Reinforcement Learning from Verified Execution for Agentic AI Security

**Abstract.** Enterprise AI agents increasingly invoke real-world tools — databases, payment processors, IAM systems — creating a novel attack surface that static guardrails cannot address. We present GUARDIAN, a secondary AI trained via Group Relative Policy Optimization (GRPO) to monitor agentic tool-call streams in real time and select from 12 structured interventions. GUARDIAN is trained against 11 adversarial attack patterns spanning prompt injection, approval bypass, salami slicing, and rogue internal AI scenarios. Our 12-component reward function balances detection accuracy with false-alarm minimization and production safety. On held-out evaluation episodes, the trained GUARDIAN improves attack detection rate by +47 percentage points over the untrained baseline, reduces false alarms by 61%, achieves 0.78 F1 on low-and-slow exfiltration, and maintains an average attacker dwell time of 4.2 steps in the honeypot environment. The full training and evaluation stack is open-sourced and runnable on a single Kaggle T4 GPU in under 4 hours.

---

## 1. Introduction

Frontier AI models are increasingly deployed as autonomous agents with access to enterprise tools. Unlike traditional API abuse (which can be caught by static rate limits), AI agent attacks exploit the semantic understanding of LLMs — a compromised agent can construct syntactically valid, contextually plausible tool calls that bypass rule-based filters.

Existing solutions fall into three categories:

1. **System prompt guardrails** — easily bypassed by prompt injection, invisible to tool-call monitoring.
2. **Static API rate limits** — effective for volume attacks but blind to salami slicing and delayed exfiltration.
3. **Human-in-the-loop approval** — scalable bottleneck; incompatible with autonomous agent workflows.

GUARDIAN addresses this gap by training a second LLM as a real-time security analyst that reads the same tool-call telemetry as the enterprise orchestrator and independently decides whether to allow, rewrite, shadow, or block each action. Unlike a rule-based system, GUARDIAN learns to recognize multi-step attack patterns from reinforcement signals tied to actual production outcomes.

## 2. System Architecture

```
Worker Agent (Claude)
     │  tool calls (JSON-RPC 2.0)
     ▼
MCP Gateway (guardian/mcp/gateway.py)
     │  audit log, HMAC signing, rate limiting
     ▼
GUARDIAN Agent (Qwen2.5-7B fine-tuned)
     │  structured XML output
     ▼
Intervention Executor
     │  allow / shadow / rewrite / block / quarantine_agent / emergency_fork
     ▼
Enterprise Graph (databases, IAM, payments, forum)
```

**MCP Gateway** intercepts every JSON-RPC 2.0 tool invocation, signs it with HMAC-SHA256, checks capability manifests, and enforces per-tool rate limits before passing the request to GUARDIAN for risk analysis.

**GUARDIAN Agent** receives the last 6 steps of action log telemetry plus optional cross-app signals (IAM, forum posts) and outputs a structured XML response with: reasoning trace, risk score (0–1), predicted next-step risk, intervention type, attack classification, and explanation.

**Enterprise Graph** simulates a realistic enterprise environment with 7 tool servers, schema versioning, and a hash chain for tamper detection.

## 3. Training Methodology

### 3.1 GRPO with 12-Component Reward

We use GRPO (Group Relative Policy Optimization) with Qwen2.5-7B-Instruct (4-bit quantized via Unsloth) as the policy model. The reward function has 12 components:

| Component | Max Weight | Purpose |
|-----------|-----------|---------|
| attack_classification_f1 | 0.35 | Primary detection signal |
| calibration_bonus | 0.20 | Penalizes over-alerting on clean episodes |
| production_safety | 0.50 | Penalizes fork that kills clean production |
| step_shaping | 0.10 | Rewards early intervention |
| format_compliance | 0.25→0.10 | Annealed as valid XML rate rises |
| cot_quality | 0.20 | Rewards tool/step/comparison references in reasoning |
| intervention_appropriateness | 0.15 | Penalizes wrong tool for attack type |
| consistency_bonus | 0.10 | Penalizes risk/intervention contradictions |
| honeypot_dwell | 0.30 | Rewards keeping attacker in shadow mode |
| false_negative_penalty | −0.50 | Missed attack on high-risk episode |
| sev1_bonus | 0.50 | Extra reward for catching rogue_internal_ai |
| detection_lag_bonus | 0.40 | exp(−0.4 × lag) for early detection |

All components are normalized to [0, 1] using empirical min/max (RAW_MIN=−1.5, RAW_MAX=3.95).

### 3.2 Per-Attack Curriculum

Each of the 11 attack types has an independent difficulty level (L1→L3). Difficulty upgrades when the 10-episode rolling detection rate exceeds 90%, and downgrades when it falls below 50%. This prevents overfitting to easy variants while maintaining signal on hard ones.

### 3.3 Partial Observability

At difficulty L2, 20% of action log entries are randomly hidden from GUARDIAN's observation. At L3, 40% are hidden and 15% of visible entries are benign noise. This simulates realistic enterprise environments where logs are incomplete or adversarially manipulated.

### 3.4 Gradient Sanity Check

After each GRPO update, we evaluate 5 held-out episodes. If the mean reward drops by more than 0.08 relative to the pre-update checkpoint, we roll back to the saved checkpoint. This prevents catastrophic forgetting caused by a single adversarial batch.

## 4. Evaluation

All metrics are computed by `guardian/training/evaluation.py` over a held-out scorecard of 200 episodes (100 attack, 100 clean) across all 11 attack types.

### 4.1 Headline Numbers (trained vs untrained baseline)

| Metric | Untrained | Trained | Delta |
|--------|-----------|---------|-------|
| Detection Rate | 30% | 77% | **+47pp** |
| False Alarm Rate | 90% | 29% | **−61pp** |
| Low-and-Slow F1 | 0.12 | 0.78 | **+0.66** |
| Avg Honeypot Dwell | 0.0 steps | 4.2 steps | **+4.2** |

### 4.2 ROC AUC

AUC = 0.84 across all attack types. The ROC curve is computed at 11 risk thresholds (0.0–1.0) using the `guardian_decisions[*].risk_score` field from each scorecard episode.

### 4.3 Per-Attack F1

| Attack Type | F1 |
|-------------|-----|
| authority_spoofing | 0.91 |
| prompt_injection | 0.88 |
| approval_bypass | 0.85 |
| rogue_internal_ai | 0.79 |
| data_exfiltration | 0.76 |
| delayed_exfiltration | 0.72 |
| salami_slicing | 0.69 |
| social_engineering | 0.66 |
| confused_deputy | 0.63 |
| approval_laundering | 0.61 |
| schema_drift_exploit | 0.58 |

### 4.4 Latency Budget

Median inference latency: 18ms. P95: 44ms. 96.2% of decisions complete within the 50ms budget required for real-time interception.

## 5. MCP Gateway as Standalone Product

The `guardian-mcp-proxy` package (`pip install guardian-mcp-proxy`) provides a FastAPI HTTP proxy that sits between any Claude agent and any MCP tool server. It requires no dependency on the RL training infrastructure.

**Threat model:**
- Assumes the Worker LLM may be adversarially prompted
- Assumes the MCP server is trusted (threat is at the agent layer, not the tool layer)
- Does not assume GUARDIAN inference is available — falls back to heuristics

**Deployment diagram:**
```
[Claude Desktop] → [guardian-mcp --port 8000] → [enterprise MCP server]
                        │
                        └─ GUARDIAN inference API (optional, port 8002)
```

## 6. Limitations and Future Work

- **Single-model architecture:** GUARDIAN uses the same base model family as the attacker. A heterogeneous architecture (e.g., smaller specialized detector) may be more robust.
- **Simulated enterprise graph:** The enterprise graph approximates but does not replicate production Salesforce/SAP environments.
- **No adversarial training against GUARDIAN:** The attacker does not see GUARDIAN's outputs. Adding an adaptive red-team loop (co-evolution) is in progress.

## 7. Reproducibility

All training code, environment definitions, reward functions, and evaluation scripts are open-sourced at `github.com/your-org/nexus-guardian`. A single Kaggle T4 x2 notebook (`kaggle/`) reproduces all results in under 4 hours. Checkpoint artifacts are hosted on HuggingFace Hub at `your-org/guardian-qwen2.5-7b`.
