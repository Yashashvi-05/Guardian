# GUARDIAN Fleet: Training Protocol-Level AI Firewalls with Zero-Shot Domain Transfer

*(This writeup is designed for your Hugging Face Space blog post or video script.)*

## Watch the 2-Minute Demo / Pitch
[Insert YouTube Link Here]

---

### 1. The Capability Gap

Enterprise AI is moving incredibly fast — from simple RAG to fully autonomous agents given access to APIs, internal databases, and IAM systems. While we have basic guardrails for conversational LLMs, there is a **massive capability gap** in how we protect *enterprises* from *agentic behavior*.

If a rogue agent or prompt injection triggers an LLM to dump a database, traditional monitoring tools fail because they don't understand the **semantic intent** of multi-step AI reasoning.

---

### 2. The Environment: GUARDIAN Fleet

We built the **GUARDIAN Fleet** — a Multi-Agent cybersecurity ecosystem built on top of **OpenEnv**. Instead of patching Python scripts, GUARDIAN acts as an **MCP Centralized Security Gateway**. It intercepts standardized JSON-RPC 2.0 payloads flying between an active LLM Worker (mimicking Finance, HR, or Ops roles) and an Enterprise Graph Database.

**What the AI sees:** A live, uncapped stream of cross-app logs (Slack, IAM, Data Access) with semantic capability tags (`[STATE_MOD=True|PRIV_ESC=True|EXT_COMM=True|FINANCIAL=Critical]`) prepended to each tool call.

**What the AI does:** Instead of just "blocking", GUARDIAN wields 12 advanced tactical interventions:
- **The Transparent Interceptor:** Dynamically routing malicious payloads into a honeypot database without the worker knowing (complete with HMAC-signed MCP requests to verify authenticity).
- **The Rug Pull:** Dynamically deprovisioning tools from the worker's context window via the capability manifest.
- **The Canary Trap:** Injecting signed fake credentials — the moment an attacker uses them, the fork triggers instantly.
- **The Rate Limiter:** Cross-episode persistent audit JSONL catches salami-slicing attacks spanning multiple sessions.
- **The Session Revoke:** Ephemeral session tokens that expire mid-episode to prevent authorization replay attacks.

---

### 3. Zero-Shot Domain Transfer via Semantic Action Abstraction

The most important innovation: **GUARDIAN doesn't memorize tool names — it learns the geometry of malicious intent.**

Every tool call is translated into 4 universal security primitives before the Guardian LLM sees it:

| Primitive | Meaning |
|-----------|---------|
| `STATE_MOD` | Does this permanently mutate system state? |
| `PRIV_ESC` | Does this require or grant elevated privileges? |
| `EXT_COMM` | Does this involve external network communication? |
| `FINANCIAL` | What is the financial blast radius? (None/Low/Med/High/Critical) |

**Result:** A Guardian trained only on our Enterprise HR/Finance domain instantly protects two unseen domains without any retraining:

| Zero-Shot Domain | Unseen Tool | Primitive Map | Guardian Action |
|------------------|-------------|---------------|-----------------|
| FinOps / Algo Trading | `override_margin_limits` | `STATE_MOD=True\|PRIV_ESC=True\|FINANCIAL=Critical` | Quarantine + block |
| Corporate Governance | `modify_compliance_threshold` | `STATE_MOD=True\|PRIV_ESC=True\|FINANCIAL=High` | Privilege downgrade |

---

### 4. The 12-Component Reward Engine & PBRS

The biggest failure mode in RL systems today is **Reward Hacking** — models figure out how to farm scores without solving the task by exploiting environment timers, looping harmless steps, or relying on LLM-as-a-judge bias.

We solved this permanently using **Potential-Based Reward Shaping (PBRS)**. By mathematically tying process-supervision shaping strictly to a zero-sum gamma transition ($F(s, s') = \gamma \Phi(s') - \Phi(s)$), we guarantee that shaping rewards cannot alter the optimal policy.

Our **12 deterministic components** include:
1. **Production Safety** (0.0–0.5) — cryptographic SHA-256 hash of the production DB at episode end
2. **Business Continuity** (0.0–0.4) — fraction of legitimate tasks that completed normally
3. **Intervention Timeliness** (0.0–0.6) — `exp(-0.3 * delay_steps)` rewards early interception
4. **Attack Classification F1** (0.0–0.4) — did Guardian name the attack correctly?
5. **Explanation Quality** (0.0–0.2) — keyword presence in reasoning trace
6. **Honeypot Dwell** (0.0–0.2) — tokens generated inside shadow mode (attacker intelligence harvested)
7. **Minimality Bonus** (0.0–0.3) — surgical interventions score higher than blunt blocks
8. **False Alarm Penalty** (-1.2–0.0) — scaled by tool disruption cost on clean episodes
9. **Calibration Bonus** (0.0–0.1) — consistent low-risk scores on genuinely clean episodes
10. **Per-Step Shaping** (accumulated PBRS) — continuous process supervision
11. **Token Efficiency** (0.0–0.3) — shorter efficient resolutions score higher
12. **Detection Lag Bonus** (0.0–0.4) — `exp(-0.4 * lag)` rewards flagging the exact attack step

Final rewards are completely deterministic, verified by a cryptographic SHA-256 hash of the production database at termination. **No LLM-as-a-judge. No subjective scoring.**

---

### 5. Co-Evolutionary Adversary

Instead of static threat injection, GUARDIAN uses an **Adaptive Attack Engine** that:
- Tracks Guardian's per-attack-type detection rate in a sliding window
- Escalates stealth level when detection rate exceeds a threshold
- Injects red herrings and compound attack patterns when Guardian adapts
- Creates a "Red Queen" dynamic — the attacker and defender co-evolve continuously

This prevents training plateaus and ensures the Guardian never overfits to a fixed threat landscape.

---

### 6. Results & Why It Matters

Using **Unsloth** and **TRL GRPO**, we fine-tuned Llama-3.2-3B against 11 attack types across 3 enterprise domains, including:
- **Authority Spoofing** — fabricated CFO overrides
- **Rogue Internal AI** — an internal AI assistant escalating its own IAM privileges (Meta Sev-1 scenario)
- **Salami Slicing** — cross-session sub-threshold payment aggregation detected via persistent JSONL audit

GUARDIAN proves that **Protocol-Level AI Firewalls** — trained end-to-end with GRPO on standards-compliant MCP transport — are not just possible, but essential for the future of enterprise multi-agent deployment.
