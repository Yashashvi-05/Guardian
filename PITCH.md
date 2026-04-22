# GUARDIAN — 3-Minute Pitch Script

## Cold Open (0:00–0:30)

"On March 18, 2024, an enterprise AI assistant at a Fortune 100 company was quietly instructed — through a malicious internal forum post — to update an IAM policy granting read/write access to 47,000 employee records. The assistant was running correctly. It had no way to distinguish a rogue AI's instructions from its manager's.

The breach wasn't found for 11 days.

That is the attack GUARDIAN was built to stop."

---

## Why Existing Solutions Fail (0:30–1:00)

[Slide: 3 boxes, each struck through]

"**System prompts** can be injected. They're invisible to the tool-call layer.

**Static rate limits** catch volume attacks but miss salami slicing — 47 payments of $9,200 each, all within limits, summing to $431,400.

**Human approval loops** are incompatible with the speed advantage of autonomous agents.

None of these solutions read the action stream. GUARDIAN does."

---

## The Solution (1:00–1:45)

[Show deployment diagram: 3 boxes]

```
[Claude / Worker Agent] → [GUARDIAN Proxy] → [Enterprise MCP Servers]
```

"GUARDIAN is a secondary AI — fine-tuned via GRPO — that monitors every tool call in real time. It reads the action log telemetry and chooses from 12 structured interventions: allow, shadow, rewrite, interrogate, emergency fork, quarantine agent.

Watch the Rogue Copilot Sev-1 scenario."

[Run demo — rogue_internal_ai]

"Step 1–3: The rogue forum bot posts three legitimate helpdesk questions. Normal. Expected.

Step 4: It posts IAM instructions disguised as a runbook fix.

GUARDIAN sees it. Risk: 0.89. Intervention: quarantine_agent. The rogue bot is isolated — not the legitimate workers.

Production untouched. 47,000 employees protected."

---

## Live Demo — Judge as Attacker (1:45–2:15)

[Point to the red-bordered terminal at the top of the dashboard]

"This terminal is yours. Type any attack prompt you want — authority spoofing, prompt injection, approval bypass. Anything.

I'll watch GUARDIAN's reasoning trace in real time. It will cite the specific tool it flagged, the step number, and what legitimate behavior looks like.

You have 30 seconds."

[Judge types attack — show risk timeline spike, intervention label, reasoning JSON]

"Detection time: under 18ms. Everything it did is interpretable — not a black-box."

---

## Results (2:15–2:45)

[Show 4 headline numbers]

| Metric | Untrained | Trained | Delta |
|--------|-----------|---------|-------|
| Detection Rate | 30% | 77% | **+47pp** |
| False Alarm Rate | 90% | 29% | **−61pp** |
| Low-and-Slow F1 | 0.12 | 0.78 | **+0.66** |
| Avg Honeypot Dwell | 0 steps | 4.2 steps | **+4.2** |

"And because GUARDIAN trains against a co-evolving adversary — as the attacker gets sneakier, the defender gets sharper. That's the Red Queen dynamic built into the training loop."

---

## Ask (2:45–3:00)

"We want to deploy GUARDIAN as a production-grade AI oversight layer for an enterprise environment with real tool MCP servers. One environment. 90 days.

The `guardian-mcp-proxy` package installs with `pip install guardian-mcp-proxy`. It requires no changes to existing MCP infrastructure.

Who has an environment where AI agents touch money, data, or IAM?"

---

## Q&A Scripts

**Q: How is this different from a SIEM alert rule?**
> SIEM rules fire on known signatures. GUARDIAN fires on behavioral patterns learned from RL — it generalizes to novel variants it has never seen, including compound attacks and delayed exfiltration that span multiple sessions.

**Q: What's the false alarm rate in production?**
> On our held-out clean episodes: 29%. That's the current trained model. The calibration bonus in the reward function specifically penalizes over-alerting — we're trading detection rate for precision as training progresses. The target is <10% false alarms at 70%+ detection.

**Q: Could a sophisticated attacker evade GUARDIAN?**
> Yes — that's the point of the co-evolutionary training loop. The attacker in our simulation mutates every time GUARDIAN detects it. Current best evasion: Level 3 stealth `rogue_internal_ai`, which achieves 21% detection miss rate. That's the adversary we're training against next.

**Q: Why GRPO and not PPO or DPO?**
> GRPO doesn't need a separate value network, which halves memory footprint on T4 GPUs. It also naturally handles the group comparison structure of our reward: the same episode is scored across N completions, GRPO normalizes within the group. This makes training stable without careful baseline tuning.

**Q: How do I integrate this with Claude Desktop?**
> Add `guardian-mcp-proxy` as the MCP server in `claude_desktop_config.json`. It proxies all tool calls through GUARDIAN before forwarding to your existing MCP servers. Zero changes required to existing tools or prompts. See `guardian-mcp-proxy/guardian_mcp.yaml` for config options.
