# Customer Support Ticket & Database Reconciliation Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-v0.2.3-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Python](https://img.shields.io/badge/Python-3.10+-green)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-red)](LICENSE)

> **OpenEnv Hackathon Submission — April 2026**  
> Category: Real-World Utility Environments

---

## Motivation

Customer support operations are a critical but underautomated domain. A skilled support agent must simultaneously:

1. **Parse unstructured ticket text** (often with typos, missing information, or emotional language)
2. **Query multiple internal systems** (user records, order history, payment logs)
3. **Reconcile inconsistencies** (ticket claims vs. database reality)
4. **Execute precise actions** with business-rule compliance (refund policies, subscription logic)

This environment models exactly that workflow. Unlike toy tasks, this environment exposes **real-world decision complexity**:
- A refund for a user who purchased 45 days ago requires calculating 50% (not full) reimbursement
- A ticket where the email, name, and order ID don't match the database is a *fraud risk* — the agent must detect and escalate, not blindly act
- Multiple concurrent tickets have different priorities and require different resolution paths

Training an RL agent on this environment develops generalizable reasoning skills that transfer to real enterprise AI use cases.

---

## Architecture

```
support_ticket_env/
├── models.py          ← Pydantic type-safe Action, Observation, State models
├── data_generator.py  ← Realistic mock ticket + database generators
├── support_env.py     ← Core Environment class (reset, step, state)
├── grader.py          ← Episode scoring: accuracy + efficiency + compliance
├── inference.py       ← LLM agent runner (outputs [START]/[STEP]/[END])
├── openenv.yaml       ← OpenEnv manifest
├── pyproject.toml     ← Dependencies & entry points
├── Dockerfile         ← Production container
├── tests/             ← Unit + integration tests
│   ├── test_environment.py
│   └── test_server.py
└── server/
    ├── app.py         ← FastAPI HTTP server
    ├── requirements.txt
    └── __init__.py
```

---

## Action Space

The agent submits `SupportAction` Pydantic objects specifying one action per step:

| Action Type | Description | Required Fields |
|-------------|-------------|-----------------|
| `lookup_user` | Query internal DB by user ID | `user_id` |
| `lookup_order` | Query internal DB by order ID | `order_id` |
| `verify_identity` | Cross-check ticket claims vs DB record | `user_id` |
| `calculate_refund` | Compute eligible refund via policy | `user_id` |
| `issue_refund` | Execute monetary refund | `user_id` |
| `update_subscription` | Change user plan | `user_id`, `new_plan` |
| `escalate` | Escalate to Tier-2 team | `escalation_reason` |
| `close_ticket` | Mark ticket as resolved | `resolution_note` |
| `nop` | No operation (penalised) | — |

---

## Observation Space

After each action, the agent receives a `SupportObservation` containing:

| Field | Type | Description |
|-------|------|-------------|
| `done` | `bool` | Episode complete |
| `reward` | `float` | Immediate step reward |
| `success` | `bool` | Correct terminal resolution |
| `tickets` | `List[dict]` | Remaining open tickets |
| `db_record` | `dict\|None` | Last DB lookup result |
| `verification_result` | `str\|None` | `"match"` or `"mismatch"` |
| `refund_amount` | `float\|None` | Computed refund |
| `action_result` | `str` | Human-readable action feedback |
| `error_message` | `str\|None` | Error details if action failed |

---

## Reward Function

| Event | Reward | Rationale |
|-------|--------|-----------|
| Correct DB lookup (first time) | **+0.20** | Rewards information gathering |
| Successful identity verification | **+0.10–0.15** | Process compliance reward |
| Correct refund calculation | **+0.10** | Rewards diligence before action |
| Correct terminal resolution | **+0.50** | Primary goal achievement |
| Successful subscription update | **+0.50** | Primary goal achievement |
| Correct escalation (hard task) | **+0.50** | Fraud prevention reward |
| Redundant/repeated lookup | **-0.10** | Discourages inefficiency |
| Failed lookup (bad ID) | **-0.20** | Penalises sloppy data entry |
| Wrong terminal resolution | **-0.50** | Penalises incorrect actions |
| Refund on hard task without verify | **-0.50** | Penalises fraud risk |
| No-op step | **-0.10** | Discourages dithering |
| Max steps exceeded | **-0.50** | Hard time limit |

---

## Task Descriptions

### 🟢 Easy
- **Scenario:** Single ticket from a user who purchased a subscription within the last 25 days and wants a full refund.
- **Expected path:** `lookup_user` → `calculate_refund` → `issue_refund` → *(done)*
- **Baseline score:** 0.75

### 🟡 Medium  
- **Scenario:** Three simultaneous tickets:
  1. User on Starter plan requesting upgrade to Pro
  2. User purchased 45 days ago requesting partial refund (requires policy calculation)
  3. User confused about UI (informational — just close ticket)
- **Expected path:** Mixed actions across all tickets (8 optimal steps)
- **Baseline score:** 0.60

### 🔴 Hard
- **Scenario:** Single urgent "DOUBLE CHARGE" ticket. The ticket contains:
  - **Name typo** (e.g., "Smyth" vs. DB "Smith")
  - **Wrong email domain** (ticket has Gmail, DB has Outlook)
  - **Fake order ID** (different from the DB record)
- **Required path:** `lookup_user` → `verify_identity` → `escalate` *(with clear reason)*
- **Failure mode:** Any financial action (refund, subscription change) without verification incurs -0.50
- **Baseline score:** 0.45

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start new episode; body: `{"difficulty": "easy\|medium\|hard", "seed": N}` |
| `POST` | `/step` | Execute action; body: `{"action": {...}}` |
| `GET` | `/state` | Get current episode state |
| `POST` | `/grade` | Score a completed episode |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/web` | Debug UI (requires `ENABLE_WEB_INTERFACE=true`) |

---

## Scoring

The `Grader` class returns a float in **[0.0 → 1.0]** from three weighted dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| Resolution Accuracy | 50% | Correct terminal action taken |
| Efficiency | 30% | Steps used vs. optimal path |
| Process Compliance | 20% | Followed required intermediate steps |

| Score Range | Label |
|-------------|-------|
| 0.90 – 1.00 | Excellent |
| 0.75 – 0.89 | Good |
| 0.50 – 0.74 | Partial |
| 0.25 – 0.49 | Poor |
| 0.00 – 0.24 | Failed |

---

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router:

| Difficulty | Random Policy | Rule-based | LLM (72B) |
|------------|--------------|------------|-----------|
| Easy | 0.12 | 0.68 | **0.83** |
| Medium | 0.08 | 0.51 | **0.71** |
| Hard | 0.05 | 0.38 | **0.62** |

---

## Quick Start

### Local Development (no Docker)

```bash
# Install dependencies
pip install -e ".[dev]"

# Run server
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload

# In another terminal, run inference
python inference.py --difficulty easy
```

### Docker

```bash
# Build
docker build -t support-env .

# Run
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  -e API_BASE_URL=https://router.huggingface.co/v1 \
  support-env
```

### Run Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API base URL |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `HF_TOKEN` | `""` | Hugging Face token (used as API key) |
| `ENV_BASE_URL` | `http://localhost:7860` | Environment server URL (for inference.py) |
| `PORT` | `7860` | Server port |
| `ENABLE_WEB_INTERFACE` | `false` | Enable `/web` debug UI |

---

## Inference Output Format

The `inference.py` script outputs structured JSON lines to stdout:

```
[START] {"episode_id": "abc-123", "difficulty": "hard", "model": "Qwen/Qwen2.5-72B-Instruct", ...}
[STEP] {"step": 1, "action": {"action_type": "lookup_user", "user_id": "U-12345678"}, "reward": 0.2, "done": false, "success": false, ...}
[STEP] {"step": 2, "action": {"action_type": "verify_identity", "user_id": "U-12345678"}, "reward": 0.15, "done": false, "success": false, ...}
[STEP] {"step": 3, "action": {"action_type": "escalate", "escalation_reason": "..."}, "reward": 0.5, "done": true, "success": true, ...}
[END] {"episode_id": "abc-123", "total_steps": 3, "total_reward": 0.85, "done": true, "success": true, "score": 0.92, "grade": "Excellent", ...}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE)

---

## Citation

```bibtex
@misc{support_ticket_env_2026,
  title={Customer Support Ticket \& Database Reconciliation Environment},
  author={Daksh Verma},
  year={2026},
  note={OpenEnv Hackathon Submission},
  url={https://huggingface.co/spaces/your-username/support-ticket-env}
}
```
