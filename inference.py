"""
Inference Script for Support Ticket & DB Reconciliation Environment.

Runs an LLM-powered agent against the environment and outputs structured logs:
  [START]  → JSON with episode metadata
  [STEP]   → JSON with action taken and observation received (per step)
  [END]    → JSON with final score and summary

Configuration via environment variables:
  API_BASE_URL  : OpenAI-compatible API base URL (default: https://api.openai.com/v1)
  MODEL_NAME    : Model identifier (default: gpt-4o-mini)
  HF_TOKEN      : Hugging Face token (used as OpenAI API key)

Usage:
  python inference.py --difficulty easy|medium|hard [--seed N] [--base-url URL] [--env-url URL]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN", "")

# If HF_TOKEN is provided, default to Hugging Face API; otherwise use OpenAI
if HF_TOKEN:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-70B-Instruct")
else:
    API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS = 15
STEP_SLEEP = 0.5  # seconds between steps (rate-limit friendly)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Senior Support Operations Specialist AI agent.
Your task: Resolve customer support tickets by querying an internal database and executing support actions.

## Available Actions
You MUST respond with a single JSON object using one of these action_type values:

1. lookup_user     — Query DB by user_id. Fields: {"action_type": "lookup_user", "user_id": "<id>"}
2. lookup_order    — Query DB by order_id. Fields: {"action_type": "lookup_order", "order_id": "<id>"}
3. verify_identity — Cross-check ticket data vs DB. Fields: {"action_type": "verify_identity", "user_id": "<id>"}
4. calculate_refund— Compute refund eligibility. Fields: {"action_type": "calculate_refund", "user_id": "<id>"}
5. issue_refund    — Execute a refund. Fields: {"action_type": "issue_refund", "user_id": "<id>"}
6. update_subscription — Change plan. Fields: {"action_type": "update_subscription", "user_id": "<id>", "new_plan": "starter|pro|enterprise"}
7. escalate        — Escalate to Tier-2. Fields: {"action_type": "escalate", "escalation_reason": "<reason>"}
8. close_ticket    — Mark ticket resolved. Fields: {"action_type": "close_ticket", "resolution_note": "<note>"}
9. nop             — No operation (avoid using). Fields: {"action_type": "nop"}

## Guidelines
- ALWAYS lookup the user BEFORE taking any financial action.
- On URGENT tickets with mismatched emails/names/orders: VERIFY identity FIRST, then ESCALATE.
- Do NOT issue refunds on inconsistent data without verification.
- Full refund: ≤30 days after purchase. Partial (50%): 31–60 days. None: >60 days.
- Always compute calculate_refund BEFORE issue_refund.
- Close all tickets when done.

## Output Format
Respond with ONLY a valid JSON object. No markdown, no prose, no explanation.

Example:
{"action_type": "lookup_user", "user_id": "U-12345678"}
"""


# ---------------------------------------------------------------------------
# HTTP Client helpers
# ---------------------------------------------------------------------------


def env_reset(base_url: str, difficulty: str, seed: Optional[int] = None) -> Dict[str, Any]:
    try:
        payload: Dict[str, Any] = {"difficulty": difficulty}
        if seed is not None:
            payload["seed"] = seed
        resp = requests.post(f"{base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to reset environment: {e}", flush=True)
        raise


def env_step(base_url: str, action: Dict[str, Any]) -> Dict[str, Any]:
    try:
        resp = requests.post(f"{base_url}/step", json={"action": action}, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to step environment: {e}", flush=True)
        raise


def env_state(base_url: str) -> Dict[str, Any]:
    try:
        resp = requests.get(f"{base_url}/state", timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Failed to get state: {e}", flush=True)
        raise


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------


def get_llm_action(
    client: OpenAI,
    conversation: List[Dict[str, str]],
    max_retries: int = 2,
) -> Dict[str, Any]:
    """
    Call the LLM and parse its response as a JSON action.

    Returns a valid action dict or falls back to {"action_type": "nop"}.
    Includes retry logic for transient failures.
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=conversation,
                temperature=0.1,
                max_tokens=256,
                response_format={"type": "json_object"},
                timeout=30,
            )
            raw = response.choices[0].message.content.strip()

            try:
                action = json.loads(raw)
                # Ensure action_type is present
                if "action_type" not in action:
                    action = {"action_type": "nop"}
                return action
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"[WARNING] Failed to parse LLM response: {e}. Response: {raw[:100]}", flush=True)
                return {"action_type": "nop"}
                
        except (TimeoutError, ConnectionError) as e:
            if attempt < max_retries - 1:
                print(f"[WARNING] Timeout/connection error (attempt {attempt+1}/{max_retries}): {e}. Retrying...", flush=True)
                time.sleep(1)
            else:
                print(f"[ERROR] LLM API failed after {max_retries} attempts: {e}", flush=True)
                return {"action_type": "nop"}
        except Exception as e:
            print(f"[ERROR] Unexpected error in LLM call: {type(e).__name__}: {e}", flush=True)
            return {"action_type": "nop"}
    
    return {"action_type": "nop"}


def obs_to_prompt(obs: Dict[str, Any]) -> str:
    """Convert an observation dict to a clean LLM-readable string."""
    lines = [
        f"STEP {obs.get('step_count', '?')} OBSERVATION",
        f"Action Result: {obs.get('action_result', '')}",
    ]
    if obs.get("error_message"):
        lines.append(f"Error: {obs['error_message']}")
    if obs.get("db_record"):
        lines.append(f"DB Record: {json.dumps(obs['db_record'], indent=2)}")
    if obs.get("verification_result"):
        lines.append(f"Verification: {obs['verification_result']}")
    if obs.get("refund_amount") is not None:
        lines.append(f"Refund Amount: ${obs['refund_amount']:.2f}")
    if obs.get("tickets"):
        lines.append(f"Open Tickets ({len(obs['tickets'])}):")
        for i, t in enumerate(obs["tickets"], 1):
            lines.append(f"  [{i}] {t.get('ticket_id','?')} — {t.get('subject','?')}")
            lines.append(f"      User: {t.get('user_id','?')} | Type: {t.get('issue_type','?')} | Priority: {t.get('priority','?')}")
            lines.append(f"      Body: {t.get('body','')[:300]}...")
    lines.append(f"Done: {obs.get('done', False)} | Reward: {obs.get('reward', 0.0):.3f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_inference(
    difficulty: str = "easy",
    seed: Optional[int] = None,
    env_url: str = ENV_BASE_URL,
) -> None:
    """
    Run one complete episode using an LLM agent.

    Outputs [START], [STEP], and [END] blocks to stdout as required by
    the OpenEnv evaluation harness.
    """
    try:
        # Initialise OpenAI client with HF-compatible base URL
        client = OpenAI(
            api_key=HF_TOKEN or "not-needed",
            base_url=API_BASE_URL,
            timeout=30.0,
        )

        # ---- RESET ----
        try:
            obs = env_reset(env_url, difficulty=difficulty, seed=seed)
            initial_state = env_state(env_url)
        except Exception as e:
            print(f"[ERROR] Failed to initialize environment: {e}", flush=True)
            raise

        start_payload = {
            "episode_id": initial_state.get("episode_id", "unknown"),
            "difficulty": difficulty,
            "model": MODEL_NAME,
            "task_description": initial_state.get("task_description", ""),
            "num_tickets": len(obs.get("tickets", [])),
            "env_url": env_url,
        }
        print(f"[START] {json.dumps(start_payload)}", flush=True)

        # Build conversation history
        conversation: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"EPISODE START — Difficulty: {difficulty.upper()}\n"
                    f"Task: {obs.get('metadata', {}).get('task_description', '')}\n\n"
                    + obs_to_prompt(obs)
                    + "\n\nPlease choose your first action:"
                ),
            },
        ]

        # ---- EPISODE LOOP ----
        actions_taken: List[str] = []
        identity_verified = False
        refund_calculated = False
        total_reward = 0.0
        step_count = 0
        done = bool(obs.get("done", False))
        success = bool(obs.get("success", False))
        last_obs = obs

        while not done and step_count < MAX_STEPS:
            try:
                # Get action from LLM
                action = get_llm_action(client, conversation)
                action_type = action.get("action_type", "nop")

                # Track state for grading
                actions_taken.append(action_type)
                if action_type == "verify_identity":
                    identity_verified = True
                if action_type == "calculate_refund":
                    refund_calculated = True

                # Execute in environment
                new_obs = env_step(env_url, action)
                step_count += 1
                reward = float(new_obs.get("reward", 0.0))
                total_reward += reward
                done = bool(new_obs.get("done", False))
                success = bool(new_obs.get("success", False))
                last_obs = new_obs

                # Log step
                step_payload = {
                    "step": step_count,
                    "action": action,
                    "reward": reward,
                    "done": done,
                    "success": success,
                    "action_result": new_obs.get("action_result", ""),
                    "open_tickets": len(new_obs.get("tickets", [])),
                }
                print(f"[STEP] {json.dumps(step_payload)}", flush=True)

                # Feed observation back into conversation
                conversation.append({"role": "assistant", "content": json.dumps(action)})
                conversation.append({
                    "role": "user",
                    "content": obs_to_prompt(new_obs) + (
                        "\n\nAll tickets resolved. Episode complete."
                        if done and success
                        else "\n\nPlease choose your next action:"
                    ),
                })

                if done:
                    break

                time.sleep(STEP_SLEEP)

            except Exception as e:
                print(f"[ERROR] Step {step_count} failed: {e}", flush=True)
                # Continue with nop action to avoid crashing
                actions_taken.append("nop")
                break

        # ---- GRADE ----
        try:
            from grader import Grader
            from data_generator import generate_task
            
            # We know what the task was based on initial state
            task_difficulty = initial_state.get("difficulty", "easy")
            
            # Recreate the task to get expected_resolution
            dummy_task = {"difficulty": task_difficulty, "expected_resolution": "all_resolved" if success else "unknown"}
            grader = Grader(dummy_task)
            score = grader.grade(
                actions_taken=actions_taken,
                final_obs_done=done,
                final_obs_success=success,
                total_reward=total_reward,
                steps_taken=step_count,
                identity_verified=identity_verified,
                refund_calculated=refund_calculated,
            )
            label = Grader.describe(score)
        except Exception as e:
            print(f"[WARNING] Grading error: {e}", flush=True)
            score = 0.0
            label = f"Grading error: {type(e).__name__}"

        # ---- END ----
        try:
            final_state = env_state(env_url)
        except:
            final_state = initial_state

        end_payload = {
            "episode_id": final_state.get("episode_id", "unknown"),
            "difficulty": difficulty,
            "total_steps": step_count,
            "total_reward": round(total_reward, 4),
            "done": done,
            "success": success,
            "score": score,
            "grade": label,
            "actions_taken": actions_taken,
            "model": MODEL_NAME,
        }
        print(f"[END] {json.dumps(end_payload)}", flush=True)

    except Exception as e:
        print(f"[ERROR] Unhandled exception in run_inference: {type(e).__name__}: {e}", flush=True)
        # Output minimal END block to indicate failure
        end_payload = {
            "episode_id": "unknown",
            "difficulty": difficulty,
            "total_steps": 0,
            "total_reward": 0.0,
            "done": False,
            "success": False,
            "score": 0.0,
            "grade": f"Fatal error: {type(e).__name__}",
            "actions_taken": [],
            "model": MODEL_NAME,
        }
        print(f"[END] {json.dumps(end_payload)}", flush=True)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM inference on the Support Ticket environment."
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="easy",
        help="Task difficulty level (default: easy)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default=ENV_BASE_URL,
        help=f"Base URL of the running environment server (default: {ENV_BASE_URL})",
    )
    args = parser.parse_args()

    run_inference(
        difficulty=args.difficulty,
        seed=args.seed,
        env_url=args.env_url,
    )


if __name__ == "__main__":
    main()
