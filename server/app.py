"""
FastAPI application for the Support Ticket & DB Reconciliation Environment.

Exposes the environment via HTTP endpoints compatible with the OpenEnv spec:
  POST /reset  → Start new episode, returns initial SupportObservation
  POST /step   → Execute an action, returns SupportObservation
  GET  /state  → Return current SupportState metadata

Also serves a web debug interface when ENABLE_WEB_INTERFACE=true.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# ---- Import from parent package (supports both standalone and in-repo) ----
try:
    from models import ActionType, SupportAction, SupportObservation, SupportState
    from support_env import SupportEnvironment
    from grader import Grader
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import ActionType, SupportAction, SupportObservation, SupportState
    from support_env import SupportEnvironment
    from grader import Grader


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Support Ticket & DB Reconciliation Environment",
    description=(
        "An OpenEnv-compatible RL environment where an AI agent acts as a "
        "Support Operations specialist. Reads tickets, queries a mock DB, and "
        "resolves issues via refunds, subscription updates, and escalations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton environment instance (each HF Space pod is single-session)
_env = SupportEnvironment()
_last_task: Dict[str, Any] = {}
_actions_taken: list = []
_identity_verified: bool = False
_refund_calculated: bool = False


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    difficulty: str = "easy"
    seed: Optional[int] = None
    episode_id: Optional[str] = None


class StepRequest(BaseModel):
    action: SupportAction


class GradeRequest(BaseModel):
    """Optional endpoint to score a completed episode."""
    actions_taken: list
    final_obs_done: bool
    final_obs_success: bool
    total_reward: float
    steps_taken: int
    identity_verified: bool = False
    refund_calculated: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/reset", response_model=SupportObservation)
async def reset(req: ResetRequest) -> SupportObservation:
    """
    Initialize a new support episode.

    Args:
        difficulty: 'easy' | 'medium' | 'hard'
        seed: Optional RNG seed
        episode_id: Optional episode ID override

    Returns:
        Initial SupportObservation with open tickets.
    """
    global _last_task, _actions_taken, _identity_verified, _refund_calculated
    obs = _env.reset(
        difficulty=req.difficulty,
        seed=req.seed,
        episode_id=req.episode_id,
    )
    _last_task = _env._task
    _actions_taken = []
    _identity_verified = False
    _refund_calculated = False
    return obs


@app.post("/step", response_model=SupportObservation)
async def step(req: StepRequest) -> SupportObservation:
    """
    Execute one agent action.

    Args:
        action: A SupportAction with action_type and relevant fields.

    Returns:
        SupportObservation with reward, done flag, and updated state.
    """
    global _actions_taken, _identity_verified, _refund_calculated

    action = req.action
    obs = _env.step(action)

    # Track for grading
    _actions_taken.append(str(action.action_type.value))
    if action.action_type == ActionType.VERIFY_IDENTITY:
        _identity_verified = obs.verification_result == "match" or True  # Track attempt
    if action.action_type == ActionType.CALCULATE_REFUND:
        _refund_calculated = True

    return obs


@app.get("/state", response_model=SupportState)
async def state() -> SupportState:
    """Return current episode state metadata."""
    return _env.state


@app.post("/grade")
async def grade(req: GradeRequest) -> Dict[str, Any]:
    """
    Grade a completed episode (optional utility endpoint).

    Returns:
        dict with 'score' (float 0-1) and 'label' (str).
    """
    if not _last_task:
        raise HTTPException(status_code=400, detail="No episode has been run yet. Call /reset first.")

    grader = Grader(_last_task)
    score = grader.grade(
        actions_taken=req.actions_taken,
        final_obs_done=req.final_obs_done,
        final_obs_success=req.final_obs_success,
        total_reward=req.total_reward,
        steps_taken=req.steps_taken,
        identity_verified=req.identity_verified,
        refund_calculated=req.refund_calculated,
    )
    return {
        "score": score,
        "label": Grader.describe(score),
        "difficulty": _last_task.get("difficulty", "unknown"),
        "expected_resolution": _last_task.get("expected_resolution", "unknown"),
    }


@app.get("/health")
async def health() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "env": "support_ticket_env"}


# ---------------------------------------------------------------------------
# Optional: Web debug interface
# ---------------------------------------------------------------------------

ENABLE_WEB = os.getenv("ENABLE_WEB_INTERFACE", "false").lower() == "true"

if ENABLE_WEB:
    @app.get("/web", response_class=HTMLResponse)
    async def web_interface() -> str:
        """Minimal web interface for interactive debugging."""
        return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Support Ticket Env — Debug UI</title>
<style>
  body { font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; }
  button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; padding: 8px 16px; cursor: pointer; border-radius: 6px; margin: 4px; }
  button:hover { background: #30363d; }
  textarea { width: 100%; background: #161b22; color: #c9d1d9; border: 1px solid #30363d; padding: 8px; border-radius: 6px; font-family: monospace; }
  pre { background: #161b22; padding: 12px; border-radius: 6px; overflow: auto; }
  .label { color: #8b949e; font-size: 0.85em; }
</style>
</head>
<body>
<h1>🎫 Support Ticket Env — Debug UI</h1>
<p class="label">POST /reset → GET /state → POST /step (loop) → POST /grade</p>
<div>
  <button onclick="resetEnv('easy')">Reset (Easy)</button>
  <button onclick="resetEnv('medium')">Reset (Medium)</button>
  <button onclick="resetEnv('hard')">Reset (Hard)</button>
  <button onclick="getState()">Get State</button>
</div>
<br>
<textarea id="action-json" rows="6" placeholder='{"action_type": "lookup_user", "user_id": "U-XXXXXXXX"}'></textarea>
<br>
<button onclick="doStep()">Execute Step</button>
<br><br>
<pre id="output">Output will appear here...</pre>
<script>
async function resetEnv(d) {
  const r = await fetch('/reset', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({difficulty:d})});
  document.getElementById('output').textContent = JSON.stringify(await r.json(), null, 2);
}
async function getState() {
  const r = await fetch('/state');
  document.getElementById('output').textContent = JSON.stringify(await r.json(), null, 2);
}
async function doStep() {
  const action = JSON.parse(document.getElementById('action-json').value);
  const r = await fetch('/step', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action})});
  document.getElementById('output').textContent = JSON.stringify(await r.json(), null, 2);
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run via: uv run server  OR  python server/app.py"""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
