"""
FastAPI application for the Support Ticket & DB Reconciliation Environment.

Uses OpenEnv's built-in create_app to generate a fully compliant server
that handles multi-sessions, WebSockets, and the required API routes.
"""

from __future__ import annotations

import os
from openenv.core.env_server.http_server import create_app

try:
    from models import SupportAction, SupportObservation
    from support_env import SupportEnvironment
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from models import SupportAction, SupportObservation
    from support_env import SupportEnvironment

app = create_app(
    SupportEnvironment,
    SupportAction,
    SupportObservation,
    env_name="support_ticket_env"
)

from pydantic import BaseModel
from typing import Dict, Any

class GradeRequest(BaseModel):
    actions_taken: list
    final_obs_done: bool
    final_obs_success: bool
    total_reward: float
    steps_taken: int
    identity_verified: bool = False
    refund_calculated: bool = False

@app.post("/grade")
async def grade(req: GradeRequest) -> Dict[str, Any]:
    from grader import Grader
    from support_env import SupportEnvironment
    # As an external utility, we don't have the exact task, but we can reconstruct it from the first task 
    # Or for simplicity, we mock task since in create_app environment sessions are dynamic.
    return {"score": 0.0, "label": "Grading skipped in standard OpenEnv app"}


def main() -> None:
    """Run via: uv run server  OR  python server/app.py"""
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "7860")),
        log_level="info",
    )

if __name__ == "__main__":
    main()
