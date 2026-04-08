"""
Support Ticket & Database Reconciliation Environment - Models

Defines all Pydantic type-safe data structures used by the environment:
- SupportAction: Actions the agent can take (lookup, refund, update, escalate, etc.)
- SupportObservation: What the agent sees after each step
- SupportState: Episode metadata tracked by the environment
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------


class ActionType(str, Enum):
    """All valid action types for the support environment."""

    LOOKUP_USER = "lookup_user"
    """Query the internal DB by user_id or email."""

    LOOKUP_ORDER = "lookup_order"
    """Query the internal DB for a specific order."""

    VERIFY_IDENTITY = "verify_identity"
    """Cross-check ticket claim against DB record to detect mismatches."""

    CALCULATE_REFUND = "calculate_refund"
    """Compute the eligible refund amount based on policy rules."""

    ISSUE_REFUND = "issue_refund"
    """Execute a refund for a resolved ticket."""

    UPDATE_SUBSCRIPTION = "update_subscription"
    """Upgrade, downgrade, or cancel a user subscription."""

    ESCALATE = "escalate"
    """Escalate ticket to Tier-2 / engineering team."""

    CLOSE_TICKET = "close_ticket"
    """Mark a ticket as resolved and close it."""

    NOP = "nop"
    """No operation — used when the agent is unsure."""


class SupportAction(BaseModel):
    """
    Typed action submitted by the agent each step.

    Attributes:
        action_type: What kind of action to perform.
        user_id: Target user identifier (optional, used with lookup/refund/update).
        order_id: Target order identifier (optional).
        new_plan: Subscription plan name for UPDATE_SUBSCRIPTION actions.
        escalation_reason: Human-readable reason for ESCALATE actions.
        resolution_note: Summary note when closing a ticket.
        metadata: Arbitrary key-value pairs for extensibility.
    """

    action_type: ActionType
    user_id: Optional[str] = Field(None, description="User ID to look up or act on")
    order_id: Optional[str] = Field(None, description="Order ID to look up or act on")
    new_plan: Optional[str] = Field(
        None, description="Target subscription plan for updates"
    )
    escalation_reason: Optional[str] = Field(
        None, description="Reason for escalation"
    )
    resolution_note: Optional[str] = Field(
        None, description="Resolution summary when closing ticket"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Extra key-value context"
    )


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class SupportObservation(BaseModel):
    """
    What the agent receives after each action.

    Attributes:
        done: Whether the episode is finished.
        reward: Immediate scalar reward for this step.
        success: True when terminal state is a correct resolution.
        tickets: List of open support tickets (JSON-serialisable dicts).
        db_record: User/order DB record returned after a lookup.
        verification_result: Result of VERIFY_IDENTITY (match / mismatch / pending).
        refund_amount: Computed refund amount (set after CALCULATE_REFUND).
        action_result: Human-readable result of the last action.
        step_count: Number of steps elapsed so far.
        error_message: Set when an action fails due to bad input.
        metadata: Extra structured data (task difficulty, episode_id, etc.).
    """

    done: bool = False
    reward: float = 0.0
    success: bool = False
    tickets: List[Dict[str, Any]] = Field(default_factory=list)
    db_record: Optional[Dict[str, Any]] = None
    verification_result: Optional[str] = None  # "match" | "mismatch" | "pending"
    refund_amount: Optional[float] = None
    action_result: str = ""
    step_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class SupportState(BaseModel):
    """
    Episode metadata (mirrors openenv.core.env_server.types.State).

    Attributes:
        episode_id: Unique identifier for the current episode.
        step_count: Total steps taken.
        difficulty: Task difficulty level ('easy' | 'medium' | 'hard').
        task_description: Human-readable description of the episode objective.
        is_resolved: Whether the episode ended with a correct resolution.
        total_reward: Cumulative reward so far.
    """

    episode_id: Optional[str] = None
    step_count: int = 0
    difficulty: str = "easy"
    task_description: str = ""
    is_resolved: bool = False
    total_reward: float = 0.0
