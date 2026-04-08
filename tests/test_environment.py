"""
Unit tests for the Support Ticket & DB Reconciliation Environment.

Tests cover:
  - Data generator (all three difficulty levels)
  - Environment reset and step mechanics
  - Reward logic for each action
  - Grader scoring
"""

from __future__ import annotations

import pytest
from models import ActionType, SupportAction
from data_generator import generate_task, REFUND_POLICY_DAYS, PARTIAL_REFUND_DAYS
from support_env import SupportEnvironment
from grader import Grader


# ---------------------------------------------------------------------------
# Data Generator Tests
# ---------------------------------------------------------------------------

class TestDataGenerator:
    def test_easy_task_structure(self):
        task = generate_task("easy", seed=42)
        assert task["difficulty"] == "easy"
        assert len(task["tickets"]) == 1
        assert len(task["database"]) >= 1
        ticket = task["tickets"][0]
        assert ticket["issue_type"] == "refund_request"
        assert ticket["user_id"] in task["database"]

    def test_medium_task_structure(self):
        task = generate_task("medium", seed=42)
        assert task["difficulty"] == "medium"
        assert len(task["tickets"]) == 3
        # One ticket should be a refund request
        types = [t["issue_type"] for t in task["tickets"]]
        assert "refund_request" in types
        assert "subscription" in types

    def test_hard_task_has_inconsistencies(self):
        task = generate_task("hard", seed=42)
        assert task["difficulty"] == "hard"
        assert len(task["tickets"]) == 1
        ticket = task["tickets"][0]
        # Should have injected inconsistencies
        assert "injected_inconsistencies" in ticket
        inconsistencies = ticket["injected_inconsistencies"]
        assert "email_mismatch" in inconsistencies
        assert "order_mismatch" in inconsistencies

    def test_reproducible_with_seed(self):
        task1 = generate_task("easy", seed=123)
        task2 = generate_task("easy", seed=123)
        # Same seed should give same ticket ID
        assert task1["tickets"][0]["ticket_id"] == task2["tickets"][0]["ticket_id"]

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValueError):
            generate_task("impossible")


# ---------------------------------------------------------------------------
# Environment Tests
# ---------------------------------------------------------------------------

class TestEnvironment:
    def setup_method(self):
        self.env = SupportEnvironment()

    def test_reset_easy_returns_observation(self):
        obs = self.env.reset(difficulty="easy", seed=42)
        assert obs.done == False
        assert len(obs.tickets) == 1
        assert obs.reward == 0.0
        assert obs.step_count == 0

    def test_reset_initialises_state(self):
        obs = self.env.reset(difficulty="easy", seed=42)
        state = self.env.state
        assert state.difficulty == "easy"
        assert state.step_count == 0
        assert state.is_resolved == False

    def test_lookup_user_valid(self):
        self.env.reset(difficulty="easy", seed=42)
        ticket = self.env._open_tickets[0]
        uid = ticket["user_id"]
        action = SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid)
        obs = self.env.step(action)
        assert obs.reward == 0.2
        assert obs.db_record is not None
        assert obs.db_record["user_id"] == uid
        assert obs.done == False

    def test_lookup_user_invalid(self):
        self.env.reset(difficulty="easy", seed=42)
        action = SupportAction(action_type=ActionType.LOOKUP_USER, user_id="NONEXISTENT")
        obs = self.env.step(action)
        assert obs.reward == -0.2
        assert "not found" in obs.error_message.lower()

    def test_lookup_user_redundant_penalised(self):
        self.env.reset(difficulty="easy", seed=42)
        uid = self.env._open_tickets[0]["user_id"]
        action = SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid)
        self.env.step(action)  # First lookup
        obs2 = self.env.step(action)  # Redundant lookup
        assert obs2.reward == -0.1  # Penalty for redundancy

    def test_calculate_refund_within_window(self):
        self.env.reset(difficulty="easy", seed=42)
        uid = self.env._open_tickets[0]["user_id"]
        db = self.env._database[uid]
        db["purchase_days_ago"] = 10  # Well within 30-day window

        SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid)
        self.env.step(SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid))
        obs = self.env.step(SupportAction(action_type=ActionType.CALCULATE_REFUND, user_id=uid))
        assert obs.reward == 0.1
        assert obs.refund_amount == db["plan_price"]  # Full refund

    def test_calculate_refund_partial_window(self):
        self.env.reset(difficulty="easy", seed=42)
        uid = self.env._open_tickets[0]["user_id"]
        db = self.env._database[uid]
        db["purchase_days_ago"] = 45  # In partial window (31-60 days)

        self.env.step(SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid))
        obs = self.env.step(SupportAction(action_type=ActionType.CALCULATE_REFUND, user_id=uid))
        assert obs.refund_amount == round(db["plan_price"] * 0.5, 2)

    def test_full_easy_episode(self):
        """Agent correctly resolves an easy refund ticket."""
        self.env.reset(difficulty="easy", seed=42)
        uid = self.env._open_tickets[0]["user_id"]

        obs1 = self.env.step(SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid))
        assert obs1.reward == 0.2

        obs2 = self.env.step(SupportAction(action_type=ActionType.CALCULATE_REFUND, user_id=uid))
        assert obs2.reward == 0.1

        obs3 = self.env.step(SupportAction(action_type=ActionType.ISSUE_REFUND, user_id=uid))
        assert obs3.reward >= 0.5  # Terminal success reward
        assert obs3.done == True
        assert obs3.success == True

    def test_nop_is_penalised(self):
        self.env.reset(difficulty="easy", seed=42)
        obs = self.env.step(SupportAction(action_type=ActionType.NOP))
        assert obs.reward == -0.1

    def test_hard_task_requires_verify_before_refund(self):
        """On hard task, issuing refund without verify_identity should be wrong."""
        self.env.reset(difficulty="hard", seed=42)
        uid = self.env._open_tickets[0]["user_id"]

        self.env.step(SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid))
        # Directly issue refund without verifying
        obs = self.env.step(SupportAction(action_type=ActionType.ISSUE_REFUND, user_id=uid))
        assert obs.reward == -0.5
        assert obs.done == True
        assert obs.success == False

    def test_escalate_on_hard_task_correct(self):
        """On hard task, escalating is the correct resolution."""
        self.env.reset(difficulty="hard", seed=42)
        uid = self.env._open_tickets[0]["user_id"]

        self.env.step(SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid))
        self.env.step(SupportAction(action_type=ActionType.VERIFY_IDENTITY, user_id=uid))
        obs = self.env.step(SupportAction(
            action_type=ActionType.ESCALATE,
            escalation_reason="Identity mismatch detected — email and order ID inconsistent with DB"
        ))
        assert obs.reward == 0.5
        assert obs.done == True

    def test_max_steps_terminates_episode(self):
        self.env.reset(difficulty="easy", seed=42)
        for _ in range(SupportEnvironment.MAX_STEPS + 2):
            obs = self.env.step(SupportAction(action_type=ActionType.NOP))
            if obs.done:
                break
        assert obs.done == True
        assert "Max steps" in obs.action_result

    def test_state_tracks_step_count(self):
        self.env.reset(difficulty="easy", seed=42)
        uid = self.env._open_tickets[0]["user_id"]
        for _ in range(3):
            self.env.step(SupportAction(action_type=ActionType.LOOKUP_USER, user_id=uid))
        assert self.env.state.step_count == 3


# ---------------------------------------------------------------------------
# Grader Tests
# ---------------------------------------------------------------------------

class TestGrader:
    def setup_method(self):
        self.easy_task = generate_task("easy", seed=42)
        self.medium_task = generate_task("medium", seed=42)
        self.hard_task = generate_task("hard", seed=42)

    def test_perfect_easy_episode(self):
        grader = Grader(self.easy_task)
        score = grader.grade(
            actions_taken=["lookup_user", "calculate_refund", "issue_refund", "close_ticket"],
            final_obs_done=True,
            final_obs_success=True,
            total_reward=0.9,
            steps_taken=4,
            identity_verified=False,
            refund_calculated=True,
        )
        assert score >= 0.85, f"Expected ≥0.85, got {score}"

    def test_failed_episode_zero_score(self):
        grader = Grader(self.easy_task)
        score = grader.grade(
            actions_taken=["nop", "nop"],
            final_obs_done=False,
            final_obs_success=False,
            total_reward=-0.2,
            steps_taken=2,
            identity_verified=False,
            refund_calculated=False,
        )
        assert score < 0.3, f"Expected <0.3 for failed episode, got {score}"

    def test_hard_task_requires_verify_for_full_score(self):
        grader = Grader(self.hard_task)
        # Without verification
        score_no_verify = grader.grade(
            actions_taken=["lookup_user", "escalate"],
            final_obs_done=True,
            final_obs_success=True,
            total_reward=0.5,
            steps_taken=2,
            identity_verified=False,
            refund_calculated=False,
        )
        # With verification
        score_with_verify = grader.grade(
            actions_taken=["lookup_user", "verify_identity", "escalate"],
            final_obs_done=True,
            final_obs_success=True,
            total_reward=0.65,
            steps_taken=3,
            identity_verified=True,
            refund_calculated=False,
        )
        assert score_with_verify > score_no_verify

    def test_efficiency_penalty_for_extra_steps(self):
        grader = Grader(self.easy_task)
        score_efficient = grader.grade(
            actions_taken=["lookup_user", "calculate_refund", "issue_refund", "close_ticket"],
            final_obs_done=True, final_obs_success=True, total_reward=0.9,
            steps_taken=4, identity_verified=False, refund_calculated=True,
        )
        score_inefficient = grader.grade(
            actions_taken=["lookup_user", "nop", "nop", "nop", "calculate_refund", "issue_refund", "close_ticket"],
            final_obs_done=True, final_obs_success=True, total_reward=0.6,
            steps_taken=7, identity_verified=False, refund_calculated=True,
        )
        assert score_efficient > score_inefficient

    def test_score_always_in_range(self):
        grader = Grader(self.easy_task)
        for done, success, reward, steps in [
            (True, True, 1.0, 2),
            (False, False, -5.0, 20),
            (True, False, 0.0, 10),
        ]:
            score = grader.grade(
                actions_taken=["nop"],
                final_obs_done=done,
                final_obs_success=success,
                total_reward=reward,
                steps_taken=steps,
            )
            assert 0.0 <= score <= 1.0, f"Score {score} out of [0, 1] range"

    def test_describe_labels(self):
        assert Grader.describe(0.95) == "Excellent"
        assert Grader.describe(0.80) == "Good"
        assert Grader.describe(0.60) == "Partial"
        assert Grader.describe(0.30) == "Poor"
        assert Grader.describe(0.10) == "Failed"
