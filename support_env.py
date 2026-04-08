"""
Support Ticket & Database Reconciliation Environment.

This is the core environment module implementing the OpenEnv gymnasium-style API:
  - reset(): Initialise a new episode and return the first observation.
  - step(action): Execute an agent action and return the resulting observation + reward.
  - state: Property returning episode metadata.

Reward shaping:
  +0.20  Correct DB lookup (user/order found and relevant)
  +0.15  Verify identity when data is inconsistent (hard task)
  +0.10  Correctly calculate refund before issuing
  +0.50  Successful terminal resolution (correct action taken)
  -0.10  Unnecessary / redundant steps
  -0.20  Lookup fails (bad user_id / order_id provided)
  -0.50  Incorrect terminal resolution (wrong action, wrong amount, etc.)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from data_generator import REFUND_POLICY_DAYS, PARTIAL_REFUND_DAYS, generate_task
from models import ActionType, SupportAction, SupportObservation, SupportState

class SupportEnvironment:
    """
    Customer Support Ticket & Database Reconciliation OpenEnv Environment.

    The agent acts as a Support Operations specialist and must:
      1. Read incoming JSON support tickets.
      2. Query the mock internal database (DB reconciliation).
      3. Execute API-based actions (refund, upgrade, escalate, close).

    Three difficulty levels:
      - easy:   Single ticket, user exists, straightforward refund.
      - medium: Multiple tickets, refund calculation with policy logic.
      - hard:   Inconsistent data requiring identity verification before acting.
    """

    # Max steps before the environment force-terminates the episode
    MAX_STEPS = 20

    def __init__(self) -> None:
        self._state = SupportState()
        self._task: Dict[str, Any] = {}
        self._database: Dict[str, Any] = {}
        self._open_tickets: list = []
        self._resolved_tickets: list = []
        self._identity_verified: bool = False
        self._refund_calculated: bool = False
        self._refund_amount: float = 0.0
        self._lookups_done: set = set()
        self._last_lookup_uid: Optional[str] = None
        self._actions_taken: list = []

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    def reset(
        self,
        difficulty: str = "easy",
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportObservation:
        """
        Initialise a new support episode.

        Args:
            difficulty: Task difficulty — 'easy', 'medium', or 'hard'.
            seed: Optional RNG seed for reproducibility.
            episode_id: Optional manually specified episode ID.

        Returns:
            Initial SupportObservation with current open tickets.
        """
        self._task = generate_task(difficulty=difficulty, seed=seed)
        self._database = self._task["database"]
        self._open_tickets = list(self._task["tickets"])  # shallow copy
        self._resolved_tickets = []
        self._identity_verified = False
        self._refund_calculated = False
        self._refund_amount = 0.0
        self._lookups_done = set()
        self._last_lookup_uid = None
        self._actions_taken = []

        eid = episode_id or str(uuid.uuid4())
        self._state = SupportState(
            episode_id=eid,
            step_count=0,
            difficulty=difficulty,
            task_description=self._task["task_description"],
            is_resolved=False,
            total_reward=0.0,
        )

        return SupportObservation(
            done=False,
            reward=0.0,
            success=False,
            tickets=self._open_tickets,
            db_record=None,
            verification_result=None,
            refund_amount=None,
            action_result=(
                f"Episode started. Task: {self._task['task_description']}. "
                f"You have {len(self._open_tickets)} open ticket(s) to resolve."
            ),
            step_count=0,
            error_message=None,
            metadata={
                "difficulty": difficulty,
                "episode_id": eid,
                "task_description": self._task["task_description"],
                "num_tickets": len(self._open_tickets),
            },
        )

    def step(self, action: SupportAction) -> SupportObservation:
        """
        Execute one agent action and return the resulting observation.

        Args:
            action: A SupportAction instance describing what the agent does.

        Returns:
            SupportObservation with updated state, reward, and done flag.
        """
        self._state.step_count += 1
        self._actions_taken.append(action.action_type)

        # Force terminate if max steps exceeded
        if self._state.step_count > self.MAX_STEPS:
            reward = -0.5
            self._state.total_reward += reward
            return self._make_obs(
                done=True,
                reward=reward,
                success=False,
                action_result="Max steps exceeded. Episode terminated.",
                error_message="Too many steps taken without resolution.",
            )

        # Dispatch to action handler
        handler = {
            ActionType.LOOKUP_USER: self._handle_lookup_user,
            ActionType.LOOKUP_ORDER: self._handle_lookup_order,
            ActionType.VERIFY_IDENTITY: self._handle_verify_identity,
            ActionType.CALCULATE_REFUND: self._handle_calculate_refund,
            ActionType.ISSUE_REFUND: self._handle_issue_refund,
            ActionType.UPDATE_SUBSCRIPTION: self._handle_update_subscription,
            ActionType.ESCALATE: self._handle_escalate,
            ActionType.CLOSE_TICKET: self._handle_close_ticket,
            ActionType.NOP: self._handle_nop,
        }.get(action.action_type)

        if handler is None:
            return self._make_obs(
                done=False,
                reward=-0.1,
                action_result=f"Unknown action type: {action.action_type}",
                error_message="Action type not recognised.",
            )

        return handler(action)

    async def reset_async(self, **kwargs) -> SupportObservation:
        """Async wrapper required by openenv wrapper."""
        return self.reset(**kwargs)

    async def step_async(self, action: SupportAction) -> SupportObservation:
        """Async wrapper required by openenv wrapper."""
        return self.step(action)

    async def state_async(self) -> SupportState:
        """Async wrapper required by openenv wrapper."""
        return self.state

    async def close_async(self) -> None:
        """Async wrapper required by openenv wrapper."""
        self.close()

    @property
    def state(self) -> SupportState:
        """Return current episode state metadata."""
        return self._state

    # ------------------------------------------------------------------
    # Action Handlers
    # ------------------------------------------------------------------

    def _handle_lookup_user(self, action: SupportAction) -> SupportObservation:
        uid = action.user_id
        if not uid:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result="LOOKUP_USER failed: user_id not provided.",
                error_message="user_id is required for LOOKUP_USER.",
            )

        db_record = self._database.get(uid)
        if db_record is None:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result=f"LOOKUP_USER failed: No record found for user_id '{uid}'.",
                error_message=f"User '{uid}' not found in database.",
            )

        # Reward only the first lookup per unique user (avoid reward hacking)
        if uid in self._lookups_done:
            reward = -0.1  # Redundant lookup
            msg = f"LOOKUP_USER: User '{uid}' already looked up. Redundant action."
        else:
            reward = 0.2
            self._lookups_done.add(uid)
            msg = f"LOOKUP_USER: Found user '{db_record['first_name']} {db_record['last_name']}' (plan: {db_record['plan']}, order: {db_record['order_id']})."

        self._last_lookup_uid = uid
        self._state.total_reward += reward
        return self._make_obs(
            done=False,
            reward=reward,
            action_result=msg,
            db_record=db_record,
        )

    def _handle_lookup_order(self, action: SupportAction) -> SupportObservation:
        order_id = action.order_id
        if not order_id:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result="LOOKUP_ORDER failed: order_id not provided.",
                error_message="order_id is required for LOOKUP_ORDER.",
            )

        # Search all DB records for a matching order
        matched_record = None
        for record in self._database.values():
            if record.get("order_id") == order_id:
                matched_record = record
                break

        if matched_record is None:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result=f"LOOKUP_ORDER failed: No order found with ID '{order_id}'.",
                error_message=f"Order '{order_id}' not found in database.",
            )

        reward = 0.2
        self._state.total_reward += reward
        self._last_lookup_uid = matched_record["user_id"]
        self._lookups_done.add(matched_record["user_id"])
        return self._make_obs(
            done=False,
            reward=reward,
            action_result=(
                f"LOOKUP_ORDER: Found order '{order_id}' for user "
                f"'{matched_record['first_name']} {matched_record['last_name']}' "
                f"(plan: {matched_record['plan']}, purchased {matched_record['purchase_days_ago']} days ago)."
            ),
            db_record=matched_record,
        )

    def _handle_verify_identity(self, action: SupportAction) -> SupportObservation:
        """
        Cross-check ticket claims against the DB record.
        Critical for the 'hard' task where data is inconsistent.
        """
        uid = action.user_id or self._last_lookup_uid
        if not uid:
            return self._make_obs(
                done=False,
                reward=-0.1,
                action_result="VERIFY_IDENTITY failed: No user selected. Run LOOKUP_USER first.",
                error_message="Must lookup a user before verifying identity.",
            )

        db_record = self._database.get(uid)
        if not db_record:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result=f"VERIFY_IDENTITY failed: No DB record for '{uid}'.",
                error_message=f"User '{uid}' not found.",
            )

        # Find the relevant ticket
        ticket = next(
            (t for t in self._open_tickets if t.get("user_id") == uid), None
        )
        if ticket is None:
            return self._make_obs(
                done=False,
                reward=-0.1,
                action_result=f"VERIFY_IDENTITY: No open ticket found for user '{uid}'.",
            )

        mismatches = []

        # Email check
        ticket_email = ticket.get("email_in_ticket") or ticket.get("email", "")
        db_email = db_record.get("email", "")
        if ticket_email.lower() != db_email.lower():
            mismatches.append(f"Email mismatch: ticket='{ticket_email}' vs DB='{db_email}'")

        # Name check (last name)
        ticket_body = ticket.get("body", "")
        db_last = db_record.get("last_name", "").lower()
        # Simple heuristic: check if db last name is in ticket body
        if db_last not in ticket_body.lower():
            mismatches.append(f"Name discrepancy detected in ticket body (DB last name '{db_record['last_name']}' not found)")

        # Order ID check (hard task injects fake order in body)
        db_order = db_record.get("order_id", "")
        if db_order not in ticket_body:
            mismatches.append(f"Order ID mismatch: DB order '{db_order}' not referenced in ticket")

        difficulty = self._task.get("difficulty", "easy")

        if mismatches:
            self._identity_verified = False
            # On hard task: award reward for CORRECTLY detecting inconsistency
            reward = 0.15 if difficulty == "hard" else 0.0
            result = "mismatch"
            msg = (
                f"VERIFY_IDENTITY: ⚠️ {len(mismatches)} inconsistency(-ies) detected!\n"
                + "\n".join(f"  • {m}" for m in mismatches)
                + "\nRecommendation: Escalate or request confirmation before acting."
            )
        else:
            self._identity_verified = True
            reward = 0.1 if difficulty in ("easy", "medium") else 0.0
            result = "match"
            msg = "VERIFY_IDENTITY: ✅ All data matches DB record. Identity confirmed."

        self._state.total_reward += reward
        return self._make_obs(
            done=False,
            reward=reward,
            action_result=msg,
            verification_result=result,
            db_record=db_record,
        )

    def _handle_calculate_refund(self, action: SupportAction) -> SupportObservation:
        """
        Calculate the eligible refund based on purchase date and policy.
        Full refund within 30 days, 50% within 60 days, none after.
        """
        uid = action.user_id or self._last_lookup_uid
        if not uid:
            return self._make_obs(
                done=False,
                reward=-0.1,
                action_result="CALCULATE_REFUND failed: No user selected.",
                error_message="Must lookup user before calculating refund.",
            )

        db_record = self._database.get(uid)
        if not db_record:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result=f"CALCULATE_REFUND failed: No DB record for '{uid}'.",
            )

        days = db_record.get("purchase_days_ago", 999)
        price = db_record.get("plan_price", 0.0)

        if days <= REFUND_POLICY_DAYS:
            refund = price
            policy_msg = f"Full refund (purchased {days} days ago, within {REFUND_POLICY_DAYS}-day window)."
        elif days <= PARTIAL_REFUND_DAYS:
            refund = round(price * 0.5, 2)
            policy_msg = f"Partial 50% refund (purchased {days} days ago, within {PARTIAL_REFUND_DAYS}-day window)."
        else:
            refund = 0.0
            policy_msg = f"No refund eligible (purchased {days} days ago, outside {PARTIAL_REFUND_DAYS}-day window)."

        self._refund_calculated = True
        self._refund_amount = refund
        reward = 0.1  # Reward for running the calculation step
        self._state.total_reward += reward

        return self._make_obs(
            done=False,
            reward=reward,
            action_result=(
                f"CALCULATE_REFUND: Eligible refund = ${refund:.2f}. {policy_msg}"
            ),
            refund_amount=refund,
            db_record=db_record,
        )

    def _handle_issue_refund(self, action: SupportAction) -> SupportObservation:
        """Issue the refund. Must have looked up user. Rewards correct terminal action."""
        uid = action.user_id or self._last_lookup_uid
        expected_resolution = self._task.get("expected_resolution", "")
        difficulty = self._task.get("difficulty", "easy")

        if not uid or uid not in self._database:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result="ISSUE_REFUND failed: Valid user must be looked up first.",
                error_message="No valid user selected for refund.",
            )

        db_record = self._database[uid]
        days = db_record.get("purchase_days_ago", 999)
        price = db_record.get("plan_price", 0.0)

        # Hard task: issuing refund without verify_identity is wrong
        if difficulty == "hard" and not self._identity_verified:
            reward = -0.5
            self._state.total_reward += reward
            self._state.is_resolved = False
            return self._make_obs(
                done=True,
                reward=reward,
                success=False,
                action_result=(
                    "ISSUE_REFUND: ❌ Incorrect! On inconsistent data you must VERIFY_IDENTITY "
                    "and ESCALATE rather than issue a refund. Identity was not verified."
                ),
                error_message="Refund issued without identity verification on hard task.",
            )

        # Check if refund is even appropriate
        if expected_resolution not in ("issue_refund", "all_resolved"):
            reward = -0.5
            self._state.total_reward += reward
            self._state.is_resolved = False
            return self._make_obs(
                done=True,
                reward=reward,
                success=False,
                action_result="ISSUE_REFUND: ❌ Incorrect resolution for this ticket type.",
            )

        # Check eligibility
        if days > PARTIAL_REFUND_DAYS:
            reward = -0.3
            self._state.total_reward += reward
            return self._make_obs(
                done=False,
                reward=reward,
                action_result=(
                    f"ISSUE_REFUND: ❌ Refund not eligible — purchased {days} days ago "
                    f"(policy window is {PARTIAL_REFUND_DAYS} days)."
                ),
            )

        # Compute correct refund
        if days <= REFUND_POLICY_DAYS:
            correct_refund = price
        else:
            correct_refund = round(price * 0.5, 2)

        # If agent calculated first, great — check consistency
        calculation_bonus = 0.0
        if self._refund_calculated:
            if abs(self._refund_amount - correct_refund) < 0.01:
                calculation_bonus = 0.05  # Bonus for accurate pre-calculation
            else:
                calculation_bonus = -0.05  # Penalty for wrong pre-calculation

        reward = 0.5 + calculation_bonus
        self._state.total_reward += reward
        self._state.is_resolved = True

        # Mark tickets for this user as resolved
        self._open_tickets = [t for t in self._open_tickets if t.get("user_id") != uid]

        done = len(self._open_tickets) == 0

        return self._make_obs(
            done=done,
            reward=reward,
            success=done,
            action_result=(
                f"ISSUE_REFUND: ✅ Refund of ${correct_refund:.2f} issued to "
                f"{db_record['first_name']} {db_record['last_name']} "
                f"({db_record['email']}) via {db_record['payment_method']}."
            ),
            refund_amount=correct_refund,
            db_record=db_record,
        )

    def _handle_update_subscription(self, action: SupportAction) -> SupportObservation:
        """Handle subscription plan changes."""
        uid = action.user_id or self._last_lookup_uid
        new_plan = action.new_plan

        if not uid or uid not in self._database:
            return self._make_obs(
                done=False,
                reward=-0.2,
                action_result="UPDATE_SUBSCRIPTION failed: Valid user must be looked up first.",
                error_message="No valid user selected.",
            )

        if not new_plan:
            return self._make_obs(
                done=False,
                reward=-0.1,
                action_result="UPDATE_SUBSCRIPTION failed: new_plan not specified.",
                error_message="new_plan is required for subscription update.",
            )

        valid_plans = ["free", "starter", "pro", "enterprise"]
        if new_plan not in valid_plans:
            return self._make_obs(
                done=False,
                reward=-0.1,
                action_result=f"UPDATE_SUBSCRIPTION failed: '{new_plan}' is not a valid plan.",
                error_message=f"Valid plans: {valid_plans}",
            )

        db_record = self._database[uid]
        old_plan = db_record["plan"]

        if old_plan == new_plan:
            return self._make_obs(
                done=False,
                reward=-0.1,
                action_result=f"UPDATE_SUBSCRIPTION: User already on '{new_plan}' plan. No change.",
            )

        # Apply the update
        db_record["plan"] = new_plan
        db_record["plan_price"] = __import__("data_generator").SUBSCRIPTION_PLANS[new_plan]["price"]

        reward = 0.5
        self._state.total_reward += reward
        self._state.is_resolved = True

        self._open_tickets = [t for t in self._open_tickets if t.get("user_id") != uid]
        done = len(self._open_tickets) == 0

        return self._make_obs(
            done=done,
            reward=reward,
            success=done,
            action_result=(
                f"UPDATE_SUBSCRIPTION: ✅ {db_record['first_name']} {db_record['last_name']} "
                f"upgraded from '{old_plan}' → '{new_plan}'."
            ),
            db_record=db_record,
        )

    def _handle_escalate(self, action: SupportAction) -> SupportObservation:
        """Escalate ticket to Tier-2. Correct on hard tasks with identity mismatches."""
        expected_resolution = self._task.get("expected_resolution", "")
        difficulty = self._task.get("difficulty", "easy")

        # On hard task, escalation is the correct answer
        if expected_resolution == "escalate" or difficulty == "hard":
            # Check if the agent detected mismatch first
            if difficulty == "hard" and self._state.step_count <= 3:
                # Agent escalated quickly — good reasoning
                reward = 0.5
            elif expected_resolution == "escalate":
                reward = 0.5
            else:
                reward = -0.1  # Escalation when it wasn't needed
            success = reward > 0
        else:
            # Escalating an easy/medium ticket is usually wrong
            reward = -0.3
            success = False

        reason = action.escalation_reason or "No reason provided"
        self._state.total_reward += reward
        self._state.is_resolved = success

        if success:
            self._open_tickets = self._open_tickets[1:]  # Remove first ticket
        done = len(self._open_tickets) == 0 or success

        return self._make_obs(
            done=done,
            reward=reward,
            success=success,
            action_result=(
                f"ESCALATE: {'✅' if success else '⚠️'} Ticket escalated to Tier-2. "
                f"Reason: {reason}"
            ),
        )

    def _handle_close_ticket(self, action: SupportAction) -> SupportObservation:
        """Close the current ticket. Appropriate for resolved or informational tickets."""
        if not self._open_tickets:
            return self._make_obs(
                done=True,
                reward=0.0,
                success=True,
                action_result="CLOSE_TICKET: No open tickets remaining. Episode complete.",
            )

        ticket = self._open_tickets[0]
        expected_resolution = self._task.get("expected_resolution", "")
        difficulty = self._task.get("difficulty", "easy")

        # On hard task, closing without verifying is incorrect
        if difficulty == "hard" and not self._identity_verified:
            reward = -0.5
            self._state.total_reward += reward
            return self._make_obs(
                done=True,
                reward=reward,
                success=False,
                action_result=(
                    "CLOSE_TICKET: ❌ You closed a high-priority billing dispute ticket "
                    "without verifying identity or escalating. Incorrect resolution."
                ),
            )

        note = action.resolution_note or "Resolved by support agent."
        self._open_tickets.pop(0)
        done = len(self._open_tickets) == 0

        # Reward appropriately
        if expected_resolution == "all_resolved" or ticket.get("issue_type") in (
            "technical", "feature_request"
        ):
            reward = 0.3  # Closing informational/technical tickets is fine
        elif ticket.get("issue_type") == "refund_request" and not self._refund_calculated:
            reward = -0.3  # Closed a refund ticket without processing refund
        else:
            reward = 0.2

        self._state.total_reward += reward
        self._state.is_resolved = done

        return self._make_obs(
            done=done,
            reward=reward,
            success=done,
            action_result=(
                f"CLOSE_TICKET: ✅ Ticket '{ticket.get('ticket_id')}' closed. Note: {note}. "
                f"{len(self._open_tickets)} ticket(s) remaining."
            ),
        )

    def _handle_nop(self, action: SupportAction) -> SupportObservation:
        """No-op: always incurs a small penalty to discourage dithering."""
        reward = -0.1
        self._state.total_reward += reward
        return self._make_obs(
            done=False,
            reward=reward,
            action_result="NOP: No action taken. Penalty applied for wasted step.",
        )

    def close(self) -> None:
        """Clean up any resources. Required by OpenEnv wrapper."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_obs(
        self,
        done: bool = False,
        reward: float = 0.0,
        success: bool = False,
        action_result: str = "",
        db_record: Optional[Dict[str, Any]] = None,
        verification_result: Optional[str] = None,
        refund_amount: Optional[float] = None,
        error_message: Optional[str] = None,
    ) -> SupportObservation:
        return SupportObservation(
            done=done,
            reward=reward,
            success=success,
            tickets=list(self._open_tickets),
            db_record=db_record,
            verification_result=verification_result,
            refund_amount=refund_amount,
            action_result=action_result,
            step_count=self._state.step_count,
            error_message=error_message,
            metadata={
                "difficulty": self._task.get("difficulty", "unknown"),
                "episode_id": self._state.episode_id,
                "total_reward": self._state.total_reward,
                "open_tickets": len(self._open_tickets),
                "identity_verified": self._identity_verified,
                "refund_calculated": self._refund_calculated,
            },
        )
