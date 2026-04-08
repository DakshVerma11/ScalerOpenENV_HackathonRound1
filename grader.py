"""
Grader: Evaluates agent performance on the Support Ticket environment.

Returns a float in [0.0, 1.0] based on:
  - Resolution accuracy  (50%): Was the correct terminal action taken?
  - Efficiency score     (30%): Were steps minimised relative to optimal?
  - Process compliance   (20%): Did the agent follow correct procedural steps?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class Grader:
    """
    Grades a completed support episode and returns a score in [0.0, 1.0].

    Scoring Dimensions:
    -------------------
    1. Resolution Accuracy (50%):
        - Correct terminal action matches expected_resolution → 1.0
        - Partially correct (right intent, wrong data) → 0.5
        - Wrong action taken → 0.0

    2. Efficiency Score (30%):
        - Agent used optimal number of steps → 1.0
        - Each extra step beyond optimal reduces score by 0.1 (min 0.0)

    3. Process Compliance (20%):
        - Agent followed required intermediate steps (e.g., lookup before refund)
        - Identity verification done on hard task → 1.0
        - Skipped required steps → proportional penalty

    Usage:
        grader = Grader(task)
        score = grader.grade(
            actions_taken=["lookup_user", "calculate_refund", "issue_refund", "close_ticket"],
            final_obs=last_observation,
        )
        # score: 0.0 – 1.0
    """

    ACCURACY_WEIGHT = 0.50
    EFFICIENCY_WEIGHT = 0.30
    COMPLIANCE_WEIGHT = 0.20

    def __init__(self, task: Dict[str, Any]) -> None:
        """
        Args:
            task: The task dict returned by data_generator.generate_task().
        """
        self.task = task
        self.difficulty = task.get("difficulty", "easy")
        self.expected_resolution = task.get("expected_resolution", "")
        self.correct_actions: List[str] = task.get("correct_actions", [])
        self.optimal_steps = len(self.correct_actions)

    def grade(
        self,
        actions_taken: List[str],
        final_obs_done: bool,
        final_obs_success: bool,
        total_reward: float,
        steps_taken: int,
        identity_verified: bool = False,
        refund_calculated: bool = False,
    ) -> float:
        """
        Compute the final grade for a completed episode.

        Args:
            actions_taken: List of action_type strings (e.g. ['lookup_user', 'issue_refund']).
            final_obs_done: Whether the episode ended with done=True.
            final_obs_success: Whether the terminal action was successful.
            total_reward: Cumulative reward over the episode.
            steps_taken: Total number of steps taken by the agent.
            identity_verified: Whether the agent ran VERIFY_IDENTITY (relevant for hard).
            refund_calculated: Whether the agent ran CALCULATE_REFUND before issuing.

        Returns:
            Float in [0.0, 1.0].
        """
        accuracy_score = self._score_accuracy(
            actions_taken, final_obs_done, final_obs_success, identity_verified
        )
        efficiency_score = self._score_efficiency(steps_taken)
        compliance_score = self._score_compliance(
            actions_taken, identity_verified, refund_calculated
        )

        if accuracy_score == 0.0:
            final_score = 0.0
        else:
            final_score = (
                accuracy_score * self.ACCURACY_WEIGHT
                + efficiency_score * self.EFFICIENCY_WEIGHT
                + compliance_score * self.COMPLIANCE_WEIGHT
            )
        return round(min(max(final_score, 0.0), 1.0), 4)

    # ------------------------------------------------------------------
    # Scoring sub-components
    # ------------------------------------------------------------------

    def _score_accuracy(
        self,
        actions_taken: List[str],
        done: bool,
        success: bool,
        identity_verified: bool,
    ) -> float:
        """
        Did the agent perform the correct terminal resolution?

        Returns:
            1.0 → Correct resolution.
            0.5 → Partially correct (e.g., issued refund but skipped verification on hard).
            0.0 → Wrong resolution or episode not completed.
        """
        if not done:
            return 0.0  # Episode never terminated

        terminal_action = actions_taken[-1] if actions_taken else ""
        expected = self.expected_resolution

        if expected == "issue_refund":
            if "issue_refund" in actions_taken:
                # Hard task: must also have verified identity
                if self.difficulty == "hard" and not identity_verified:
                    return 0.5
                return 1.0
            return 0.0

        elif expected == "escalate":
            if "escalate" in actions_taken:
                # Extra points if verification was done before escalating
                if identity_verified:
                    return 1.0
                return 0.75  # Escalated but without verification
            # Wrong choices
            if "issue_refund" in actions_taken:
                return 0.0  # Very wrong — refunded inconsistent data
            return 0.2

        elif expected == "all_resolved":
            # Must have closed all tickets via appropriate actions
            close_count = actions_taken.count("close_ticket")
            refund_done = "issue_refund" in actions_taken
            upgrade_done = "update_subscription" in actions_taken
            # Medium task expects: upgrade + refund + 3 closes
            if refund_done and upgrade_done and close_count >= 1 and success:
                return 1.0
            elif (refund_done or upgrade_done) and success:
                return 0.7
            elif success:
                return 0.5
            return 0.2

        # Fallback
        return 1.0 if success else 0.0

    def _score_efficiency(self, steps_taken: int) -> float:
        """
        How efficiently did the agent operate?

        Score = max(0, 1 - 0.1 * extra_steps)
        where extra_steps = max(0, steps_taken - optimal_steps)
        """
        if self.optimal_steps == 0:
            return 1.0
        extra_steps = max(0, steps_taken - self.optimal_steps)
        score = max(0.0, 1.0 - 0.1 * extra_steps)
        return round(score, 4)

    def _score_compliance(
        self,
        actions_taken: List[str],
        identity_verified: bool,
        refund_calculated: bool,
    ) -> float:
        """
        Did the agent follow correct procedural steps?

        Checks:
          - lookup_user before any destructive action
          - verify_identity on hard tasks
          - calculate_refund before issue_refund (bonus)
        """
        score = 1.0
        penalties = []

        # Rule 1: Must look up user before taking financial action
        financial_actions = {"issue_refund", "update_subscription"}
        first_financial_idx = next(
            (i for i, a in enumerate(actions_taken) if a in financial_actions), None
        )
        first_lookup_idx = next(
            (i for i, a in enumerate(actions_taken) if a == "lookup_user"), None
        )
        if first_financial_idx is not None:
            if first_lookup_idx is None or first_lookup_idx > first_financial_idx:
                penalties.append(0.4)  # Large deduction

        # Rule 2: Hard task requires verify_identity
        if self.difficulty == "hard":
            if not identity_verified:
                penalties.append(0.5)  # Major deduction
            elif "issue_refund" in actions_taken:
                # Verified but still issued refund (wrong on hard)
                penalties.append(0.3)

        # Rule 3: Refund should be calculated before being issued (optional but good)
        if "issue_refund" in actions_taken and not refund_calculated:
            penalties.append(0.1)  # Minor deduction

        # Apply penalties
        for p in penalties:
            score -= p

        return max(0.0, round(score, 4))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def describe(score: float) -> str:
        """Return a human-readable label for the score."""
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.75:
            return "Good"
        elif score >= 0.5:
            return "Partial"
        elif score >= 0.25:
            return "Poor"
        else:
            return "Failed"
