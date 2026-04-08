"""Tests for the FastAPI server endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from server.app import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestResetEndpoint:
    def test_reset_easy(self):
        resp = client.post("/reset", json={"difficulty": "easy", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert data["done"] == False
        assert len(data["tickets"]) == 1
        assert data["step_count"] == 0

    def test_reset_medium(self):
        resp = client.post("/reset", json={"difficulty": "medium", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tickets"]) == 3

    def test_reset_hard(self):
        resp = client.post("/reset", json={"difficulty": "hard", "seed": 42})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tickets"]) == 1


class TestStateEndpoint:
    def test_state_after_reset(self):
        client.post("/reset", json={"difficulty": "easy", "seed": 1})
        resp = client.get("/state")
        assert resp.status_code == 200
        data = resp.json()
        assert data["difficulty"] == "easy"
        assert data["step_count"] == 0


class TestStepEndpoint:
    def test_lookup_user_step(self):
        # Reset first
        reset_resp = client.post("/reset", json={"difficulty": "easy", "seed": 42})
        ticket = reset_resp.json()["tickets"][0]
        uid = ticket["user_id"]

        resp = client.post("/step", json={
            "action": {"action_type": "lookup_user", "user_id": uid}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["db_record"] is not None
        assert data["reward"] == 0.2

    def test_invalid_action_type_handled(self):
        client.post("/reset", json={"difficulty": "easy", "seed": 42})
        resp = client.post("/step", json={
            "action": {"action_type": "nop"}
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["reward"] == -0.1

    def test_full_easy_resolution_via_api(self):
        reset_resp = client.post("/reset", json={"difficulty": "easy", "seed": 42})
        uid = reset_resp.json()["tickets"][0]["user_id"]

        # Lookup
        client.post("/step", json={"action": {"action_type": "lookup_user", "user_id": uid}})
        # Calculate refund
        client.post("/step", json={"action": {"action_type": "calculate_refund", "user_id": uid}})
        # Issue refund
        final_resp = client.post("/step", json={"action": {"action_type": "issue_refund", "user_id": uid}})

        data = final_resp.json()
        assert data["done"] == True
        assert data["success"] == True
        assert data["reward"] >= 0.5


class TestGradeEndpoint:
    def test_grade_after_episode(self):
        # Run an episode first
        reset_resp = client.post("/reset", json={"difficulty": "easy", "seed": 42})
        uid = reset_resp.json()["tickets"][0]["user_id"]
        client.post("/step", json={"action": {"action_type": "lookup_user", "user_id": uid}})
        client.post("/step", json={"action": {"action_type": "calculate_refund", "user_id": uid}})
        client.post("/step", json={"action": {"action_type": "issue_refund", "user_id": uid}})

        grade_resp = client.post("/grade", json={
            "actions_taken": ["lookup_user", "calculate_refund", "issue_refund"],
            "final_obs_done": True,
            "final_obs_success": True,
            "total_reward": 0.8,
            "steps_taken": 3,
            "identity_verified": False,
            "refund_calculated": True,
        })
        assert grade_resp.status_code == 200
        data = grade_resp.json()
        assert "score" in data
        assert 0.0 <= data["score"] <= 1.0
        assert "label" in data
