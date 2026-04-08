"""
Mock Data Generator for Support Ticket & DB Reconciliation Environment.

Generates realistic support tickets and internal database records for three
difficulty levels. Data is intentionally nuanced:
  - easy:   Single ticket, consistent user data, straightforward resolution.
  - medium: Multiple tickets, one requires refund calculation with policy logic.
  - hard:   Inconsistent data between ticket and DB (e.g., name typo, wrong email)
            requiring the agent to perform identity verification before acting.
"""

from __future__ import annotations

import random
import string
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Constants / Policy definitions
# ---------------------------------------------------------------------------

SUBSCRIPTION_PLANS = {
    "free": {"price": 0.0, "features": ["5 projects", "1 GB storage"]},
    "starter": {"price": 9.99, "features": ["20 projects", "10 GB storage", "Email support"]},
    "pro": {"price": 29.99, "features": ["Unlimited projects", "100 GB storage", "Priority support"]},
    "enterprise": {"price": 99.99, "features": ["Unlimited everything", "SSO", "SLA", "Dedicated support"]},
}

# Refund policy: days since purchase → eligibility
REFUND_POLICY_DAYS = 30  # Full refund within 30 days
PARTIAL_REFUND_DAYS = 60  # 50% refund within 60 days

ISSUE_TYPES = [
    "billing",
    "subscription",
    "technical",
    "account_access",
    "refund_request",
    "data_loss",
    "feature_request",
    "escalation_request",
]

FIRST_NAMES = [
    "James", "Maria", "Wei", "Priya", "Carlos", "Fatima",
    "Liam", "Olivia", "Noah", "Emma", "Aiden", "Sophia",
    "Lucas", "Mia", "Ethan", "Isabella", "Mason", "Ava",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones",
    "Garcia", "Martinez", "Davis", "Miller", "Wilson",
    "Patel", "Kim", "Nguyen", "Chen", "Kumar",
]

DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "company.io", "protonmail.com"]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _uid() -> str:
    return "U-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=8))


def _oid() -> str:
    return "ORD-" + "".join(random.choices(string.digits, k=8))


def _tid() -> str:
    return "TKT-" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _days_ago(n: int) -> str:
    return (datetime.utcnow() - timedelta(days=n)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _introduce_typo(name: str) -> str:
    """Introduce a realistic typo in a name (swap two chars or drop a letter)."""
    if len(name) < 3:
        return name
    idx = random.randint(1, len(name) - 2)
    chars = list(name)
    variant = random.choice(["swap", "drop", "replace"])
    if variant == "swap" and idx + 1 < len(name):
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    elif variant == "drop":
        chars.pop(idx)
    else:
        chars[idx] = random.choice(string.ascii_lowercase)
    return "".join(chars)


def _make_user_db_record(
    user_id: str,
    first_name: str,
    last_name: str,
    plan: str,
    purchase_days_ago: int,
    email: str,
) -> Dict[str, Any]:
    """Build a realistic internal DB record for a user."""
    purchase_date = _days_ago(purchase_days_ago)
    order_id = _oid()
    price = SUBSCRIPTION_PLANS[plan]["price"]
    return {
        "user_id": user_id,
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "plan": plan,
        "plan_price": price,
        "order_id": order_id,
        "purchase_date": purchase_date,
        "purchase_days_ago": purchase_days_ago,
        "account_status": "active",
        "total_payments": round(price * random.randint(1, 12), 2),
        "payment_method": random.choice(["credit_card", "paypal", "bank_transfer"]),
        "created_at": _days_ago(purchase_days_ago + random.randint(0, 30)),
    }


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------


def generate_easy_task() -> Dict[str, Any]:
    """
    Easy: Single ticket where the user exists in DB and wants a simple refund.
    The agent should: lookup user → (optionally) calculate refund → issue refund → close ticket.
    """
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    email = f"{first.lower()}.{last.lower()}@{random.choice(DOMAINS)}"
    user_id = _uid()
    plan = random.choice(["starter", "pro"])
    purchase_days_ago = random.randint(5, 25)  # Within full-refund window

    db_record = _make_user_db_record(user_id, first, last, plan, purchase_days_ago, email)

    ticket_body = (
        f"Hi, I recently purchased the {plan.capitalize()} plan {purchase_days_ago} days ago "
        f"but it's not what I expected. I'd like to request a full refund. "
        f"My email is {email} and my user ID is {user_id}."
    )
    ticket = {
        "ticket_id": _tid(),
        "user_id": user_id,
        "email": email,
        "subject": "Refund request for subscription",
        "body": ticket_body,
        "issue_type": "refund_request",
        "priority": "medium",
        "created_at": _days_ago(1),
        "status": "open",
    }

    correct_actions = ["lookup_user", "calculate_refund", "issue_refund", "close_ticket"]
    expected_resolution = "issue_refund"

    return {
        "difficulty": "easy",
        "tickets": [ticket],
        "database": {user_id: db_record},
        "correct_actions": correct_actions,
        "expected_resolution": expected_resolution,
        "task_description": (
            f"A single user ({first} {last}) purchased the {plan.capitalize()} plan "
            f"{purchase_days_ago} days ago and requests a full refund. "
            "Verify their identity in the DB and process the refund."
        ),
    }


def generate_medium_task() -> Dict[str, Any]:
    """
    Medium: Three tickets. One is a simple subscription upgrade, one is a borderline
    refund (within partial-refund window requiring calculation), and one is a standard
    account issue that just needs acknowledgement and closure.
    """
    users = []
    tickets = []
    database: Dict[str, Any] = {}

    # --- Ticket 1: Subscription upgrade ---
    f1, l1 = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
    e1 = f"{f1.lower()}.{l1.lower()}@{random.choice(DOMAINS)}"
    uid1 = _uid()
    plan1 = "starter"
    db1 = _make_user_db_record(uid1, f1, l1, plan1, random.randint(60, 180), e1)
    database[uid1] = db1
    tickets.append({
        "ticket_id": _tid(),
        "user_id": uid1,
        "email": e1,
        "subject": "Upgrade my plan to Pro",
        "body": (
            f"Hello, I'm {f1} {l1} (user ID: {uid1}). "
            "I've been using the Starter plan for a while and would like to upgrade to Pro. "
            "Please help me complete the upgrade."
        ),
        "issue_type": "subscription",
        "priority": "low",
        "created_at": _days_ago(2),
        "status": "open",
    })

    # --- Ticket 2: Partial refund (35-55 days ago, borderline) ---
    f2, l2 = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
    e2 = f"{f2.lower()}.{l2.lower()}@{random.choice(DOMAINS)}"
    uid2 = _uid()
    plan2 = "pro"
    days2 = random.randint(31, 55)  # In the partial-refund window
    db2 = _make_user_db_record(uid2, f2, l2, plan2, days2, e2)
    database[uid2] = db2
    tickets.append({
        "ticket_id": _tid(),
        "user_id": uid2,
        "email": e2,
        "subject": "Requesting refund — cancelled subscription",
        "body": (
            f"Hi support, I am {f2} {l2} and I purchased the Pro plan {days2} days ago. "
            "I've cancelled my account but I haven't received a refund yet. "
            "I understand there may be a partial refund based on your policy. "
            f"Please process whatever I'm eligible for. My email: {e2}."
        ),
        "issue_type": "refund_request",
        "priority": "high",
        "created_at": _days_ago(1),
        "status": "open",
    })

    # --- Ticket 3: Basic technical issue, just close ---
    f3, l3 = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
    e3 = f"{f3.lower()}.{l3.lower()}@{random.choice(DOMAINS)}"
    uid3 = _uid()
    db3 = _make_user_db_record(uid3, f3, l3, "free", random.randint(10, 90), e3)
    database[uid3] = db3
    tickets.append({
        "ticket_id": _tid(),
        "user_id": uid3,
        "email": e3,
        "subject": "Can't find the export button",
        "body": (
            f"Hi, I'm {f3} {l3}. I can't seem to find the data export button in the dashboard. "
            "It used to be under Settings > Data. Is this a bug or was the UI changed?"
        ),
        "issue_type": "technical",
        "priority": "low",
        "created_at": _days_ago(3),
        "status": "open",
    })

    correct_actions = [
        "lookup_user",  # for uid2 (refund ticket is priority)
        "calculate_refund",
        "issue_refund",
        "lookup_user",  # for uid1 (upgrade)
        "update_subscription",
        "close_ticket",  # uid3 technical
        "close_ticket",
        "close_ticket",
    ]
    expected_resolution = "all_resolved"

    return {
        "difficulty": "medium",
        "tickets": tickets,
        "database": database,
        "correct_actions": correct_actions,
        "expected_resolution": expected_resolution,
        "task_description": (
            "Three tickets: (1) a subscription upgrade request, "
            f"(2) a refund request {days2} days after purchase (partial refund applies), "
            "and (3) a simple UI navigation question. Resolve all three efficiently."
        ),
    }


def generate_hard_task() -> Dict[str, Any]:
    """
    Hard: A single ticket with deliberate inconsistencies between the ticket's
    claims and the actual DB record. The agent MUST run VERIFY_IDENTITY before
    acting, or it will act on incorrect data and receive a penalty.

    Inconsistencies injected:
    - Ticket contains a slight name typo (e.g., 'Smyth' instead of 'Smith')
    - Ticket email domain doesn't match DB email
    - Order ID in ticket doesn't match the DB record
    The agent must detect these mismatches and either escalate or cautiously proceed.
    """
    first = random.choice(FIRST_NAMES)
    last = random.choice(LAST_NAMES)
    email_real = f"{first.lower()}.{last.lower()}@{random.choice(DOMAINS)}"
    user_id = _uid()
    plan = random.choice(["pro", "enterprise"])
    purchase_days_ago = random.randint(10, 25)

    db_record = _make_user_db_record(
        user_id, first, last, plan, purchase_days_ago, email_real
    )
    real_order_id = db_record["order_id"]

    # Inject mismatches into the ticket
    typo_last = _introduce_typo(last)
    fake_domain = random.choice([d for d in DOMAINS if d not in email_real])
    email_ticket = f"{first.lower()}.{typo_last.lower()}@{fake_domain}"
    fake_order_id = _oid()  # Different from real order

    ticket_body = (
        f"Hello, I'm {first} {typo_last} and I'm writing regarding order {fake_order_id}. "
        f"I purchased the {plan.capitalize()} plan about {purchase_days_ago} days ago "
        f"using email {email_ticket}. I was double-charged this month and need an immediate refund. "
        f"My user ID should be {user_id}. Please resolve ASAP."
    )

    ticket = {
        "ticket_id": _tid(),
        "user_id": user_id,
        "email_in_ticket": email_ticket,
        "email_real": email_real,  # For environment grading only
        "subject": "URGENT: Double charge on my account — need refund NOW",
        "body": ticket_body,
        "issue_type": "billing",
        "priority": "urgent",
        "created_at": _days_ago(0),
        "status": "open",
        "injected_inconsistencies": {
            "name_mismatch": f"'{typo_last}' vs DB '{last}'",
            "email_mismatch": f"'{email_ticket}' vs DB '{email_real}'",
            "order_mismatch": f"'{fake_order_id}' vs DB '{real_order_id}'",
        },
    }

    correct_actions = [
        "lookup_user",
        "verify_identity",  # Required — must detect mismatch
        "escalate",         # Correct because data is inconsistent and urgent
    ]
    expected_resolution = "escalate"

    return {
        "difficulty": "hard",
        "tickets": [ticket],
        "database": {user_id: db_record},
        "correct_actions": correct_actions,
        "expected_resolution": expected_resolution,
        "task_description": (
            f"URGENT ticket from {first} {typo_last} (user ID: {user_id}). "
            "Claims double charge but the ticket contains name typo, wrong email, "
            "and an unrecognised order ID. The agent must verify identity first "
            "to detect inconsistencies before taking any financial action."
        ),
        "inconsistencies": ticket["injected_inconsistencies"],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


TASK_GENERATORS = {
    "easy": generate_easy_task,
    "medium": generate_medium_task,
    "hard": generate_hard_task,
}


def generate_task(difficulty: str = "easy", seed: int | None = None) -> Dict[str, Any]:
    """
    Generate a task scenario for the given difficulty.

    Args:
        difficulty: One of 'easy', 'medium', 'hard'.
        seed: Optional random seed for reproducibility.

    Returns:
        A dictionary with keys: difficulty, tickets, database,
        correct_actions, expected_resolution, task_description.
    """
    if seed is not None:
        random.seed(seed)
    if difficulty not in TASK_GENERATORS:
        raise ValueError(f"Unknown difficulty '{difficulty}'. Choose from: {list(TASK_GENERATORS)}")
    return TASK_GENERATORS[difficulty]()
