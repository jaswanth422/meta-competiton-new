from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import SupportTicket


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    tickets: list[SupportTicket]


TASKS: dict[str, TaskDefinition] = {
    "easy_refund": TaskDefinition(
        task_id="easy_refund",
        difficulty="easy",
        objective="Correctly process a simple refund request.",
        tickets=[
            SupportTicket(
                ticket_id="T-100",
                customer_message="I was charged twice for my order #55. Please refund one payment.",
                expected_label="refund",
                must_include_reply_terms=["refund", "charged twice", "3-5 business days"],
            )
        ],
    ),
    "medium_shipping": TaskDefinition(
        task_id="medium_shipping",
        difficulty="medium",
        objective="Handle two delayed-shipping tickets with accurate follow-up.",
        tickets=[
            SupportTicket(
                ticket_id="T-200",
                customer_message="My package says delivered but I never received it.",
                expected_label="shipping",
                must_include_reply_terms=["carrier", "trace", "24 hours"],
            ),
            SupportTicket(
                ticket_id="T-201",
                customer_message="Tracking has not moved in 5 days. Can you check?",
                expected_label="shipping",
                must_include_reply_terms=["tracking", "investigate", "update"],
            ),
        ],
    ),
    "hard_multi_issue": TaskDefinition(
        task_id="hard_multi_issue",
        difficulty="hard",
        objective="Resolve mixed billing and technical issues across multiple tickets.",
        tickets=[
            SupportTicket(
                ticket_id="T-300",
                customer_message="I cannot log in after the latest update.",
                expected_label="technical",
                must_include_reply_terms=["reset password", "cache", "support"],
            ),
            SupportTicket(
                ticket_id="T-301",
                customer_message="Your monthly plan charged me annual pricing.",
                expected_label="billing",
                must_include_reply_terms=["billing", "invoice", "adjustment"],
            ),
            SupportTicket(
                ticket_id="T-302",
                customer_message="Cancel and refund my last payment immediately.",
                expected_label="refund",
                must_include_reply_terms=["cancel", "refund", "confirmation"],
            ),
        ],
    ),
}


def list_task_ids() -> list[str]:
    return sorted(TASKS.keys())


def get_task(task_id: str) -> TaskDefinition:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return TASKS[task_id]
