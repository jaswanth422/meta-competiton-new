from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

EPSILON = 0.001
MAX_SCORE = 0.999


class SupportTicket(BaseModel):
    ticket_id: str
    customer_message: str
    expected_label: Literal["refund", "shipping", "billing", "technical"]
    must_include_reply_terms: list[str] = Field(default_factory=list)


class SupportAction(BaseModel):
    action_type: Literal["classify", "reply", "close"]
    ticket_id: str
    label: Optional[str] = None
    message: Optional[str] = None


class TicketProgress(BaseModel):
    classified: bool = False
    replied: bool = False
    closed: bool = False
    chosen_label: Optional[str] = None
    last_reply: Optional[str] = None


class SupportObservation(BaseModel):
    benchmark: str
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    current_ticket: Optional[SupportTicket] = None
    completed_tickets: int
    total_tickets: int
    steps_taken: int
    remaining_steps: int
    last_error: Optional[str] = None


class StepInfo(BaseModel):
    reason: str


class StepResult(BaseModel):
    observation: SupportObservation
    reward: float
    done: bool
    info: StepInfo


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class GradeResult(BaseModel):
    task_id: str
    score: float
    details: str


def clamp_open_unit_interval(value: float) -> float:
    return min(max(value, EPSILON), MAX_SCORE)
