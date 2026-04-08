from __future__ import annotations

from .models import SupportTicket, clamp_open_unit_interval


def classification_reward(correct: bool) -> float:
    base = 0.72 if correct else 0.08
    return clamp_open_unit_interval(base)


def reply_reward(ticket: SupportTicket, reply: str) -> float:
    lowered = reply.lower()
    hits = sum(1 for term in ticket.must_include_reply_terms if term.lower() in lowered)
    coverage = hits / max(len(ticket.must_include_reply_terms), 1)
    # Strong partial-credit shaping.
    score = 0.10 + (0.75 * coverage)
    return clamp_open_unit_interval(score)


def close_reward(ready_to_close: bool) -> float:
    return clamp_open_unit_interval(0.88 if ready_to_close else 0.05)


def invalid_action_penalty() -> float:
    return clamp_open_unit_interval(0.02)
