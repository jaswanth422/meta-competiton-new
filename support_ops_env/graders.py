from __future__ import annotations

from .models import GradeResult, clamp_open_unit_interval


def _coverage_ratio(flags: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(1 for flag in flags if flag) / len(flags)


def _score_from_coverage(task_id: str, coverage: float, strictness: float) -> GradeResult:
    # Deterministic continuous score in (0,1). strictness weights harder tasks.
    raw = 0.05 + ((coverage ** strictness) * 0.90)
    score = clamp_open_unit_interval(raw)
    details = f"coverage={coverage:.3f}, strictness={strictness:.2f}"
    return GradeResult(task_id=task_id, score=score, details=details)


def grade_easy_refund(correct_label: bool, adequate_reply: bool, closed: bool) -> GradeResult:
    coverage = _coverage_ratio([correct_label, adequate_reply, closed])
    return _score_from_coverage("easy_refund", coverage, strictness=1.0)


def grade_medium_shipping(ticket_outcomes: list[tuple[bool, bool, bool]]) -> GradeResult:
    flat_flags: list[bool] = []
    for outcome in ticket_outcomes:
        flat_flags.extend(list(outcome))
    coverage = _coverage_ratio(flat_flags)
    return _score_from_coverage("medium_shipping", coverage, strictness=1.15)


def grade_hard_multi_issue(ticket_outcomes: list[tuple[bool, bool, bool]]) -> GradeResult:
    flat_flags: list[bool] = []
    for outcome in ticket_outcomes:
        flat_flags.extend(list(outcome))
    coverage = _coverage_ratio(flat_flags)
    return _score_from_coverage("hard_multi_issue", coverage, strictness=1.30)
