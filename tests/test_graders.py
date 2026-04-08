from support_ops_env.graders import (
    grade_easy_refund,
    grade_hard_multi_issue,
    grade_medium_shipping,
)


def test_easy_grader_score_in_open_interval() -> None:
    score = grade_easy_refund(True, True, True).score
    assert 0.0 < score < 1.0


def test_medium_grader_score_in_open_interval() -> None:
    score = grade_medium_shipping([(True, True, True), (False, True, True)]).score
    assert 0.0 < score < 1.0


def test_hard_grader_score_in_open_interval() -> None:
    score = grade_hard_multi_issue([(True, True, True), (True, False, True), (False, False, True)]).score
    assert 0.0 < score < 1.0
