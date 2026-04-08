from support_ops_env.support_ops_environment import SupportOpsEnvironment


def test_env_reset_step_and_grade() -> None:
    env = SupportOpsEnvironment()
    obs = env.reset("easy_refund")
    ticket = obs.current_ticket
    assert ticket is not None

    step1 = env.step(
        {
            "action_type": "classify",
            "ticket_id": ticket.ticket_id,
            "label": "refund",
        }
    )
    assert 0.0 < step1.reward < 1.0

    step2 = env.step(
        {
            "action_type": "reply",
            "ticket_id": ticket.ticket_id,
            "message": "Refund charged twice 3-5 business days",
        }
    )
    assert 0.0 < step2.reward < 1.0

    step3 = env.step({"action_type": "close", "ticket_id": ticket.ticket_id})
    assert step3.done is True

    grade = env.grade()
    assert 0.0 < grade.score < 1.0
