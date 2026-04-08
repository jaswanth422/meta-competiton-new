from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from support_ops_env.models import clamp_open_unit_interval
from support_ops_env.support_ops_environment import SupportOpsEnvironment
from support_ops_env.task_bank import list_task_ids

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
BENCHMARK = os.getenv("BENCHMARK", "support_ops_v1")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def _heuristic_label(message: str) -> str:
    lowered = message.lower()
    if "refund" in lowered or "charged" in lowered:
        return "refund"
    if "tracking" in lowered or "delivered" in lowered or "package" in lowered:
        return "shipping"
    if "invoice" in lowered or "pricing" in lowered or "charged" in lowered:
        return "billing"
    return "technical"


def _heuristic_reply(label: str) -> str:
    if label == "refund":
        return "We are processing your refund for the charged twice issue. Confirmation in 3-5 business days."
    if label == "shipping":
        return "We will trace the carrier tracking and investigate, then share an update within 24 hours."
    if label == "billing":
        return "Our billing team will review your invoice and apply an adjustment where required."
    return "Please reset password and clear cache. If issue persists, support will assist directly."


def _model_decide(client: OpenAI, message: str) -> tuple[str, str]:
    prompt = (
        "You are a support triage assistant. "
        "Return ONLY compact JSON with keys label and reply. "
        "label must be one of: refund, shipping, billing, technical. "
        "reply must be a concise professional response.\n"
        f"Customer message: {message}"
    )
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0,
        messages=[
            {"role": "system", "content": "Output valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (completion.choices[0].message.content or "").strip()
    parsed = json.loads(content)
    label = str(parsed.get("label", "")).strip().lower()
    reply = str(parsed.get("reply", "")).strip()
    if label not in {"refund", "shipping", "billing", "technical"}:
        label = _heuristic_label(message)
    if not reply:
        reply = _heuristic_reply(label)
    return label, reply


def run_task(task_id: str, client: OpenAI) -> float:
    env = SupportOpsEnvironment()
    obs = env.reset(task_id)

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    rewards: list[float] = []
    step_no = 0
    done = False

    while not done and obs.current_ticket is not None:
        ticket = obs.current_ticket
        step_no += 1
        try:
            label, reply = _model_decide(client, ticket.customer_message)
        except Exception:
            # Fallback keeps script robust while still attempting proxy calls first.
            label = _heuristic_label(ticket.customer_message)
            reply = _heuristic_reply(label)
        res = env.step(
            action={
                "action_type": "classify",
                "ticket_id": ticket.ticket_id,
                "label": label,
            }
        )
        rewards.append(res.reward)
        log_step(step_no, f"classify('{ticket.ticket_id}','{label}')", res.reward, res.done, obs.last_error)
        done = res.done

        if done:
            break

        step_no += 1
        res = env.step(
            action={
                "action_type": "reply",
                "ticket_id": ticket.ticket_id,
                "message": reply,
            }
        )
        rewards.append(res.reward)
        log_step(step_no, f"reply('{ticket.ticket_id}')", res.reward, res.done, obs.last_error)
        done = res.done

        if done:
            break

        step_no += 1
        res = env.step(action={"action_type": "close", "ticket_id": ticket.ticket_id})
        rewards.append(res.reward)
        log_step(step_no, f"close('{ticket.ticket_id}')", res.reward, res.done, obs.last_error)
        done = res.done
        obs = res.observation

    grade = env.grade()
    score = clamp_open_unit_interval(grade.score)
    success = score >= 0.5
    log_end(success=success, steps=step_no, score=score, rewards=rewards)
    return score


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required. Set it in environment or Space secrets.")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    scores = [run_task(task_id, client) for task_id in list_task_ids()]
    avg = clamp_open_unit_interval(sum(scores) / len(scores))
    print(f"baseline_average_score={avg:.3f}", flush=True)


if __name__ == "__main__":
    main()
