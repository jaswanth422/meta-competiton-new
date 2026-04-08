from __future__ import annotations

from dataclasses import dataclass

from .graders import grade_easy_refund, grade_hard_multi_issue, grade_medium_shipping
from .models import (
    GradeResult,
    StepInfo,
    StepResult,
    SupportAction,
    SupportObservation,
    TicketProgress,
    clamp_open_unit_interval,
)
from .reward import classification_reward, close_reward, invalid_action_penalty, reply_reward
from .task_bank import TaskDefinition, get_task, list_task_ids

MAX_STEPS = 30
BENCHMARK_NAME = "support_ops_v1"


@dataclass
class RuntimeState:
    task: TaskDefinition
    progress: dict[str, TicketProgress]
    cursor: int = 0
    steps_taken: int = 0
    done: bool = False
    last_error: str | None = None


class SupportOpsEnvironment:
    def __init__(self) -> None:
        first_task = get_task(list_task_ids()[0])
        self._state = self._init_state(first_task)

    def _init_state(self, task: TaskDefinition) -> RuntimeState:
        progress = {ticket.ticket_id: TicketProgress() for ticket in task.tickets}
        return RuntimeState(task=task, progress=progress)

    def reset(self, task_id: str | None = None) -> SupportObservation:
        chosen_task = get_task(task_id) if task_id else get_task(list_task_ids()[0])
        self._state = self._init_state(chosen_task)
        return self.state()

    def _current_ticket(self):
        if self._state.cursor >= len(self._state.task.tickets):
            return None
        return self._state.task.tickets[self._state.cursor]

    def state(self) -> SupportObservation:
        current = self._current_ticket()
        completed = sum(1 for p in self._state.progress.values() if p.closed)
        return SupportObservation(
            benchmark=BENCHMARK_NAME,
            task_id=self._state.task.task_id,
            difficulty=self._state.task.difficulty,
            current_ticket=current,
            completed_tickets=completed,
            total_tickets=len(self._state.task.tickets),
            steps_taken=self._state.steps_taken,
            remaining_steps=max(0, MAX_STEPS - self._state.steps_taken),
            last_error=self._state.last_error,
        )

    def _finish_if_needed(self) -> None:
        all_closed = all(item.closed for item in self._state.progress.values())
        if all_closed or self._state.steps_taken >= MAX_STEPS:
            self._state.done = True

    def _validate_ticket(self, ticket_id: str) -> tuple[bool, str | None]:
        current = self._current_ticket()
        if current is None:
            return False, "No active ticket remaining."
        if ticket_id != current.ticket_id:
            return False, f"Only current ticket {current.ticket_id} may be edited now."
        return True, None

    def step(self, action: SupportAction | dict) -> StepResult:
        action = SupportAction.model_validate(action)
        if self._state.done:
            return StepResult(
                observation=self.state(),
                reward=invalid_action_penalty(),
                done=True,
                info=StepInfo(reason="Episode already finished."),
            )

        self._state.steps_taken += 1
        valid, err = self._validate_ticket(action.ticket_id)
        if not valid:
            self._state.last_error = err
            self._finish_if_needed()
            return StepResult(
                observation=self.state(),
                reward=invalid_action_penalty(),
                done=self._state.done,
                info=StepInfo(reason=err or "Invalid ticket selection."),
            )

        ticket = self._current_ticket()
        assert ticket is not None
        progress = self._state.progress[ticket.ticket_id]

        reward = invalid_action_penalty()
        reason = "Invalid action."

        if action.action_type == "classify":
            if not action.label:
                reason = "Missing label for classify action."
            else:
                progress.classified = True
                progress.chosen_label = action.label
                is_correct = action.label == ticket.expected_label
                reward = classification_reward(is_correct)
                reason = "Classification accepted."
                self._state.last_error = None

        elif action.action_type == "reply":
            if not action.message:
                reason = "Missing message for reply action."
            else:
                progress.replied = True
                progress.last_reply = action.message
                reward = reply_reward(ticket, action.message)
                reason = "Reply accepted."
                self._state.last_error = None

        elif action.action_type == "close":
            ready = progress.classified and progress.replied
            progress.closed = ready
            reward = close_reward(ready)
            reason = "Ticket closed." if ready else "Ticket cannot close before classify+reply."
            if ready:
                self._state.cursor += 1
                self._state.last_error = None

        self._finish_if_needed()
        return StepResult(
            observation=self.state(),
            reward=clamp_open_unit_interval(reward),
            done=self._state.done,
            info=StepInfo(reason=reason),
        )

    def grade(self) -> GradeResult:
        outcomes: list[tuple[bool, bool, bool]] = []
        for ticket in self._state.task.tickets:
            progress = self._state.progress[ticket.ticket_id]
            correct_label = progress.chosen_label == ticket.expected_label
            adequate_reply = bool(progress.last_reply) and all(
                term.lower() in progress.last_reply.lower() for term in ticket.must_include_reply_terms
            )
            outcomes.append((correct_label, adequate_reply, progress.closed))

        task_id = self._state.task.task_id
        if task_id == "easy_refund":
            return grade_easy_refund(*outcomes[0])
        if task_id == "medium_shipping":
            return grade_medium_shipping(outcomes)
        return grade_hard_multi_issue(outcomes)
