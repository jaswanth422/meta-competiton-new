from __future__ import annotations

import requests

from .models import GradeResult, StepResult, SupportAction, SupportObservation


class SupportOpsClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self, task_id: str | None = None) -> SupportObservation:
        resp = requests.post(f"{self.base_url}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        return SupportObservation.model_validate(resp.json())

    def step(self, action: SupportAction) -> StepResult:
        resp = requests.post(f"{self.base_url}/step", json=action.model_dump(), timeout=30)
        resp.raise_for_status()
        return StepResult.model_validate(resp.json())

    def state(self) -> SupportObservation:
        resp = requests.get(f"{self.base_url}/state", timeout=30)
        resp.raise_for_status()
        return SupportObservation.model_validate(resp.json())

    def grade(self) -> GradeResult:
        resp = requests.post(f"{self.base_url}/grade", timeout=30)
        resp.raise_for_status()
        return GradeResult.model_validate(resp.json())
