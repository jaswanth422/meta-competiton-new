from __future__ import annotations

from fastapi import FastAPI

from support_ops_env.models import ResetRequest, StepResult, SupportAction, SupportObservation
from support_ops_env.support_ops_environment import SupportOpsEnvironment

app = FastAPI(title="Support Ops OpenEnv", version="0.1.0")
ENV = SupportOpsEnvironment()


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Support Ops OpenEnv",
        "status": "ok",
        "health": "/health",
        "reset": "/reset",
        "state": "/state",
        "step": "/step",
        "grade": "/grade",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/reset", response_model=SupportObservation)
def reset(req: ResetRequest | None = None) -> SupportObservation:
    task_id = req.task_id if req else None
    return ENV.reset(task_id)


@app.post("/step", response_model=StepResult)
def step(action: SupportAction) -> StepResult:
    return ENV.step(action)


@app.get("/state", response_model=SupportObservation)
def state() -> SupportObservation:
    return ENV.state()


@app.post("/grade")
def grade():
    return ENV.grade()
