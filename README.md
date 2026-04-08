---
title: Support Ops OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
python_version: "3.11"
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Support Ops OpenEnv

Support Ops OpenEnv is a deterministic, real-world inspired environment where an agent handles customer support tickets through three core actions:

- Classify the ticket category.
- Draft a reply that addresses key facts.
- Close the ticket.

The environment is designed for reproducible evaluation with 3 tasks (`easy`, `medium`, `hard`) and deterministic graders that return scores strictly in `(0, 1)`.

## Why This Environment

Support triage is a common production workflow where quality depends on:

- Correct intent classification.
- Clear and policy-safe communication.
- Accurate resolution handling.

These are practical capabilities for evaluating task-oriented agents.

## Action Space

`SupportAction` contains:

- `action_type`: one of `classify`, `reply`, `close`.
- `ticket_id`: ticket identifier.
- `label`: required for `classify`.
- `message`: required for `reply`.

## Observation Space

`SupportObservation` includes:

- Current task metadata.
- Next pending ticket.
- Completed ticket count.
- Remaining steps.
- Last error message (if any).

## Reward Design

Rewards are dense and deterministic and are clamped to strict open bounds `(0, 1)`:

- Correct classification gives a higher reward than incorrect.
- Reply quality is scored by keyword coverage.
- Closing a fully handled ticket gives a high reward.
- Invalid or destructive actions are penalized to a low positive reward.

## Graders

Three deterministic task graders are included:

- `grade_easy_refund`
- `grade_medium_shipping`
- `grade_hard_multi_issue`

All graders return continuous scores strictly in `(0, 1)` and avoid binary `0.0`/`1.0` outputs.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --reload --port 7860
```

Then test endpoints:

```bash
curl -X POST http://127.0.0.1:7860/reset
curl http://127.0.0.1:7860/state
```

## Baseline Inference

The baseline script runs one episode per task and emits strict logs:

- `[START]`
- `[STEP]`
- `[END]`

Run:

```bash
python inference.py
```

## Docker

```bash
docker build -t support-ops-openenv .
docker run --rm -p 7860:7860 support-ops-openenv
```

## Project Layout

- `support_ops_env/`: environment logic, task bank, reward, graders, models.
- `support_ops_env/server/`: environment API wiring.
- `server/`: ASGI entrypoint.
- `inference.py`: deterministic baseline runner.
- `openenv.yaml`: environment metadata.
- `scripts/validate-submission.sh`: validation helper.
