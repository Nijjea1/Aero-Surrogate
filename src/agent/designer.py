"""The Nemotron design agent.  OWNER: Pranoy.  GATE: running Oct 17.

    python -m src.agent.designer --iterations 25 --dry-run

The uncertainty gate is the point of this file. When the surrogate is not
confident, reject the proposal and TELL THE AGENT WHY, so it adapts instead of
hammering the same region.

Build a --dry-run scripted proposer first so the loop can be tested before
Token Factory is wired up. The gate for Oct 17 is "the loop runs end to end",
not "the predictions are good".
"""

from __future__ import annotations

from ..contracts import UNCERTAINTY_THRESHOLD, IterationRecord

SYSTEM_PROMPT = """TODO Pranoy: you are an aerodynamicist minimising Cd...
- stay inside the bounds given
- if a proposal is REJECTED for low confidence, move back toward regions the
  model understands; do not repeat the rejected move
- change one or two parameters at a time so effects are attributable
- respond ONLY with JSON: {"params": {...}, "reasoning": "..."}"""


def evaluate_design(params: dict[str, float]) -> dict:
    """POST to the service's /evaluate."""
    raise NotImplementedError("TODO Pranoy")


def ask_nemotron(history: list[dict], api_key: str) -> dict:
    """One planning step via Token Factory. -> {"params": ..., "reasoning": ...}"""
    raise NotImplementedError("TODO Pranoy: Oct 16")


def scripted_proposer(history: list[dict]) -> dict:
    """Offline stand-in for Nemotron. Build this FIRST."""
    raise NotImplementedError("TODO Pranoy: Oct 14")


def run(iterations: int, run_id: str, dry_run: bool) -> str:
    """Propose -> evaluate -> gate -> log. Repeat."""
    raise NotImplementedError("TODO Pranoy: Oct 17")
