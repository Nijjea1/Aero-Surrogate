"""Seam 3: the run log. Pranoy writes, Paramveer reads.

One JSONL file per run in runs/logs/. Append-only so a chart can stream a run
that is still going.
"""

from __future__ import annotations

from ..contracts import IterationRecord


class RunLog:
    def __init__(self, root: str = "runs/logs"):
        raise NotImplementedError("TODO Pranoy")

    def append(self, record: IterationRecord) -> None:
        raise NotImplementedError

    def read(self, run_id: str) -> list[dict]:
        raise NotImplementedError

    def list_runs(self) -> list[dict]:
        raise NotImplementedError
