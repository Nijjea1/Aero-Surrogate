"""Seam 3: the run log. Pranoy writes, Paramveer reads.

One JSONL file per run in runs/logs/. Append-only so a chart can stream a run
that is still going.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from threading import RLock

from ..contracts import IterationRecord

_LOCK = RLock()
_RUN_ID = re.compile(r"[A-Za-z0-9_-]{1,128}")
_RESERVED = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


class RunLog:
    def __init__(self, root: str = "runs/logs"):
        self.root = Path(root).resolve()

    def _path(self, run_id: str) -> Path:
        if (not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id)
                or run_id.upper() in _RESERVED):
            raise ValueError("run_id must contain 1-128 letters, digits, underscores or hyphens")
        path = self.root / f"{run_id}.jsonl"
        if path.resolve().parent != self.root:
            raise ValueError("run log path escapes its root")
        return path

    @staticmethod
    def _decode(line: str, run_id: str, line_number: int) -> dict:
        try:
            record = json.loads(line)
            # Also catches NaN, Infinity and overflow (e.g. 1e999).
            json.dumps(record, allow_nan=False)
            if not isinstance(record, dict) or record.get("run_id") != run_id:
                raise ValueError("record must be an object with the matching run_id")
            return record
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Malformed run log {run_id}, line {line_number}: {exc}") from exc

    def _read_path(self, path: Path, run_id: str) -> list[dict]:
        try:
            data = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Malformed run log {run_id}: invalid UTF-8 storage") from exc
        lines = data.splitlines(keepends=True)
        records = []
        for number, line in enumerate(lines, 1):
            # An interrupted append is not a committed JSONL record. Completed
            # lines are always checked, including blank or malformed lines.
            if not line.endswith("\n"):
                break
            records.append(self._decode(line, run_id, number))
        return records

    def append(self, record: IterationRecord) -> None:
        path = self._path(record.run_id)
        payload = json.dumps(record.to_dict(), allow_nan=False, separators=(",", ":")) + "\n"
        with _LOCK:
            self.root.mkdir(parents=True, exist_ok=True)
            if path.exists():
                self._read_path(path, record.run_id)
                with path.open("rb") as existing:
                    existing.seek(0, 2)
                    if existing.tell():
                        existing.seek(-1, 2)
                        if existing.read(1) != b"\n":
                            raise ValueError(f"Run log {record.run_id} has an incomplete trailing record")
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)

    def reserve(self, run_id: str) -> None:
        """Create an empty run exclusively; existing IDs raise FileExistsError."""
        with _LOCK:
            path = self._path(run_id)
            self.root.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8"):
                pass

    def read(self, run_id: str) -> list[dict]:
        with _LOCK:
            return self._read_path(self._path(run_id), run_id)

    def validate_run_id(self, run_id: str) -> None:
        """Raise ValueError unless run_id is a safe run identifier."""
        self._path(run_id)

    def _event_path(self, run_id: str) -> Path:
        # A subdirectory, so list_runs' *.jsonl glob only ever sees scored runs.
        return self._path(run_id).parent / "events" / f"{run_id}.jsonl"

    def append_event(self, run_id: str, event: dict) -> None:
        """Record something that is not a scored attempt: run start/finish,
        invalid or duplicate proposals, provider failures. These have no valid
        Evaluation, so they never go into the IterationRecord log."""
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("event must be an object with a string 'type'")
        payload = json.dumps(dict(event, run_id=run_id), allow_nan=False,
                             separators=(",", ":")) + "\n"
        path = self._event_path(run_id)
        with _LOCK:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)

    def read_events(self, run_id: str) -> list[dict]:
        path = self._event_path(run_id)
        with _LOCK:
            if not path.exists():
                return []
            events = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.endswith("}"):
                    continue  # an interrupted append, not a committed event
                try:
                    event = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"Malformed run log {run_id} events, line {number}: {exc}") from None
                if not isinstance(event, dict):
                    raise ValueError(f"Malformed run log {run_id} events, line {number}: not an object")
                events.append(event)
            return events

    def list_runs(self) -> list[dict]:
        """Return sorted run IDs and their counts of completed records.

        Locks coordinate readers and writers across instances in this process.
        Multiple backend processes must use separate log roots.
        """
        with _LOCK:
            if not self.root.exists():
                return []
            runs = []
            for path in sorted(self.root.glob("*.jsonl")):
                run_id = path.stem
                records = self._read_path(self._path(run_id), run_id)
                runs.append({"run_id": run_id, "iterations": len(records)})
            return runs
