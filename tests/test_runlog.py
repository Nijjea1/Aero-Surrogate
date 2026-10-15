from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import pytest

from src.contracts import IterationRecord, default_params
from src.service.runlog import RunLog


def record(run_id="test-run", iteration=0):
    return IterationRecord(
        run_id=run_id, iteration=iteration, params=default_params(), cd=0.3,
        cl=-0.04, uncertainty=0.005, in_distribution=True, accepted=True,
        reasoning="Reduced rear pressure drag — hypothesis only.",
    )


def test_round_trip_and_listing(tmp_path):
    log = RunLog(tmp_path / "nested")
    assert log.list_runs() == []
    first = record()
    second = replace(first, iteration=1, accepted=False, reject_reason="uncertain")
    log.append(first)
    log.append(second)
    log.append(record("another"))
    assert RunLog(log.root).read(first.run_id) == [first.to_dict(), second.to_dict()]
    assert log.list_runs() == [
        {"run_id": "another", "iterations": 1},
        {"run_id": "test-run", "iterations": 2},
    ]


@pytest.mark.parametrize("run_id", ["", "../escape", "a/b", "a\\b", "x.y", "x" * 129, "CON"])
def test_invalid_ids(tmp_path, run_id):
    log = RunLog(tmp_path)
    with pytest.raises(ValueError):
        log.read(run_id)
    with pytest.raises(ValueError):
        log.append(record(run_id))
    assert list(tmp_path.iterdir()) == []


def test_missing_run(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunLog(tmp_path).read("absent")


def test_reserve_refuses_existing_empty_run(tmp_path):
    log = RunLog(tmp_path)
    log.reserve("reserved")
    assert log.read("reserved") == []
    with pytest.raises(FileExistsError):
        log.reserve("reserved")
    log.append(record("reserved"))
    assert len(log.read("reserved")) == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_append_does_not_create_log(tmp_path, value):
    with pytest.raises(ValueError):
        RunLog(tmp_path).append(replace(record(), cd=value))
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("tail", ["not json\n", "\n", "[]\n", '{"run_id":"other"}\n',
                                    '{"run_id":"test-run","cd":NaN}\n',
                                    '{"run_id":"test-run","cd":1e999}\n'])
def test_corrupt_completed_line_is_reported(tmp_path, tail):
    log = RunLog(tmp_path)
    log.append(record())
    with (tmp_path / "test-run.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(tail)
    with pytest.raises(ValueError, match="line 2"):
        log.read("test-run")
    with pytest.raises(ValueError):
        log.append(record(iteration=2))


def test_partial_tail_readable_but_append_refused(tmp_path):
    log = RunLog(tmp_path)
    log.append(record())
    path = tmp_path / "test-run.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"run_id":')
    assert log.read("test-run") == [record().to_dict()]
    before = path.read_bytes()
    with pytest.raises(ValueError, match="incomplete"):
        log.append(record(iteration=1))
    assert path.read_bytes() == before


def test_multiple_instances_concurrent_append_and_read(tmp_path):
    def write(iteration):
        log = RunLog(tmp_path)
        log.append(record(iteration=iteration))
        assert all(item["run_id"] == "test-run" for item in log.read("test-run"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write, range(80)))
    records = RunLog(tmp_path).read("test-run")
    assert sorted(item["iteration"] for item in records) == list(range(80))
    lines = (tmp_path / "test-run.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 80
    assert all(isinstance(json.loads(line), dict) for line in lines)


def test_event_log_survives_torn_and_garbled_lines(tmp_path):
    log = RunLog(tmp_path)
    log.append_event("evt", {"type": "run_started"})
    path = tmp_path / "events" / "evt.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"type": "step", "x": {"y": 1}')  # a crash mid-append, ending in }
    # The next event starts a fresh line instead of being glued onto the torn one.
    log.append_event("evt", {"type": "run_finished"})
    assert [e["type"] for e in log.read_events("evt")] == ["run_started", "run_finished"]
    # A garbled line is skipped, not fatal: events are auxiliary.
    path.write_text('{"type": "ok"}\nnot json but ends}\n{"type": "after"}\n', encoding="utf-8")
    assert [e["type"] for e in log.read_events("evt")] == ["ok", "after"]


def test_run_id_validation_is_public(tmp_path):
    log = RunLog(tmp_path)
    log.validate_run_id("agent-0123abcd")
    for bad in ("../x", "a.b", "CON", ""):
        with pytest.raises(ValueError):
            log.validate_run_id(bad)
