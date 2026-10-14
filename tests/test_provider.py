"""Token Factory client, against a fake HTTP session. No network, no spend."""

import json

import pytest
import requests

from src.agent import provider
from src.agent.provider import (
    BudgetExceeded, ProviderError, SpendLedger, TokenFactoryClient, call_cost, load_env_file,
)

KEY = "v1.secret-test-key-0123456789"


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code, self._payload, self.text = status, payload, text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def completion(content="ok", tool_calls=None, prompt=100, completion_tokens=50, finish="stop"):
    message = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return FakeResponse(payload={"choices": [{"message": message, "finish_reason": finish}],
                                 "usage": {"prompt_tokens": prompt, "completion_tokens": completion_tokens}})


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(provider.time, "sleep", lambda _: None)


def client(responses, tmp_path, cap=5.0, **kwargs):
    return TokenFactoryClient(api_key=KEY, model="nano", session=FakeSession(responses),
                              ledger=SpendLedger(tmp_path / "spend.jsonl", cap), **kwargs)


def test_request_shape_auth_and_usage_accounting(tmp_path):
    c = client([completion(prompt=1000, completion_tokens=500)], tmp_path)
    reply = c.chat([{"role": "user", "content": "hi"}], max_tokens=64)
    url, kwargs = c.session.requests[0]
    assert url == "https://api.tokenfactory.nebius.com/v1/chat/completions"
    assert kwargs["headers"]["Authorization"] == f"Bearer {KEY}"
    assert kwargs["json"]["model"] == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert kwargs["json"]["max_tokens"] == 64
    assert kwargs["timeout"] == c.timeout_s
    expected = call_cost(c.model, 1000, 500)
    assert reply["cost_usd"] == pytest.approx(expected) == pytest.approx(1000 * 0.06e-6 + 500 * 0.24e-6)
    assert c.usage.calls == 1 and c.usage.cost_usd == pytest.approx(expected)
    text = (tmp_path / "spend.jsonl").read_text()
    reserve, settle = map(json.loads, text.splitlines())
    assert reserve["kind"] == "reserve" and settle["kind"] == "settle" and settle["id"] == reserve["id"]
    assert settle["billed_usd"] == pytest.approx(expected)
    assert c.ledger.total() == pytest.approx(expected)
    assert KEY not in text


def test_retries_are_bounded_and_then_fail(tmp_path):
    c = client([FakeResponse(503), FakeResponse(429), FakeResponse(500)], tmp_path, max_retries=2)
    with pytest.raises(ProviderError, match="after 3 attempts"):
        c.chat([{"role": "user", "content": "hi"}])
    assert len(c.session.requests) == 3
    assert c.usage.calls == 0


def test_transient_failure_then_success(tmp_path):
    c = client([requests.ConnectionError("reset"), completion()], tmp_path)
    assert c.chat([{"role": "user", "content": "hi"}])["message"]["content"] == "ok"


def test_errors_never_contain_the_key(tmp_path):
    c = client([FakeResponse(401, text=f"bad key {KEY}")], tmp_path)
    with pytest.raises(ProviderError) as info:
        c.chat([{"role": "user", "content": "hi"}])
    assert KEY not in str(info.value) and "<redacted>" in str(info.value)
    c = client([requests.ConnectionError(f"proxy said {KEY}")] * 3, tmp_path)
    with pytest.raises(ProviderError) as info:
        c.chat([{"role": "user", "content": "hi"}])
    assert KEY not in str(info.value)
    assert KEY not in repr(c)


def test_malformed_response_is_an_error(tmp_path):
    c = client([FakeResponse(payload={"choices": []})], tmp_path)
    with pytest.raises(ProviderError, match="malformed"):
        c.chat([{"role": "user", "content": "hi"}])


def test_spend_cap_refuses_before_sending(tmp_path):
    ledger = tmp_path / "spend.jsonl"
    ledger.write_text(json.dumps({"cost_usd": 4.9999}) + "\n")
    c = client([completion()], tmp_path, cap=5.0)
    with pytest.raises(BudgetExceeded, match="spend cap"):
        c.chat([{"role": "user", "content": "x" * 3000}], max_tokens=4096)
    assert c.session.requests == []


def test_unknown_model_is_priced_conservatively():
    assert call_cost("someone/new-model", 1000, 1000) >= call_cost("nvidia/nemotron-3-super-120b-a12b", 1000, 1000)


def test_missing_key_is_refused():
    with pytest.raises(ProviderError, match="NEBIUS_API_KEY"):
        TokenFactoryClient(api_key="")


def test_env_file_does_not_override_real_environment(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nAERO_TEST_A=from-file\nAERO_TEST_B='quoted'\nAERO_TEST_EMPTY=\n")
    monkeypatch.setenv("AERO_TEST_A", "from-shell")
    monkeypatch.delenv("AERO_TEST_B", raising=False)
    monkeypatch.delenv("AERO_TEST_EMPTY", raising=False)
    load_env_file(env)
    import os
    assert os.environ["AERO_TEST_A"] == "from-shell"
    assert os.environ.pop("AERO_TEST_B") == "quoted"
    assert "AERO_TEST_EMPTY" not in os.environ


# ---------------------------------------------------- audit regressions, Oct 14


def test_concurrent_calls_cannot_jointly_pass_the_cap(tmp_path):
    # Audit 1: two callers both passed a check-then-spend preflight. Now each
    # reserves its worst case atomically, so the second is refused up front.
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)
    worst_case = call_cost(provider.MODELS["nano"], 1, 40)
    ledger = SpendLedger(tmp_path / "spend.jsonl", worst_case * 1.5)
    clients = [TokenFactoryClient(api_key=KEY, model="nano", ledger=ledger,
                                  session=FakeSession([completion(prompt=0, completion_tokens=40)]))
               for _ in range(2)]

    def attempt(c):
        barrier.wait(timeout=5)
        try:
            c.chat([], max_tokens=40)
            return "sent"
        except BudgetExceeded:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(attempt, clients))
    assert outcomes == ["refused", "sent"]
    assert ledger.total() <= ledger.cap_usd


def test_reservation_is_released_when_nothing_is_billed(tmp_path):
    c = client([FakeResponse(503)] * 3, tmp_path)
    with pytest.raises(ProviderError):
        c.chat([{"role": "user", "content": "hi"}])
    assert c.ledger.total() == pytest.approx(0.0, abs=1e-15)


def test_stale_ledger_lock_from_a_dead_process_is_broken(tmp_path):
    import os

    ledger = SpendLedger(tmp_path / "spend.jsonl", 1.0)
    lock = tmp_path / "spend.jsonl.lock"
    lock.write_text("held by a crashed process")
    old = provider.time.time() - 120
    os.utime(lock, (old, old))
    assert ledger.total() == 0.0
    assert not lock.exists()


def test_unreadable_ledger_refuses_to_spend(tmp_path):
    (tmp_path / "spend.jsonl").write_text("not json\n")
    c = client([completion()], tmp_path)
    with pytest.raises(ProviderError, match="unreadable"):
        c.chat([{"role": "user", "content": "hi"}])
    assert c.session.requests == []


def test_billed_malformed_reply_is_still_accounted(tmp_path):
    # Audit 6: an HTTP 200 with usage but no choices is billed by the provider.
    c = client([FakeResponse(payload={"choices": [], "usage": {"prompt_tokens": 0,
                                                              "completion_tokens": 40}})], tmp_path)
    with pytest.raises(ProviderError, match="malformed"):
        c.chat([], max_tokens=40)
    expected = call_cost(c.model, 0, 40)
    assert c.usage.cost_usd == pytest.approx(expected)
    assert c.ledger.total() == pytest.approx(expected)


def test_missing_usage_is_charged_the_reserved_maximum(tmp_path):
    c = client([FakeResponse(payload={"choices": [{"message": {"content": "ok"}}]})], tmp_path)
    c.chat([{"role": "user", "content": "hi"}], max_tokens=100)
    assert c.usage.cost_usd >= call_cost(c.model, 0, 100)
    assert "usage missing" in (tmp_path / "spend.jsonl").read_text()


@pytest.mark.parametrize("key", ["max_tokens", "model", "messages", "tools", "stream", "n"])
def test_extra_cannot_override_protected_fields(tmp_path, key):
    # Audit 7: extra={"max_tokens": 1_000_000} used to bypass the reservation.
    c = client([completion()], tmp_path)
    with pytest.raises(ValueError, match="may not override"):
        c.chat([], max_tokens=1, extra={key: 1_000_000})
    assert c.session.requests == []
    c.chat([], max_tokens=1, extra={"chat_template_kwargs": {"enable_thinking": False}})
    assert c.session.requests[0][1]["json"]["max_tokens"] == 1


def test_run_budget_is_checked_before_sending(tmp_path):
    c = client([completion()] * 3, tmp_path)
    c.chat([], max_tokens=10, budget_usd=1.0)
    with pytest.raises(BudgetExceeded, match="run budget"):
        c.chat([], max_tokens=10, budget_usd=c.usage.cost_usd)
    assert len(c.session.requests) == 1
