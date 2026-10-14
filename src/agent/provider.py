"""Nebius Token Factory client for the design agent.  OWNER: Pranoy.

    python -m src.agent.provider --smoke            # cheap Nano check, < $0.01
    python -m src.agent.provider --smoke --model super

Token Factory is OpenAI-compatible: POST {base_url}/chat/completions with a
Bearer key. Plain `requests`, no SDK, to keep the service dependencies small.

Cost control is enforced here, not left to the caller:
- every call is appended to a spend ledger (JSONL) with its token usage and
  price, so spend survives restarts and is reportable;
- a call is refused once the ledger total reaches the spend cap;
- max_tokens is always sent, retries are bounded, and the API key never
  appears in errors, logs or the ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import requests

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

# Aliases so the CLI and config can say "nano" or "super".
MODELS: dict[str, str] = {
    "nano": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    "super": "nvidia/nemotron-3-super-120b-a12b",
}

# USD per token (prompt, completion), from GET /v1/models?verbose=true on
# 2026-10-14. Unknown models are priced at the most expensive listed rate so
# the cap errs towards refusing, never towards overspending.
PRICING: dict[str, tuple[float, float]] = {
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B": (0.06e-6, 0.24e-6),
    "nvidia/nemotron-3-super-120b-a12b": (0.30e-6, 0.90e-6),
    "nvidia/Nemotron-3-Ultra-550b-a55b": (1.00e-6, 3.00e-6),
}
_FALLBACK_PRICE = (1.00e-6, 3.00e-6)

_RETRY_STATUS = {429, 500, 502, 503, 504}
_REPO_ROOT = Path(__file__).resolve().parents[2]


class ProviderError(RuntimeError):
    """A provider call failed. Messages never contain the API key."""


class BudgetExceeded(ProviderError):
    """The spend cap would be exceeded; no request was sent."""


def load_env_file(path: str | os.PathLike | None = None) -> None:
    """Load KEY=VALUE lines from .env without overriding real environment variables.

    AERO_ENV_FILE overrides the default location (the repository root).
    """
    env_path = Path(path or os.environ.get("AERO_ENV_FILE") or _REPO_ROOT / ".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def resolve_model(name: str) -> str:
    return MODELS.get(name, name)


def call_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = PRICING.get(model, _FALLBACK_PRICE)
    return prompt_tokens * prompt_price + completion_tokens * completion_price


class SpendLedger:
    """Append-only record of every billed call, shared by all runs."""

    _lock = Lock()

    def __init__(self, path: str | os.PathLike, cap_usd: float):
        if not cap_usd >= 0:
            raise ValueError("spend cap must be nonnegative")
        self.path = Path(path)
        self.cap_usd = float(cap_usd)

    def total(self) -> float:
        if not self.path.is_file():
            return 0.0
        total = 0.0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += float(json.loads(line)["cost_usd"])
        return total

    def check(self, reserve_usd: float) -> None:
        spent = self.total()
        if spent + reserve_usd > self.cap_usd:
            raise BudgetExceeded(
                f"spend cap ${self.cap_usd:.2f} reached: ${spent:.4f} spent, "
                f"next call could cost up to ${reserve_usd:.4f}"
            )

    def record(self, entry: dict) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, allow_nan=False) + "\n")


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, prompt: int, completion: int, cost: float) -> None:
        self.calls += 1
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.cost_usd += cost


@dataclass
class TokenFactoryClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = MODELS["nano"]
    timeout_s: float = 120.0
    max_retries: int = 2
    ledger: SpendLedger | None = None
    purpose: str = "agent"
    usage: Usage = field(default_factory=Usage)
    session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self):
        if not self.api_key:
            raise ProviderError("NEBIUS_API_KEY is not set")
        self.base_url = self.base_url.rstrip("/")
        self.model = resolve_model(self.model)

    def __repr__(self) -> str:  # never show the key
        return f"TokenFactoryClient(model={self.model!r}, base_url={self.base_url!r})"

    @classmethod
    def from_env(cls, model: str | None = None, purpose: str = "agent") -> "TokenFactoryClient":
        load_env_file()
        ledger = SpendLedger(
            os.environ.get("AERO_SPEND_LEDGER", str(_REPO_ROOT / "runs" / "spend.jsonl")),
            float(os.environ.get("AERO_SPEND_CAP_USD", "5.0")),
        )
        return cls(
            api_key=os.environ.get("NEBIUS_API_KEY", ""),
            base_url=os.environ.get("NEBIUS_BASE_URL", DEFAULT_BASE_URL),
            model=model or os.environ.get("NEMOTRON_MODEL", MODELS["nano"]),
            ledger=ledger,
            purpose=purpose,
        )

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _scrub(self, text: str) -> str:
        return text.replace(self.api_key, "<redacted>") if self.api_key else text

    def list_models(self) -> list[dict]:
        """Free: model metadata including pricing and supported features."""
        try:
            response = self.session.get(f"{self.base_url}/models", params={"verbose": "true"},
                                        headers=self._headers(), timeout=30)
        except requests.RequestException as exc:
            raise ProviderError(self._scrub(f"model listing failed: {exc}")) from None
        if response.status_code != 200:
            raise ProviderError(f"model listing failed: HTTP {response.status_code}")
        return response.json().get("data", [])

    def chat(self, messages: list[dict], *, tools: list[dict] | None = None,
             tool_choice: str | dict | None = None, max_tokens: int = 1024,
             temperature: float = 0.6, extra: dict | None = None) -> dict:
        """One chat completion. Returns the first choice's message plus metadata."""
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        prompt_chars = len(json.dumps(messages)) + len(json.dumps(tools or []))
        # Generous estimate (~3 chars per token) so the cap check is conservative.
        reserve = call_cost(self.model, prompt_chars // 3 + 1, max_tokens)
        if self.ledger:
            self.ledger.check(reserve)
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature}
        if tools:
            body["tools"] = tools
            if tool_choice is not None:
                body["tool_choice"] = tool_choice
        if extra:
            body.update(extra)

        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            if attempt:
                time.sleep(min(2 ** attempt, 8))
            started = time.perf_counter()
            try:
                response = self.session.post(f"{self.base_url}/chat/completions", json=body,
                                             headers=self._headers(), timeout=self.timeout_s)
            except requests.RequestException as exc:
                last_error = self._scrub(f"{type(exc).__name__}: {exc}")
                continue
            latency_s = time.perf_counter() - started
            if response.status_code in _RETRY_STATUS:
                last_error = f"HTTP {response.status_code}"
                continue
            if response.status_code != 200:
                raise ProviderError(self._scrub(
                    f"HTTP {response.status_code}: {response.text[:300]}"))
            try:
                data = response.json()
                choice = data["choices"][0]
                message = choice["message"]
            except (ValueError, KeyError, IndexError, TypeError):
                raise ProviderError("malformed completion response") from None
            usage = data.get("usage") or {}
            prompt_tokens = int(usage.get("prompt_tokens", 0))
            completion_tokens = int(usage.get("completion_tokens", 0))
            cost = call_cost(self.model, prompt_tokens, completion_tokens)
            self.usage.add(prompt_tokens, completion_tokens, cost)
            if self.ledger:
                self.ledger.record({
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": self.model,
                    "purpose": self.purpose, "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens, "cost_usd": cost,
                    "latency_s": round(latency_s, 3),
                })
            return {"message": message, "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                    "cost_usd": cost, "latency_s": latency_s}
        raise ProviderError(f"provider call failed after {self.max_retries + 1} attempts: {last_error}")


_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "evaluate",
        "description": "Score a car design with the aerodynamic surrogate.",
        "parameters": {
            "type": "object",
            "properties": {"B_Ramp_Angle": {"type": "number", "description": "deg, -8 to 15"}},
            "required": ["B_Ramp_Angle"],
        },
    },
}


def smoke(model: str) -> dict:
    """Two tiny billed calls: a plain reply and a forced tool call."""
    client = TokenFactoryClient.from_env(model=model, purpose="smoke")
    available = {m["id"]: m for m in client.list_models()}
    if client.model not in available:
        raise ProviderError(f"model {client.model} is not available to this key")
    meta = available[client.model]
    plain = client.chat([{"role": "user", "content": "Reply with exactly: ok"}], max_tokens=256)
    tool = client.chat(
        [{"role": "user", "content": "Evaluate a design with B_Ramp_Angle = 2.5 using the tool."}],
        tools=[_PROBE_TOOL], tool_choice="auto", max_tokens=512,
    )
    calls = tool["message"].get("tool_calls") or []
    result = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base_url": client.base_url,
        "model": client.model,
        "supported_features": meta.get("supported_features"),
        "context_length": meta.get("context_length"),
        "plain_reply": (plain["message"].get("content") or "")[:200],
        "plain_finish_reason": plain["finish_reason"],
        "tool_call_returned": bool(calls),
        "tool_call": calls[0]["function"] if calls else None,
        "tool_finish_reason": tool["finish_reason"],
        "calls": [{k: plain[k] for k in ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s")},
                  {k: tool[k] for k in ("prompt_tokens", "completion_tokens", "cost_usd", "latency_s")}],
        "cost_usd": client.usage.cost_usd,
        "ledger_total_usd": client.ledger.total() if client.ledger else None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Token Factory provider checks")
    parser.add_argument("--smoke", action="store_true", help="two tiny billed calls")
    parser.add_argument("--model", default="nano", help="nano | super | full model ID")
    parser.add_argument("--out", default="runs/provider-smoke.json")
    args = parser.parse_args()
    if not args.smoke:
        parser.error("nothing to do; pass --smoke")
    try:
        result = smoke(args.model)
    except ProviderError as exc:
        parser.exit(1, f"Smoke failed: {exc}\n")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
