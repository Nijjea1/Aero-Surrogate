"""Service configuration, read once from the environment.  OWNER: Pranoy.

Everything a deployment changes lives here, so local development needs no
configuration at all and a host sets only environment variables (never files
baked into an image). Invalid values fail startup rather than being guessed.

    AERO_CORS_ORIGINS            comma-separated origins allowed to call the API
    AERO_ADMIN_TOKEN             bearer token that lifts public limits (unset = no admin)
    AERO_PUBLIC_RUNS             1/0: may callers without the token start agent runs
    AERO_MAX_CONCURRENT_RUNS     agent runs executing at once
    AERO_PUBLIC_MAX_ITERATIONS   evaluation budget ceiling for public runs
    AERO_ADMIN_MAX_ITERATIONS    ... and for admin runs
    AERO_PUBLIC_RUN_BUDGET_USD   spend ceiling per public run
    AERO_ADMIN_RUN_BUDGET_USD    ... per admin run
    AERO_PUBLIC_DAILY_BUDGET_USD total public spend per UTC day
    AERO_RATE_LIMIT_PER_MINUTE   scoring requests per client IP per minute (0 = off)
    AERO_MAX_BODY_BYTES          largest accepted request body
    AERO_RUNLOG_DIR              run logs (put on a persistent volume when hosted)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

AGENT_MODELS = ("nano", "super")


def _number(env: dict, name: str, default: float, *, integer: bool = False, minimum: float = 0.0):
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError:
        raise ValueError(f"{name} must be {'an integer' if integer else 'a number'}, got {raw!r}") from None
    if not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite number >= {minimum:g}, got {raw!r}")
    return value


def _flag(env: dict, name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    if raw.strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if raw.strip().lower() in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be 1/0 or true/false, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")
    admin_token: str | None = None
    public_runs: bool = True
    max_concurrent_runs: int = 1
    public_max_iterations: int = 25
    admin_max_iterations: int = 50
    public_run_budget_usd: float = 0.10
    admin_run_budget_usd: float = 0.50
    public_daily_budget_usd: float = 1.00
    rate_limit_per_minute: int = 120
    max_body_bytes: int = 65536
    runlog_dir: str = "runs/logs"

    @classmethod
    def from_env(cls, env: dict | None = None) -> "Settings":
        env = os.environ if env is None else env
        default = cls()
        origins = env.get("AERO_CORS_ORIGINS")
        if origins is not None and origins.strip():
            parsed = tuple(o.strip().rstrip("/") for o in origins.split(",") if o.strip())
            for origin in parsed:
                if origin == "*" or not origin.startswith(("http://", "https://")):
                    raise ValueError(f"AERO_CORS_ORIGINS entries must be explicit http(s) origins, got {origin!r}")
        else:
            parsed = default.cors_origins
        token = (env.get("AERO_ADMIN_TOKEN") or "").strip() or None
        if token is not None and len(token) < 16:
            raise ValueError("AERO_ADMIN_TOKEN must be at least 16 characters")
        return cls(
            cors_origins=parsed,
            admin_token=token,
            public_runs=_flag(env, "AERO_PUBLIC_RUNS", default.public_runs),
            max_concurrent_runs=_number(env, "AERO_MAX_CONCURRENT_RUNS", default.max_concurrent_runs,
                                        integer=True, minimum=1),
            public_max_iterations=_number(env, "AERO_PUBLIC_MAX_ITERATIONS", default.public_max_iterations,
                                          integer=True, minimum=1),
            admin_max_iterations=_number(env, "AERO_ADMIN_MAX_ITERATIONS", default.admin_max_iterations,
                                         integer=True, minimum=1),
            public_run_budget_usd=_number(env, "AERO_PUBLIC_RUN_BUDGET_USD", default.public_run_budget_usd),
            admin_run_budget_usd=_number(env, "AERO_ADMIN_RUN_BUDGET_USD", default.admin_run_budget_usd),
            public_daily_budget_usd=_number(env, "AERO_PUBLIC_DAILY_BUDGET_USD",
                                            default.public_daily_budget_usd),
            rate_limit_per_minute=_number(env, "AERO_RATE_LIMIT_PER_MINUTE", default.rate_limit_per_minute,
                                          integer=True),
            max_body_bytes=_number(env, "AERO_MAX_BODY_BYTES", default.max_body_bytes, integer=True,
                                   minimum=1024),
            runlog_dir=(env.get("AERO_RUNLOG_DIR") or "").strip() or default.runlog_dir,
        )
