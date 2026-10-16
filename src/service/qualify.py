"""Verify scalar references through service routes and the agent's shared gate.

The default runs the real FastAPI application in process, including its lifespan
and request/response validation. --service-url checks an already running HTTP
service instead. Neither mode starts Nemotron, writes agent logs, or establishes
CFD accuracy. Checkpoint/reference qualification and remote identity checks run
before any scoring request; a synthetic or different server fails closed.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

import requests

from ..agent import designer
from ..contracts import BODY_STYLES, DEFAULT_BODY_STYLE, PARAM_BOUNDS, PRESSURE_REGIONS, SCHEMA_VERSION, default_params
from .verify_checkpoint import verify_checkpoint

_EVALUATION_KEYS = {"cd", "cl", "uncertainty", "in_distribution", "pressure_summary",
                    "model_version", "latency_ms", "ood_score"}


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _equal_number(value, expected, tolerance):
    return _number(value) and _number(expected) and abs(value - expected) <= tolerance


def _check_evaluation(actual, expected, tolerance):
    errors = []
    if not isinstance(actual, dict) or set(actual) != _EVALUATION_KEYS:
        return ["Evaluation response fields differ from the frozen contract"]
    for key in ("cd", "cl", "uncertainty", "ood_score"):
        if not _equal_number(actual[key], expected[key], tolerance):
            errors.append(f"{key} differs from qualified prediction")
    if any(_number(actual[key]) and actual[key] < 0 for key in ("uncertainty", "ood_score")):
        errors.append("uncertainty and ood_score must be nonnegative")
    if type(actual["in_distribution"]) is not bool or actual["in_distribution"] != expected["in_distribution"]:
        errors.append("in_distribution differs from qualified prediction")
    if actual["model_version"] != expected["model_version"]:
        errors.append("model_version differs from qualified prediction")
    pressure = actual["pressure_summary"]
    if (not isinstance(pressure, dict) or set(pressure) != set(expected["pressure_summary"])
            or any(not _equal_number(value, expected["pressure_summary"][key], tolerance)
                   for key, value in pressure.items())):
        errors.append("pressure_summary differs from qualified prediction")
    if not _number(actual["latency_ms"]) or actual["latency_ms"] < 0:
        errors.append("latency_ms must be finite and nonnegative")
    return errors


class _Remote:
    def __init__(self, url):
        parsed = urlsplit(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("service URL must be http(s) without embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("service URL must not contain a query or fragment")
        self.url = url.rstrip("/")
        self.session = requests.Session()
        # Do not infer credentials from a developer's .netrc or proxy settings.
        self.session.trust_env = False

    def get(self, path):
        return self.session.get(self.url + path, timeout=(3.05, 30), allow_redirects=False)

    def post(self, path, **kwargs):
        return self.session.post(self.url + path, timeout=(3.05, 30), allow_redirects=False, **kwargs)

    def close(self):
        self.session.close()


@contextmanager
def _client(checkpoint, service_url, log_dir):
    if service_url is not None:
        client = _Remote(service_url)
        try:
            yield client
        finally:
            client.close()
        return
    from fastapi.testclient import TestClient

    from .app import create_app
    from .predictor import TorchPredictor
    from .runlog import RunLog
    from .settings import Settings

    # A temporary private log root prevents startup orphan recovery from touching
    # the user's existing runs. No provider or field capability is configured.
    with tempfile.TemporaryDirectory(prefix="aero-qualification-", dir=log_dir) as temporary:
        application = create_app(predictor=TorchPredictor(checkpoint), runlog=RunLog(temporary),
                                 settings=Settings(public_runs=False, rate_limit_per_minute=0,
                                                   runlog_dir=temporary))
        with TestClient(application) as client:
            yield client


def _request(client, path, *, body=None):
    response = client.get(path) if body is None else client.post(path, json=body)
    if response.status_code != 200:
        # Do not print response bodies, URLs or remote headers that could carry
        # deployment details. The route and status are sufficient evidence.
        raise ValueError(f"{path} returned HTTP {response.status_code}, expected 200")
    result = response.json()
    if not isinstance(result, dict):
        raise ValueError(f"{path} did not return a JSON object")
    return result


def _check_identity(health, schema, qualification):
    expected = qualification["provenance"]
    identity = health.get("provenance")
    if health.get("ok") is not True or health.get("synthetic") is not False or not isinstance(identity, dict):
        raise ValueError("service health must identify a real ready predictor")
    keys = ("checkpoint_sha256", "model_version", "schema_version", "preprocess_version",
            "uncertainty_threshold", "ood_score_threshold", "output_provenance", "serving_policy")
    if any(identity.get(key) != expected.get(key) for key in keys):
        raise ValueError("service identity differs from qualified checkpoint or serving conditions")
    if (identity.get("synthetic") is not False or health.get("schema_version") != SCHEMA_VERSION
            or health.get("model_version") != expected["model_version"]
            or health.get("output_provenance") != expected["output_provenance"]):
        raise ValueError("service health/output provenance differs from qualification")
    thresholds = qualification["thresholds"]
    if (schema.get("schema_version") != SCHEMA_VERSION or schema.get("param_bounds") !=
            {key: list(value) for key, value in PARAM_BOUNDS.items()} or schema.get("defaults") != default_params()
            or schema.get("body_styles") != BODY_STYLES
            or schema.get("default_body_style") != DEFAULT_BODY_STYLE
            or schema.get("pressure_regions") != PRESSURE_REGIONS
            or not _number(schema.get("uncertainty_threshold"))
            or not _number(schema.get("ood_score_threshold"))
            or schema.get("uncertainty_threshold") != thresholds["uncertainty"]
            or schema.get("ood_score_threshold") != thresholds["ood_score"]):
        raise ValueError("service schema or gate thresholds differ from qualification")


def qualify_service(checkpoint, reference_pack, *, service_url=None, log_dir=None):
    """Return explicit route/gate parity evidence, retaining failed checks.

    Agent-gate evidence exercises the exact rejection_reason imported by the
    designer with the service thresholds. It is not a model conversation or
    proof of adaptation. Every reference is evaluated and every unordered pair
    is compared, so both trust outcomes and cross-style flags are covered when
    the independent reference pack contains them.
    """
    qualification = verify_checkpoint(checkpoint, reference_pack)
    report = {"qualification_version": 1, "passed": False,
              "scope": "remote_http" if service_url is not None else "in_process_asgi",
              "claim": "serving and shared-agent-gate parity; no Nemotron calls or CFD accuracy claim",
              "provenance": qualification.get("provenance"),
              "checkpoint_qualification": qualification, "ready": None, "health": None, "schema": None,
              "evaluations": [], "comparisons": [], "agent_gate": [], "errors": []}
    if not qualification["passed"]:
        report["errors"].append("checkpoint/reference qualification failed; no route requests made")
        return report
    tolerance = qualification.get("absolute_tolerance", qualification.get("tolerance"))
    try:
        if not _number(tolerance) or not 0 < tolerance <= 1e-4:
            raise ValueError("qualification did not supply a valid tolerance")
        with _client(checkpoint, service_url, log_dir) as client:
            ready = _request(client, "/ready")
            report["ready"] = ready
            if ready.get("ready") is not True:
                raise ValueError("service readiness check did not report ready=true")
            health = _request(client, "/health")
            schema = _request(client, "/schema")
            report.update(health=health, schema=schema)
            _check_identity(health, schema, qualification)
            rows = qualification["results"]
            thresholds = {"threshold": schema["uncertainty_threshold"], "ood_threshold": schema["ood_score_threshold"]}
            for row in rows:
                body = {"params": row["params"], "body_style": row["body_style"]}
                actual = _request(client, "/evaluate", body=body)
                errors = _check_evaluation(actual, row["actual"], tolerance)
                report["evaluations"].append({"id": row["id"], "passed": not errors, "actual": actual,
                                               "errors": errors})
                report["errors"].extend(f"/evaluate {row['id']}: {error}" for error in errors)
                if errors:
                    continue
                reason = designer.rejection_reason(actual, body_style=row["body_style"], **thresholds)
                agrees = reason == row["actual"]["reject_reason"]
                report["agent_gate"].append({"id": row["id"], "accepted": reason is None,
                                             "reject_reason": reason, "passed": agrees})
                if not agrees:
                    report["errors"].append(f"agent gate {row['id']}: differs from independent reference")
            for index, left in enumerate(rows):
                for right in rows[index + 1:]:
                    actual = _request(client, "/compare", body={
                        "design_a": {"params": left["params"], "body_style": left["body_style"]},
                        "design_b": {"params": right["params"], "body_style": right["body_style"]}})
                    errors = []
                    for key, row in (("design_a", left), ("design_b", right)):
                        errors.extend(f"{key}: {error}" for error in
                                      _check_evaluation(actual.get(key), row["actual"], tolerance))
                    delta = actual.get("delta", {})
                    for key in ("cd", "cl"):
                        expected_delta = right["actual"][key] - left["actual"][key]
                        if not _equal_number(delta.get(key), expected_delta, tolerance):
                            errors.append(f"delta.{key} must be design_b minus design_a")
                    common = left["actual"]["pressure_summary"].keys() & right["actual"]["pressure_summary"].keys()
                    expected_pressure = {key: right["actual"]["pressure_summary"][key] -
                                         left["actual"]["pressure_summary"][key] for key in common}
                    if delta.get("pressure_summary") != expected_pressure:
                        errors.append("delta.pressure_summary differs from qualified predictions")
                    reasons = {"design_a": left["actual"]["reject_reason"],
                               "design_b": right["actual"]["reject_reason"]}
                    if actual.get("reject_reasons") != reasons:
                        errors.append("comparison rejection reasons differ")
                    trusted = all(reason is None for reason in reasons.values())
                    if type(actual.get("trusted")) is not bool or actual["trusted"] != trusted:
                        errors.append("comparison trusted flag differs")
                    if (type(actual.get("same_body_style")) is not bool or actual["same_body_style"] !=
                            (left["body_style"] == right["body_style"])):
                        errors.append("comparison body-style flag differs")
                    if actual.get("delta_direction") != "design_b minus design_a":
                        errors.append("comparison delta direction differs")
                    if actual.get("comparison_kind") != "scalar and regional summaries; not a surface-field comparison":
                        errors.append("comparison scope differs")
                    if actual.get("unavailable_pressure_regions") != sorted(set(PRESSURE_REGIONS) - common):
                        errors.append("comparison unavailable pressure regions differ")
                    report["comparisons"].append({"design_a": left["id"], "design_b": right["id"],
                                                  "passed": not errors, "errors": errors, "actual": actual})
                    report["errors"].extend(f"/compare {left['id']}/{right['id']}: {error}" for error in errors)
    except Exception as exc:
        # Explicit qualification returns a report even for an unavailable server.
        # Avoid requests exception strings: they can contain URL credentials.
        detail = "HTTP request failed" if isinstance(exc, requests.RequestException) else str(exc)
        report["errors"].append(f"{type(exc).__name__}: {detail}")
    report["passed"] = not report["errors"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--service-url")
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.exists():
        parser.error("output exists; choose a fresh report path")
    try:
        pack = json.loads(Path(args.references).read_text(encoding="utf-8"))
        # Exclusive reservation proves evidence can be saved before remote scoring.
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump({"status": "qualification not yet complete"}, stream)
        report = qualify_service(args.checkpoint, pack, service_url=args.service_url)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"Service qualification {'passed' if report['passed'] else 'failed'}; report: {output}")
        return 0 if report["passed"] else 1
    except (OSError, ValueError, TypeError) as exc:
        print(f"Service qualification stopped: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
