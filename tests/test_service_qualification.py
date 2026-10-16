"""Service-route qualification with independent trainer references, no provider."""

import json
import shutil
from contextlib import contextmanager

import pytest

from src.agent.provider import TokenFactoryClient
from src.service import qualify
from tests.test_torch_predictor import trained  # noqa: F401


@pytest.fixture(autouse=True)
def no_live_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("qualification tests never create a provider or use remote HTTP")

    monkeypatch.setattr(TokenFactoryClient, "from_env", forbidden)
    monkeypatch.setattr(qualify.requests.Session, "request", forbidden)


@pytest.fixture
def native_pack(trained, tmp_path):
    from src.train.reference import build_scalar

    checkpoint = tmp_path / "runs" / "drivaernet-random" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    shutil.copyfile(trained[0], checkpoint)
    return build_scalar(tmp_path, tmp_path / "handover")


def test_all_reference_routes_and_agent_gate_agree(trained, native_pack, tmp_path):
    report = qualify.qualify_service(trained[0], native_pack, log_dir=tmp_path)
    assert report["passed"], report["errors"]
    assert report["scope"] == "in_process_asgi"
    assert len(report["evaluations"]) == len(report["agent_gate"]) == 5
    assert len(report["comparisons"]) == 10
    assert any(not item["accepted"] for item in report["agent_gate"])
    assert any(item["accepted"] for item in report["agent_gate"])
    assert all(item["passed"] for item in report["agent_gate"])
    assert "no Nemotron calls" in report["claim"]
    assert not list(tmp_path.glob("aero-qualification-*"))


def test_bad_independent_reference_stops_before_routes(trained, native_pack, monkeypatch):
    native_pack["designs"][0]["expected"]["cd"] += .1

    def forbidden(*args, **kwargs):
        raise AssertionError("route client must not open after failed qualification")

    monkeypatch.setattr(qualify, "_client", forbidden)
    report = qualify.qualify_service(trained[0], native_pack)
    assert not report["passed"] and not report["evaluations"]
    assert not report["checkpoint_qualification"]["passed"]


@pytest.mark.parametrize("mutation", ["hash", "schema", "threshold", "provenance", "synthetic",
                                     "ready", "styles", "default_style", "regions"])
def test_service_identity_mismatch_stops_before_scoring(trained, native_pack, tmp_path, monkeypatch, mutation):
    original = qualify._request
    seen = []

    def changed(client, path, **kwargs):
        seen.append(path)
        result = original(client, path, **kwargs)
        if path == "/ready" and mutation == "ready":
            result["ready"] = False
        elif path == "/health":
            if mutation == "hash":
                result["provenance"]["checkpoint_sha256"] = "f" * 64
            elif mutation == "provenance":
                result["output_provenance"]["cl"] = "model"
            elif mutation == "synthetic":
                result["synthetic"] = True
        elif path == "/schema":
            if mutation == "schema":
                result["schema_version"] = "1.0.0"
            elif mutation == "threshold":
                result["ood_score_threshold"] += 1
            elif mutation == "styles":
                result["body_styles"] = ["fastback"]
            elif mutation == "default_style":
                result["default_body_style"] = "estateback"
            elif mutation == "regions":
                result["pressure_regions"] = ["invented"]
        return result

    monkeypatch.setattr(qualify, "_request", changed)
    report = qualify.qualify_service(trained[0], native_pack, log_dir=tmp_path)
    assert not report["passed"] and not report["evaluations"]
    assert "/evaluate" not in seen and "/compare" not in seen


@pytest.mark.parametrize("mutation", ["cd", "cl", "numeric_bool", "pressure", "missing_field"])
def test_evaluate_route_changes_cannot_pass(trained, native_pack, tmp_path, monkeypatch, mutation):
    original = qualify._request

    def changed(client, path, **kwargs):
        result = original(client, path, **kwargs)
        if path == "/evaluate":
            if mutation in ("cd", "cl"):
                result[mutation] += .02
            elif mutation == "numeric_bool":
                result["in_distribution"] = int(result["in_distribution"])
            elif mutation == "pressure":
                result["pressure_summary"] = {"roof": 0.}
            else:
                del result["ood_score"]
        return result

    monkeypatch.setattr(qualify, "_request", changed)
    report = qualify.qualify_service(trained[0], native_pack, log_dir=tmp_path)
    assert not report["passed"]
    assert any(error.startswith("/evaluate") for error in report["errors"])


@pytest.mark.parametrize("mutation", ["delta", "trust", "reason", "style", "scope", "unavailable"])
def test_compare_route_semantics_cannot_drift(trained, native_pack, tmp_path, monkeypatch, mutation):
    original = qualify._request

    def changed(client, path, **kwargs):
        result = original(client, path, **kwargs)
        if path == "/compare":
            if mutation == "delta":
                result["delta"]["cd"] += .01
            elif mutation == "trust":
                result["trusted"] = not result["trusted"]
            elif mutation == "reason":
                result["reject_reasons"]["design_b"] = "wrong reason"
            elif mutation == "style":
                result["same_body_style"] = not result["same_body_style"]
            elif mutation == "scope":
                result["comparison_kind"] = "validated fields"
            else:
                result["unavailable_pressure_regions"] = []
        return result

    monkeypatch.setattr(qualify, "_request", changed)
    report = qualify.qualify_service(trained[0], native_pack, log_dir=tmp_path)
    assert not report["passed"]
    assert any(error.startswith("/compare") for error in report["errors"])


def test_agent_gate_uses_exact_service_thresholds(trained, native_pack, tmp_path, monkeypatch):
    seen = []
    original = qualify.designer.rejection_reason

    def tracked(evaluation, **kwargs):
        seen.append(kwargs)
        return original(evaluation, **kwargs)

    monkeypatch.setattr(qualify.designer, "rejection_reason", tracked)
    report = qualify.qualify_service(trained[0], native_pack, log_dir=tmp_path)
    assert report["passed"], report["errors"]
    assert len(seen) == 5
    assert all(item["threshold"] == report["schema"]["uncertainty_threshold"] and
               item["ood_threshold"] == report["schema"]["ood_score_threshold"] for item in seen)


def test_remote_scope_is_explicit_and_identity_check_precedes_scoring(trained, native_pack, tmp_path, monkeypatch):
    original = qualify._client

    @contextmanager
    def local_transport(checkpoint, service_url, log_dir):
        assert service_url == "https://example.invalid"
        with original(checkpoint, None, tmp_path) as client:
            yield client

    monkeypatch.setattr(qualify, "_client", local_transport)
    report = qualify.qualify_service(trained[0], native_pack, service_url="https://example.invalid")
    assert report["passed"] and report["scope"] == "remote_http"


def test_changed_agent_gate_fails_rejected_reference(trained, native_pack, tmp_path, monkeypatch):
    monkeypatch.setattr(qualify.designer, "rejection_reason", lambda *args, **kwargs: None)
    report = qualify.qualify_service(trained[0], native_pack, log_dir=tmp_path)
    assert not report["passed"]
    assert any(error.startswith("agent gate") for error in report["errors"])


def test_http_error_status_is_visible_without_response_body():
    class Response:
        status_code = 503

        def json(self):
            raise AssertionError("error bodies must not be copied to the report")

    class Client:
        def get(self, path):
            return Response()

    with pytest.raises(ValueError, match="/health returned HTTP 503"):
        qualify._request(Client(), "/health")


@pytest.mark.parametrize("url", ["file:///tmp/aero", "http://user:secret@localhost:8000", "http://localhost/?token=x"])
def test_remote_url_rejects_credentials_and_unsupported_scheme(url):
    with pytest.raises(ValueError):
        qualify._Remote(url)


def test_cli_writes_complete_report_and_refuses_overwrite(trained, native_pack, tmp_path):
    references, output = tmp_path / "references.json", tmp_path / "qualification.json"
    references.write_text(json.dumps(native_pack), encoding="utf-8")
    args = ["--checkpoint", trained[0], "--references", str(references), "--out", str(output)]
    assert qualify.main(args) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] and report["scope"] == "in_process_asgi"
    before = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        qualify.main(args)
    assert exc.value.code == 2 and output.read_bytes() == before


def test_cli_saves_failed_qualification_report(trained, native_pack, tmp_path):
    references, output = tmp_path / "references.json", tmp_path / "failure.json"
    native_pack["checkpoint"]["sha256"] = "0" * 64
    references.write_text(json.dumps(native_pack), encoding="utf-8")
    assert qualify.main(["--checkpoint", trained[0], "--references", str(references),
                         "--out", str(output)]) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["passed"] and not report["evaluations"]
