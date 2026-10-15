"""Tests for the track-A handover package.  OWNER: Avneet.

The handover is the last deliverable in my lane: the evaluation harness, not
me, produces the submission numbers. So what matters here is that the package
cannot mislead whoever reads it -- every checkpoint's contract is explicit,
the one that must not be reported is marked, and a stale checkpoint is caught
before it reaches the harness rather than after.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.train.handover import KNOWN_RUNS, build_splits, cmd_build, cmd_verify


REPO = Path(__file__).resolve().parents[1]


def test_every_known_run_declares_whether_it_is_reportable():
    """A checkpoint with a negative R2 exists in this repo on purpose. If it
    is not explicitly flagged, somebody evaluates it and reports it."""
    for entry in KNOWN_RUNS:
        assert "report" in entry, entry["name"]
        assert isinstance(entry["report"], bool)
        assert entry["role"].strip(), f"{entry['name']} has no stated role"
        assert (REPO / entry["config"]).exists(), entry["config"]


def test_the_negative_r2_model_is_marked_do_not_report():
    geometry = next(e for e in KNOWN_RUNS if e["name"] == "stage1-geometry")
    assert geometry["report"] is False
    assert "DO NOT REPORT" in geometry["role"]


def test_the_ensemble_is_the_reportable_stage1_headline():
    ens = next(e for e in KNOWN_RUNS if e["name"] == "stage1-ensemble")
    assert ens["report"] is True
    assert ens.get("ensemble") is True
    assert "REPORT THIS" in ens["role"]


def test_run_names_are_unique():
    names = [e["name"] for e in KNOWN_RUNS]
    assert len(names) == len(set(names))


def test_build_splits_covers_both_stages_and_records_reproducibility():
    """The harness has to be able to regenerate any split, so every entry
    carries the seed and the config that produced it."""
    splits = build_splits(REPO)
    assert "stage1" in splits and "stage2" in splits
    assert splits["generated"]

    for group in (splits["stage1"], splits["stage2"]):
        for name, entry in group.items():
            if "error" in entry:
                continue
            assert "seed" in entry, name
            assert "config" in entry, name


def test_stage1_splits_are_disjoint_and_by_design_id():
    splits = build_splits(REPO)
    params = splits["stage1"].get("parameters")
    if params is None or "error" in params:
        pytest.skip("DrivAerNet parametric table not fetched")

    random = params["random"]
    joined = [i for s in random.values() for i in s]
    assert len(joined) == len(set(joined)), "a design is in two splits"
    assert all(isinstance(i, str) for i in joined), "stage 1 splits by design id"

    held = params.get("held_out_category")
    if held:
        name = held["_held_out_category"][0]
        assert name
        # the held-out style must be entirely absent from train and val
        assert set(held["test"]).isdisjoint(held["train"])
        assert set(held["test"]).isdisjoint(held["val"])


def test_stage2_splits_are_by_run_not_by_point():
    """The one split mistake that would silently inflate every stage-2
    number, so it is asserted rather than trusted."""
    splits = build_splits(REPO)
    fields = splits["stage2"].get("fields")
    if fields is None:
        pytest.skip("fields not extracted")

    assert "run" in fields["split_by"]
    runs = fields["runs"]
    joined = [r for s in runs.values() for r in s]
    assert len(joined) == len(set(joined))
    assert all(isinstance(r, int) for r in joined), "stage 2 splits by run number"


def test_build_writes_a_complete_package(tmp_path):
    out = tmp_path / "handover"
    assert cmd_build(REPO, out) == 0

    for name in ("splits.json", "manifest.json", "README.md"):
        assert (out / name).exists(), name

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["checkpoints"], "no checkpoints described"
    for e in manifest["checkpoints"]:
        assert e["kind"] != "unknown", f"{e['name']} has no preprocess version"
        assert "safe_to_report" in e
        assert e["role"].strip()
        # Anything with a detector must say which space it measures, because
        # scoring the wrong space is silent noise.
        ood = e.get("ood_detector")
        if ood is not None:
            # "unknown" is allowed for a detector predating the space field,
            # but it must never be called trustworthy.
            assert ood["space"] in ("parameters", "features", "unknown")
            assert ood["trustworthy"] == (ood["space"] == "parameters")
            if ood["space"] != "parameters":
                assert ood["trustworthy"] is False


def test_readme_warns_against_the_three_known_traps(tmp_path):
    out = tmp_path / "handover"
    cmd_build(REPO, out)
    readme = (out / "README.md").read_text()
    # each of these cost real debugging time in this repo
    assert "safe_to_report" in readme
    assert "do not hardcode" in readme.lower()
    assert "integrated Cd" in readme
    assert "ceiling" in readme


def test_verify_detects_a_stale_checkpoint(tmp_path):
    """--verify exists because the architecture changed under a checkpoint
    once already and the mismatch only showed at load time. Corrupt a
    checkpoint and the verifier must fail rather than pass it to the harness."""
    out = tmp_path / "handover"
    cmd_build(REPO, out)
    manifest = json.loads((out / "manifest.json").read_text())

    entry = next((e for e in manifest["checkpoints"]
                  if e["kind"] == "params-v1"), None)
    if entry is None:
        pytest.skip("no params-v1 checkpoint trained yet")

    broken = tmp_path / "broken.pt"
    ckpt = torch.load(REPO / entry["checkpoint"], map_location="cpu",
                      weights_only=False)
    ckpt["state_dict"] = {k: v for i, (k, v) in enumerate(ckpt["state_dict"].items())
                          if i > 0}          # drop a tensor
    torch.save(ckpt, broken)

    manifest["checkpoints"] = [dict(entry, checkpoint=str(broken.relative_to(REPO))
                                    if broken.is_relative_to(REPO) else str(broken))]
    (out / "manifest.json").write_text(json.dumps(manifest))
    assert cmd_verify(REPO, out) == 1, "verify passed a broken checkpoint"


def test_verify_refuses_without_a_manifest(tmp_path):
    assert cmd_verify(REPO, tmp_path / "nothing") == 1


def test_a_detector_of_unknown_space_is_never_called_trustworthy(tmp_path):
    """Defaulting an old detector to "parameters" would claim the
    100%-detection result for one that might be the 26% one."""
    from src.train.handover import _describe

    out = _describe(next(e for e in KNOWN_RUNS if e["name"] == "stage1-geometry"),
                    REPO)
    if out is None or out.get("ood_detector") is None:
        pytest.skip("geometry checkpoint or its detector not present")
    ood = out["ood_detector"]
    if ood["space"] == "unknown":
        assert ood["trustworthy"] is False
        assert ood["stale"] is True
