"""New Track A handoff metadata must not overstate model training coverage."""

import copy

import pytest

from tests.test_torch_predictor import trained, _save  # noqa: F401
from src.contracts import BODY_STYLES
from src.service.predictor import CheckpointError, TorchPredictor
from src.service.provenance import describe_predictor


def test_new_trainer_provenance_is_exposed_without_disclosing_split_ids(trained):
    predictor = TorchPredictor(trained[0])
    metadata = describe_predictor(predictor)
    assert metadata["trained_body_styles"] == sorted(BODY_STYLES)
    assert metadata["training_git_revision"] == trained[1]["git_commit"]
    assert metadata["training_split_counts"] == {key: len(value) for key, value in trained[1]["splits"].items()}
    assert "splits" not in metadata


@pytest.mark.parametrize("styles", [["fastback"], ["fastback", "notchback", "notchback"],
                                    "fastback", [True, "notchback", "estateback"]])
def test_declared_incomplete_or_invalid_training_coverage_is_refused(trained, tmp_path, styles):
    checkpoint = copy.deepcopy(trained[1])
    checkpoint["trained_body_styles"] = styles
    with pytest.raises(CheckpointError, match="trained_body_styles"):
        TorchPredictor(_save(tmp_path, checkpoint))


@pytest.mark.parametrize("splits", [{"train": ["a"], "val": ["b"], "test": ["a"]},
                                   {"train": ["a", "a"], "val": ["b"], "test": ["c"]},
                                   {"train": ["a"], "val": ["b"], "test": []},
                                   {"train": ["a"]}])
def test_invalid_declared_splits_are_refused(trained, tmp_path, splits):
    checkpoint = copy.deepcopy(trained[1])
    checkpoint["splits"] = splits
    with pytest.raises(CheckpointError, match="split"):
        TorchPredictor(_save(tmp_path, checkpoint))


def test_legacy_absent_training_metadata_is_unknown_not_invented(trained, tmp_path):
    checkpoint = copy.deepcopy(trained[1])
    for key in ("trained_body_styles", "git_commit", "splits"):
        checkpoint.pop(key, None)
    metadata = describe_predictor(TorchPredictor(_save(tmp_path, checkpoint)))
    assert metadata["trained_body_styles"] is None
    assert metadata["training_git_revision"] is None
    assert metadata["training_split_counts"] is None


@pytest.mark.parametrize("seed", [True, float("nan"), "0", -1])
def test_invalid_training_seed_cannot_break_health_serialization(trained, tmp_path, seed):
    checkpoint = copy.deepcopy(trained[1])
    checkpoint["config"]["seed"] = seed
    with pytest.raises(CheckpointError, match="training seed"):
        TorchPredictor(_save(tmp_path, checkpoint))
