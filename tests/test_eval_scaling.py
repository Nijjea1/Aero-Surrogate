"""The data-scaling curve: nested subsets, the trainer left as it was, and metrics updated in place."""

from __future__ import annotations

import json

import pytest

from src.eval import data as D
from src.eval import harness, scaling

IDS = [f"F_D_{i:04d}" for i in range(400)]


def test_subsets_are_nested_deterministic_and_sized():
    small, big = scaling.nested_subset(IDS, 0.05), scaling.nested_subset(IDS, 0.25)
    assert len(small) == 20 and len(big) == 100
    assert set(small) <= set(big)
    assert small == scaling.nested_subset(IDS, 0.05)
    assert sorted(scaling.nested_subset(IDS, 1.0)) == sorted(IDS)
    with pytest.raises(ValueError):
        scaling.nested_subset(IDS, 0)


def test_training_cuts_only_the_train_split_and_restores_the_trainer(monkeypatch, tmp_path):
    from src.train import train as trainer

    full = {"train": IDS[:300], "val": IDS[300:350], "test": IDS[350:]}
    seen = []
    monkeypatch.setattr(trainer, "make_splits", lambda cfg, seed=0: {k: list(v) for k, v in full.items()})
    original = trainer.make_splits

    def fake_train(cfg, split_mode="random", device_str=None):
        splits = trainer.make_splits(cfg, seed=0)
        seen.append((cfg["run_name"], cfg["train"]["out_dir"], splits))
        return {"checkpoint": f"{cfg['train']['out_dir']}/{cfg['run_name']}-random/best.pt"}

    monkeypatch.setattr(trainer, "train", fake_train)
    monkeypatch.setattr("src.data.dataset.load_config", lambda path: {"run_name": "x", "train": {"out_dir": "runs"}})
    paths = scaling.train_fractions("cfg.yaml", (0.5, 0.1), out_dir=str(tmp_path))
    assert [name for name, _, _ in seen] == ["scaling-010", "scaling-050"]
    assert [len(s["train"]) for _, _, s in seen] == [30, 150]
    for _, out, s in seen:
        assert out == str(tmp_path)
        assert s["val"] == full["val"] and s["test"] == full["test"]
    assert len(paths) == 2 and trainer.make_splits is original


def test_update_replaces_only_the_model_curve(monkeypatch, tmp_path):
    metrics = {"splits": {"random": {"test": 50}}, "headline": {"r2_in_distribution": 0.87},
               "data_scaling": {"model": [], "linear_reference": [{"n_train": 10, "r2": 0.5}]},
               "warnings": ["no --scaling checkpoints: the data-scaling curve has the linear reference only", "keep me"]}
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    split = {"random": {"train": IDS[:300], "val": IDS[300:350], "test": IDS[350:]}}
    monkeypatch.setattr(D, "find_data", lambda p=None: "table.csv")
    monkeypatch.setattr(D, "load_table", lambda p: object())
    monkeypatch.setattr(D, "load_splits", lambda p: split)
    points = [{"n_train": 15, "r2": 0.6, "mae": 0.01}, {"n_train": 300, "r2": 0.85, "mae": 0.006}]
    monkeypatch.setattr(harness, "score_scaling", lambda table, rs, paths, device, warnings: points)

    out = scaling.update_metrics(str(path), "splits.json", ["a.pt", "b.pt"])
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["data_scaling"]["model"] == points == out["data_scaling"]["model"]
    assert saved["data_scaling"]["linear_reference"] == metrics["data_scaling"]["linear_reference"]
    assert saved["headline"] == metrics["headline"]
    assert saved["warnings"] == ["keep me"]


def test_update_refuses_a_different_test_set(monkeypatch, tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"splits": {"random": {"test": 625}}}), encoding="utf-8")
    monkeypatch.setattr(D, "find_data", lambda p=None: "table.csv")
    monkeypatch.setattr(D, "load_table", lambda p: object())
    monkeypatch.setattr(D, "load_splits", lambda p: {"random": {"train": [], "val": [], "test": IDS[:10]}})
    with pytest.raises(ValueError, match="test designs"):
        scaling.update_metrics(str(path), "splits.json", ["a.pt"])
