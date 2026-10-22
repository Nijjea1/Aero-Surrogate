"""The evaluation harness: each metric against a known answer, the splits, and
an end-to-end run on a small synthetic table."""

from __future__ import annotations

import csv
import json
from statistics import NormalDist

import numpy as np
import pytest

from src.eval import data as D
from src.eval import harness
from src.eval.metrics import (
    Mahalanobis,
    auroc,
    calibration_curve,
    field_error,
    gate_report,
    noise_ceiling,
    ood_detection,
    per_category,
    r2_score,
    regression_metrics,
    spearman,
)
from src.eval.models import LinearModel, MeanModel

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def test_r2_known_values():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2_score(y, y) == 1.0
    assert r2_score(y, np.full(4, y.mean())) == 0.0
    assert r2_score(y, y[::-1]) == pytest.approx(-3.0)
    assert np.isnan(r2_score([2.0, 2.0], [1.0, 3.0]))


def test_regression_metrics_do_not_turn_solver_spread_into_performance_bound():
    m = regression_metrics([0.0, 1.0, 2.0], [0.0, 1.0, 3.0], r2_ceiling=0.8)
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert m["max_error"] == 1.0 and m["n"] == 3
    assert "share_of_ceiling" not in m and "r2_ceiling" not in m


def test_per_category_splits_by_label():
    y = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
    pred = np.array([1.0, 2.0, 3.0, 20.0, 20.0, 20.0])
    out = per_category(y, pred, ["a", "a", "a", "b", "b", "b"])
    assert out["a"]["r2"] == 1.0
    assert out["b"]["r2"] == 0.0
    assert set(out) == {"a", "b"}


def test_noise_ceiling():
    y = np.array([0.0, 2.0])  # variance 1
    ref = noise_ceiling(y, [0.5, 0.5])
    assert ref["solver_spread_ratio_reference"] == pytest.approx(0.75)
    assert ref["is_performance_bound"] is False
    assert "r2_ceiling" not in ref


def test_calibration_is_on_the_diagonal_for_a_calibrated_model():
    rng = np.random.default_rng(0)
    sigma = rng.uniform(0.5, 2.0, 40_000)
    y = rng.normal(0.0, sigma)
    cal = calibration_curve(y, np.zeros_like(y), sigma)
    assert cal["coverage_1sigma"] == pytest.approx(0.683, abs=0.01)
    assert cal["coverage_2sigma"] == pytest.approx(0.954, abs=0.01)
    assert cal["miscalibration"] < 0.01
    assert cal["z_std"] == pytest.approx(1.0, abs=0.02)
    # Sigma ranks the risk: the RMSE rises bin by bin.
    rmse = [b["rmse"] for b in cal["sigma_bins"]]
    assert rmse == sorted(rmse)
    assert cal["sigma_error_spearman"] > 0.2


def test_calibration_sees_overconfidence():
    rng = np.random.default_rng(1)
    y = rng.normal(0.0, 1.0, 20_000)
    cal = calibration_curve(y, np.zeros_like(y), np.full_like(y, 0.5), noise_floor=1.0)
    assert cal["coverage_1sigma"] < 0.45
    assert cal["z_std"] > 1.9
    assert cal["share_below_label_std_reference"] == 1.0
    assert "noise_floor" not in cal


def test_calibration_without_sigma():
    assert calibration_curve([1.0], [1.0], [0.0])["available"] is False


def test_spearman_and_auroc():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
    assert auroc([0, 1, 2], [3, 4, 5]) == 1.0
    assert auroc([3, 4, 5], [0, 1, 2]) == 0.0
    assert auroc([1, 1], [1, 1]) == 0.5


def test_ood_detection_rates():
    out = ood_detection([1.0, 2.0, 3.0, 10.0], [5.0, 6.0, 1.0, 7.0], threshold=4.0)
    assert out["detection_rate"] == 0.75
    assert out["false_alarm_rate"] == 0.25


def test_gate_report():
    out = gate_report([0.0, 0.0, 0.0, 0.0], [0.1, 0.1, 0.5, 0.5], [True, True, False, False])
    assert out["pass_rate"] == 0.5
    assert out["mae_accepted"] == pytest.approx(0.1)
    assert out["mae_rejected"] == pytest.approx(0.5)
    assert gate_report([0.0], [0.1], [True])["mae_rejected"] is None


def test_mahalanobis_matches_the_training_detector_formula():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(500, 4)) @ np.array([[2, 0, 0, 0], [1, 1, 0, 0], [0, 0, 3, 0], [0, 0, 0, 0.5]])
    det = Mahalanobis.fit(x[:400], x[400:], quantile=0.99)
    cov = np.cov(x[:400], rowvar=False)
    cov += 1e-3 * np.eye(4) * np.trace(cov) / 4
    delta = x[400:] - x[:400].mean(axis=0)
    expected = np.sqrt(np.einsum("ij,jk,ik->i", delta, np.linalg.inv(cov), delta))
    np.testing.assert_allclose(det.score(x[400:]), expected)
    assert det.threshold == pytest.approx(np.quantile(expected, 0.99))
    # Far away scores far.
    assert det.score(x[:1] * 50)[0] > det.threshold


def test_field_error():
    true = np.array([1.0, -1.0, 0.5, -0.5])
    out = field_error(true, true * 1.1, regions=[0, 0, 1, 1], region_names=["roof", "hood"])
    assert out["relative_l2"] == pytest.approx(0.1)
    assert set(out["per_region"]) == {"roof", "hood"}


def test_reference_models():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 3))
    y = x @ np.array([1.0, -2.0, 0.5]) + 4.0
    mu, sigma = LinearModel(x, y).predict(x)
    np.testing.assert_allclose(mu, y, atol=1e-2)
    assert np.all(sigma == sigma[0])
    mu, _ = MeanModel(x, y).predict(x[:5])
    assert np.all(mu == pytest.approx(y.mean()))


# --------------------------------------------------------------------------
# Data and splits
# --------------------------------------------------------------------------

STYLES = {"F": "fastback", "N": "notchback", "E": "estateback"}
PARAMS = ["A_Car_Length", "A_Car_Width", "B_Ramp_Angle"]


def _write_table(path, n_per_style=60, seed=0, raw=True):
    """A DrivAerNet-shaped table: Cd linear in the dials plus label noise."""
    rng = np.random.default_rng(seed)
    offset = {"F": 0.0, "N": 0.01, "E": 0.02}
    rows = []
    for prefix in STYLES:
        for k in range(n_per_style):
            p = rng.uniform(-1, 1, len(PARAMS))
            cd = 0.26 + offset[prefix] + 0.01 * p[0] - 0.005 * p[1] + 0.003 * p[2] + rng.normal(0, 0.001)
            row = {"Experiment": f"{prefix}_S_WWC_WM_{k:03d}", **dict(zip(PARAMS, p)),
                   "Average Cd": cd, "Std Cd": 0.001, "Average Cl": 0.0}
            if not raw:
                row["Category"] = STYLES[prefix]
            rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def test_load_table_derives_body_style(tmp_path):
    t = D.load_table(_write_table(tmp_path / "raw.csv"))
    assert len(t.ids) == 180
    assert dict(zip(t.ids, t.categories))["N_S_WWC_WM_003"] == "notchback"
    prepared = D.load_table(_write_table(tmp_path / "prep.csv", raw=False))
    assert prepared.categories == t.categories


def test_design_matrix_fills_body_style_columns(tmp_path):
    t = D.load_table(_write_table(tmp_path / "raw.csv"))
    ids = ["E_S_WWC_WM_000", "F_S_WWC_WM_001"]
    x = D.design_matrix(t, ids, ["A_Car_Width", "cat_fastback", "cat_estateback"])
    assert x[:, 1].tolist() == [0.0, 1.0]
    assert x[:, 2].tolist() == [1.0, 0.0]
    assert x[0, 0] == t.column("A_Car_Width", ids[:1])[0]


def test_splits_are_disjoint_and_deterministic(tmp_path):
    t = D.load_table(_write_table(tmp_path / "raw.csv"))
    a, b = D.build_splits(t), D.build_splits(t)
    assert a["random"] == b["random"] and a["category"] == b["category"]
    D.check_disjoint(a["random"])
    D.check_disjoint(a["category"])
    assert sum(len(a["random"][k]) for k in ("train", "val", "test")) == 180
    assert a["category"]["held_out_category"] == "notchback"
    cats = dict(zip(t.ids, t.categories))
    assert {cats[i] for i in a["category"]["test"]} == {"notchback"}
    assert "notchback" not in {cats[i] for i in a["category"]["train"]}


def test_overlapping_splits_are_refused():
    with pytest.raises(ValueError, match="train/test"):
        D.check_disjoint({"train": ["a", "b"], "val": ["c"], "test": ["b"]})


def test_splits_file_round_trip(tmp_path):
    payload = {"seed": 0, "random": {"train": ["a"], "val": ["b"], "test": ["c"]},
               "held_out_category": {"train": ["a"], "val": ["b"], "test": ["d"],
                                     "_held_out_category": ["notchback"]}}
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(payload))
    s = D.load_splits(path)
    assert s["category"]["held_out_category"] == "notchback"
    assert s["random"]["test"] == ["c"]


def test_full_handover_splits_load_without_manual_extraction(tmp_path):
    params = {"seed": 7, "random": {"train": ["a"], "val": ["b"], "test": ["c"]},
              "held_out_category": {"train": ["a"], "val": ["b"], "test": ["d"],
                                    "_held_out_category": ["notchback"]}}
    path = tmp_path / "handover.json"
    path.write_text(json.dumps({"generated": "historical", "stage1": {"parameters": params}}))
    result = D.load_splits(path)
    assert result["random"] == params["random"] and result["seed"] == 7
    assert result["category"]["held_out_category"] == "notchback"


@pytest.mark.parametrize("payload,match", [
    ({"stage1": {}}, "stage1.parameters"),
    ({"random": {"train": ["a", "a"], "val": ["b"], "test": ["c"]}}, "duplicate"),
    ({"random": {"train": ["a"], "val": ["b"], "test": ["a"]}}, "train/test"),
    ({"random": {"train": "a", "val": ["b"], "test": ["c"]}}, "list of design IDs"),
    ({"random": {"train": ["a"], "val": [], "test": ["c"]}}, "list of design IDs"),
])
def test_bad_handover_splits_fail_before_grading(tmp_path, payload, match):
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match=match):
        D.load_splits(path)


def test_matches_the_training_split_algorithm():
    """Mirror of src.data.dataset.make_splits, written out, so a change to
    either side shows up here."""
    ids = [f"F_{i:03d}" for i in range(100)]
    order = np.random.default_rng(0).permutation(100)
    shuffled = [sorted(ids)[i] for i in order]
    s = D.make_splits(list(reversed(ids)))
    assert s["test"] == sorted(shuffled[:15])
    assert s["val"] == sorted(shuffled[15:30])


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_harness_runs_without_a_checkpoint(tmp_path):
    data = _write_table(tmp_path / "raw.csv", n_per_style=80)
    m = harness.evaluate(data_path=data, n_synthetic=300)
    out = harness.write(m, tmp_path / "metrics.json")
    loaded = json.loads(out.read_text(encoding="utf-8"))  # valid JSON, no NaN

    assert loaded["format"] == harness.METRICS_FORMAT
    assert loaded["data"]["n_designs"] == 240
    assert loaded["inputs"][-3:] == ["cat_fastback", "cat_notchback", "cat_estateback"]
    # Nothing is quoted that was not measured.
    assert loaded["headline"]["r2_in_distribution"] is None
    assert loaded["headline"]["r2_linear_reference"] > 0.8
    assert any("no --checkpoint" in w for w in loaded["warnings"])

    lin = loaded["in_distribution"]["references"]["linear"]
    assert lin["overall"]["r2"] > 0.8
    assert set(lin["per_category"]) == {"fastback", "notchback", "estateback"}
    assert loaded["out_of_distribution"]["held_out_category"] == "notchback"

    det = loaded["ood_detection"]
    assert det["harness_detector"]["corners_flagged"] > 0.9
    assert det["held_out_style"]["harness_detector_all_inputs"]["detection_rate"] == 1.0
    curve = loaded["data_scaling"]["linear_reference"]
    assert [p["n_train"] for p in curve] == sorted(p["n_train"] for p in curve)
    assert loaded["field"]["available"] is False
    assert "share_of_ceiling" not in loaded["headline"]
    assert loaded["data"]["label_variability"]["is_performance_bound"] is False


def _checkpoint(path, table, split, columns, held_out=None):
    """A param_mlp checkpoint in the layout src/train/train.py saves, with
    untrained weights: the test is about the plumbing, not the numbers."""
    torch = pytest.importorskip("torch")
    model_mod = pytest.importorskip("src.train.model")
    if not hasattr(model_mod, "build_model") or not hasattr(model_mod, "split_prediction"):
        pytest.skip("src.train.model has no parameter model yet (contracts 2.0.0 not merged)")
    cfg = {"model": {"kind": "param_mlp", "in_dim": len(columns), "hidden_dim": 16, "n_layers": 1}}
    torch.manual_seed(0)
    model = model_mod.build_model(cfg)
    x = D.design_matrix(table, split["train"], columns)
    y = table.column(D.TARGET, split["train"])
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std < 1e-9] = 1.0
    z = (x - mean) / std
    det = Mahalanobis.fit(z, (D.design_matrix(table, split["val"], columns) - mean) / std)
    torch.save({
        "state_dict": model.state_dict(), "config": cfg, "version": "test",
        "norm_stats": {"target_mean": float(y.mean()), "target_std": float(y.std()),
                       "param_mean": mean.tolist(), "param_std": std.tolist()},
        "preprocess": {"version": "params-v1", "param_columns": columns},
        "ood": {"mean": det.mean.tolist(), "cov_inv": det.cov_inv.tolist(), "threshold": det.threshold,
                "quantile": 0.99, "n_train": len(x), "train_scores": {}},
        "test_metrics": {"r2": 0.5}, "held_out_category": held_out, "schema_version": "2.0.0",
    }, path)
    return model_mod, model, cfg


def test_harness_grades_checkpoints(tmp_path):
    torch = pytest.importorskip("torch")
    table = D.load_table(_write_table(tmp_path / "raw.csv", n_per_style=80))
    splits = D.build_splits(table)
    columns = PARAMS + ["cat_fastback", "cat_notchback", "cat_estateback"]
    model_mod, model, _ = _checkpoint(tmp_path / "random.pt", table, splits["random"], columns)
    # A different set of inputs, in a different order: each checkpoint must
    # still get its own columns in its own order.
    _checkpoint(tmp_path / "category.pt", table, splits["category"], list(reversed(columns[1:])), "notchback")

    m = harness.evaluate(data_path=table.path, checkpoint=tmp_path / "random.pt",
                         ood_checkpoint=tmp_path / "category.pt", scaling=[tmp_path / "random.pt"],
                         n_synthetic=200)
    m = json.loads(json.dumps(harness._clean(m), allow_nan=False))

    # The harness's prediction is the model's, denormalised the way HANDOFF.md says.
    ckpt = torch.load(tmp_path / "random.pt", weights_only=False)
    x = D.design_matrix(table, splits["random"]["test"], columns)
    z = (x - np.array(ckpt["norm_stats"]["param_mean"])) / np.array(ckpt["norm_stats"]["param_std"])
    with torch.no_grad():
        mu, _ = model_mod.split_prediction(model(torch.from_numpy(z.astype(np.float32))))
    mu = mu.numpy() * ckpt["norm_stats"]["target_std"] + ckpt["norm_stats"]["target_mean"]
    y = table.column(D.TARGET, splits["random"]["test"])
    assert m["in_distribution"]["overall"]["r2"] == pytest.approx(r2_score(y, mu), abs=1e-6)

    h = m["headline"]
    assert h["r2_in_distribution"] is not None and h["r2_held_out_category"] is not None
    assert h["held_out_category"] == "notchback" and h["latency_ms"] > 0
    assert m["models"]["in_distribution"]["sha256"]
    assert set(m["in_distribution"]["gate"]) >= {"calibrated"}
    assert m["ood_detection"]["held_out_style"]["checkpoint_detector"]["detection_rate"] == 1.0
    assert m["data_scaling"]["model"][0]["n_train"] == len(splits["random"]["train"])
    # The untrained model disagrees with its own report, and the checkpoints
    # take their inputs in different orders: both are flagged.
    assert any("differs from the checkpoint's own report" in w for w in m["warnings"])
    assert any("different inputs" in w for w in m["warnings"])


def test_cli_writes_metrics(tmp_path, capsys):
    data = _write_table(tmp_path / "raw.csv")
    assert harness.main(["--data", str(data), "--out", str(tmp_path / "m.json")]) == 0
    assert (tmp_path / "m.json").exists()
    assert "R2 ceiling" in capsys.readouterr().out


def test_missing_data_says_how_to_get_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="drivaernet --fetch"):
        D.find_data(str(tmp_path / "nope.csv"))


def test_normal_quantiles_used_for_coverage():
    # The 68.3% level is the 1-sigma interval.
    assert NormalDist().inv_cdf(0.5 + 0.683 / 2) == pytest.approx(1.0, abs=0.002)
