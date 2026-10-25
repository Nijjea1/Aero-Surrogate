"""Release presentation must work without ungraded category/field experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _builder():
    spec = importlib.util.spec_from_file_location("readme_build", ROOT / "docs/readme/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_graphics_do_not_need_category_or_ceiling_fields():
    metrics = {
        "headline": {"r2_in_distribution": .875, "r2_linear_reference": .835,
                     "mae_in_distribution": .00616, "coverage_90": .8656,
                     "latency_ms": .42, "ood_false_alarm_rate": .0032},
        "in_distribution": {"n_test": 625, "per_category": {}},
        "field": {"available": False},
    }
    module = _builder()
    stats = module.stats(metrics, {"comparisons": []})
    block = module.results_block(metrics, {"comparisons": []})
    assert "86.6%" in stats
    assert "625 test designs" in stats
    assert "86.56%" in block
    assert "no independently graded release category checkpoint" in block
    assert "full release test-set grading is unavailable" in block
    assert "ceiling" not in stats.lower()
    assert "achievable" not in block.lower()


def test_agent_model_id_is_recognised_in_benchmark_copy():
    methods = [{"method": name, "label": label, "n_runs": 5,
                "final_pct": {"median": change, "best": change - .5},
                "found_at_median": 21, "gated_median": 0}
               for name, label, change in [
                   ("agent:nvidia/nemotron-3-super-120b-a12b", "Nemotron Super", -3.875),
                   ("cmaes", "CMA-ES", -3.841), ("random", "Random search", -2.793)]]
    block = _builder().bench_block({"comparisons": [{"body_style": "fastback", "budget": 25,
                                                     "methods": methods}]})
    assert "Nemotron Super" in block and "CMA-ES" in block and "Random search" in block
    assert "comparable" in block
    assert "not CFD-validated" in block
    assert "join this table once" not in block


def test_completed_field_results_keep_stronger_reference_and_sampling_scope():
    metrics = {
        "headline": {"r2_in_distribution": .875, "r2_linear_reference": .835,
                     "mae_in_distribution": .00616, "r2_held_out_category": .4792,
                     "held_out_category": "notchback"},
        "in_distribution": {"n_test": 625},
        "field": {"available": True, "test_runs": [3, 11, 12, 22, 27], "n_points": 500000,
                  "overall": {"r2": .50803, "mae": .145535, "coverage_1sigma": .799376},
                  "references": {"nearest_training_car": {"r2": .690518, "mae": .10251}},
                  "integrated_cd": {"mae": .0218415, "mean_relative_error": .085677}},
        "data_scaling": {"model": [{"n_train": 146, "r2": .75}, {"n_train": 2915, "r2": .87}]},
    }
    block = _builder().results_block(metrics, {})
    assert "500,000 stored query cells" in block
    assert "reference beats the field network" in block
    assert "0.691" in block and "0.508" in block
    assert "over-coverage" in block
    assert "full native CFD surface quadrature" in block
    assert "0.02184" in block and "8.57%" in block
    assert "evaluation-only" in block
    assert "do not replace the serving checkpoint" in block


def test_field_renderer_reports_stronger_baseline_without_viewer_calibration_claim():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    source = (ROOT / "web/results.js").read_text(encoding="utf-8")
    renderer = source.split("function renderField() {", 1)[1].split("// ---------------------------------------------------------------- scaling", 1)[0]
    script = """
const elements = {};
const $ = (id) => elements[id] ??= {};
const fmt = (v, digits) => Number(v).toFixed(digits);
const signed = fmt;
const charts = [];
const barChart = (el, rows) => charts.push(rows);
const state = {metrics: {field: {available:true, test_runs:[3,11,12,22,27], n_points:500000,
  overall:{r2:.50803,mae:.145535,coverage_1sigma:.799376},
  references:{nearest_training_car:{r2:.690518}},
  integrated_cd:{pred_mean:.23024,true_mean:.24575,mae:.0218415,mean_relative_error:.085677}}}};
""" + "function renderField() {" + renderer + """
renderField();
console.log(JSON.stringify({elements, charts}));
"""
    result = subprocess.run([node, "-e", script], capture_output=True, text=True,
                            encoding="utf-8", timeout=30, check=True)
    rendered = json.loads(result.stdout)
    lede = rendered["elements"]["field-lede"]["textContent"]
    note = rendered["elements"]["field-note"]["textContent"]
    assert "outperforms the network" in lede and "over-coverage" in lede
    assert "stored surface query cells" in lede
    assert "not full native CFD surface quadrature" in note
    assert "orientation and accuracy remain unvalidated" in note
    assert rendered["charts"][0][1]["label"] == "Pooled nearest training points"


def test_missing_numeric_values_never_render_as_zero_or_nan():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    common = (ROOT / "web/common.js").as_uri()
    script = f"""
globalThis.location = {{search: ""}};
globalThis.window = {{matchMedia: () => ({{matches: false}})}};
const {{ fmt }} = await import({json.dumps(common)});
console.log(JSON.stringify([fmt(null, 2), fmt(undefined, 2), fmt(NaN, 2),
                          fmt(Infinity, 2), fmt(0, 2), fmt(.875, 3)]));
"""
    result = subprocess.run([node, "--input-type=module", "-e", script], capture_output=True,
                            text=True, encoding="utf-8", timeout=30, check=True)
    values = json.loads(result.stdout)
    assert values[:4] == ["\u2013"] * 4
    assert values[4:] == ["0.00", "0.875"]
