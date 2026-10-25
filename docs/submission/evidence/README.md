# Release evaluation evidence

The [scalar metrics](../../../web/results/metrics.json) independently grade the
actual served checkpoint `49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`
on its frozen DrivAerNet random split: 2,915 train, 625 validation, 625 test.
The CSV hash and split counts are recorded in that file. It reports R² 0.87497,
MAE 0.0061606, 1σ coverage 64.48%, and nominal 90% interval coverage 86.56%.
The linear reference uses the same inputs and split. These are parameter-model
results and cannot be directly compared with published mesh-input leaderboards.

`Std Cd` is descriptive solver-coefficient variability. It is not automatically
the error variance of `Average Cd`, so the historical ratio under
`data.label_variability` is explicitly **not a performance bound**. Headline
R² is not divided by that ratio. Error bars and domain screening do not establish
physical correctness, and sigma alone barely ranks held-out errors.

The verified incoming category checkpoint and all thirty DrivAerML arrays now
support fresh independent release grading. Category results are R² 0.4792 /
MAE 0.01016 on 1,403 held-out notchbacks. Fields score Cp R² 0.5080 /
MAE 0.14554 across 500,000 stored query cells; the pooled nearest-training-point
reference is stronger (R² 0.6905 / MAE 0.10251). Pressure-only per-car drag MAE
is 0.0218415, not the earlier 4,096-point/car estimate of 0.0326. Use the fresh
current report for final provenance and timing; old evidence snapshots retain
their original dates and are not silently relabelled. Parameter-space detector experiments remain available:
100% detection of sampled joint-extreme corners and 0.32% false alarms on the
random-split test set. The held-out-style detector comparison shows that the
all-input result depends on body-style indicators; the aero-only detector catches
about 2.64% of those styles. This is not learned detection of unknown physics.

## Historical matched search experiment

The [benchmark](../../../web/results/benchmark.json) aggregates all fifteen
[matched summaries](matched/). The exporter revalidates their protocol, every
recorded predictor call, gate decision, best-so-far curve and budget accounting
using `src.agent.comparison.summarize_pair`. Summary source hashes and published
hashes are recorded in the benchmark. The experiment retains its original Git
revision and checkpoint identity; publication is not a new experiment.

Five runs per method, one all-zero fastback start, 25 scored calls including the
baseline. Rejected predictions consume calls; invalid and duplicate proposals
are refused before scoring. All three methods use the same gate and artifact.

| Method | Median predicted Cd reduction | Runs |
|---|---:|---:|
| Nemotron Super on Nebius Token Factory | 3.875% | 5 |
| CMA-ES | 3.841% | 5 |
| Random search | 2.793% | 5 |

These are **surrogate-predicted Cd changes**, not new CFD validation or evidence
of physical drag-force reduction. The small difference between agent and CMA-ES
does not establish superiority. Evaluation budgets are matched; wall-clock time,
LLM cost and proposal policies differ. Historical cost is not a current price
quote. The website uses negative percentages for decreases, so its direction
convention is the opposite sign of the positive reductions in this table.

The summaries include the complete scored parameter trajectories, model outputs,
gate outcomes, provider model ID, numeric constraints and historical timings/cost
where recorded. They do not contain provider credentials. Scalar model outputs
have no pressure summary and use labelled placeholder lift; this experiment is
Cd-guided rather than pressure-guided agent search.

The original [scored logbooks and lifecycle events](logs/) are also published,
preserving recorded hypotheses, outcomes and original experiment timestamps.
They are checked JSONL records with no provider credentials. The local artifact
bundle installs copies under `runs/logs/` so a new checkout can inspect historical
comparisons immediately; no run is replayed as a new model call.

## Reproduce

The serving ZIP supplies the scalar and optional field models plus native
references; reproducing the complete evaluation also needs the separate immutable
evaluation bundle and newly trained scaling checkpoints. Use the commands and
identities in [the independent evaluation report](../INDEPENDENT_EVALUATION.md).
They write private outputs under `runs/`, leaving the published report untouched.

For a scalar-only check from the trusted serving artifact and public table:

```bash
python -m src.eval.harness --checkpoint runs/drivaernet-random/best.pt --data data/drivaernet/parametric.csv --splits runs/handover/splits.json --out runs/regrade/scalar-only.json --label "Scalar-only reproduction"
python -m src.eval.benchmark --matched-summaries docs/submission/evidence/matched --out runs/regrade/benchmark.json
```

The harness accepts the complete nested handover directly. It rejects malformed,
overlapping or repeated split IDs. A fresh regrade produces new timing and
timestamp metadata; it should reproduce accuracy to numerical tolerance.
Scalar-only output deliberately omits category, field and scaling experiments;
do not replace the complete published metrics with that partial report.
