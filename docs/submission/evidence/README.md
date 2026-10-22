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

The category checkpoint and full DrivAerML arrays were unavailable for this
release evaluation. Their independent model metrics are absent, not copied from
training-side reports. Parameter-space detector experiments remain available:
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

## Reproduce

After obtaining the trusted model artifact, public table and original handover:

```bash
python -m src.eval.harness --checkpoint runs/drivaernet-random/best.pt --data data/drivaernet/parametric.csv --splits runs/handover/splits.json --out web/results/metrics.json --label "Release scalar evaluation"
python -m src.eval.benchmark --matched-summaries docs/submission/evidence/matched --out web/results/benchmark.json
```

The harness accepts the complete nested handover directly. It rejects malformed,
overlapping or repeated split IDs. A fresh regrade produces new timing and
timestamp metadata; it should reproduce accuracy to numerical tolerance.
