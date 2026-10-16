# Evaluation harness

Owner: Paramveer. Grades the surrogate and writes `metrics.json`, the single
source for every number in the README, the charts and the video.

Avneet trains the model, and this module grades it independently. It reads the
data with its own loader, rebuilds or loads the exact training splits, checks
they don't overlap, and keeps the checkpoint's self-reported numbers beside its
own so any disagreement shows.

## Run it

```powershell
# The data: 1.4 MB, public (DrivAerNet, CC BY-NC 4.0). Lands in data/, which git ignores.
python -m src.data.drivaernet --fetch

# Splits, exactly as training used them
python -m src.data.dataset --dump-splits runs/splits.json --config configs/drivaernet.yaml

# Grade
python -m src.eval.harness `
    --checkpoint runs/drivaernet-random/best.pt `
    --ood-checkpoint runs/drivaernet-category/best.pt `
    --splits runs/splits.json
```

The harness also runs with no checkpoint at all. It then grades two reference
models fitted from the data (the mean, and a straight line) and runs the
distance checks, which need only the data.

Grading a checkpoint needs torch. The CPU build is enough:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

| Flag | What |
|---|---|
| `--checkpoint` | trained on the random split: the in-distribution numbers |
| `--ood-checkpoint` | trained with one body style held out (`--split category`): the out-of-distribution numbers |
| `--scaling a.pt b.pt …` | trained on growing subsets: the data-scaling curve |
| `--splits` | `runs/splits.json`. Without it, the splits are rebuilt with the training algorithm, which matches exactly. |
| `--cfd-hours` | CFD wall-clock per design. Adds the speedup figure. |
| `--out` | default `runs/metrics.json`. Use `web/results/metrics.json` to publish to the results page. |
| `--label` | one line shown on the results page, e.g. `"Final: H100 checkpoints, Oct 26"` |

The written file never contains absolute paths. Paths inside the repo become
repo-relative, and anything else is cut to its file name, so a published file
names nobody's disk.

## Data-scaling curve

```powershell
# train the stage-1 model on 5, 10, 25, 50 and 100 % of the training split
python -m src.eval.scaling train --config configs/drivaernet.yaml
# grade them on the fixed test set and write the curve into the published metrics
python -m src.data.dataset --config configs/drivaernet.yaml --dump-splits runs/splits.json
python -m src.eval.scaling update --metrics web/results/metrics.json --splits runs/splits.json `
    runs/scaling/scaling-005-random/best.pt runs/scaling/scaling-010-random/best.pt `
    runs/scaling/scaling-025-random/best.pt runs/scaling/scaling-050-random/best.pt `
    runs/scaling/scaling-100-random/best.pt
```

Training runs Track A's own trainer unchanged; only the random split's training
ids are cut, to nested subsets (the 10 % set contains the 5 % set), so
validation and the 625 test designs are the same for every point. `update`
replaces `data_scaling.model` and leaves every other graded number alone. On CPU
the five runs take about five minutes in all. The same checkpoints can instead
be passed to the harness with `--scaling` in a full regrade.

## Surface pressure (stage 2)

```powershell
# with a full grade
python -m src.eval.harness ... --field-checkpoint runs/fields/best.pt
# or into the published metrics, leaving every other number alone
python -m src.eval.fields --checkpoint runs/fields/best.pt --metrics web/results/metrics.json
```

Graded on the checkpoint's own held-out cars (`splits["test"]`, whole runs),
from `data/drivaerml/fields/run_<n>.npz` (`python -m src.data.fields --extract`).
It reports Cp R², MAE and σ coverage against two references with nothing
learned about the car (the training mean everywhere, and the nearest point
on any training car), the error per region, and the pressure drag
integrated over each car's predicted and measured field. A missing
checkpoint or missing arrays are reported in `field.reason`, never a crash.

## Agent against the baselines

```powershell
# CMA-ES and random search for seeds 0-4, then summarise every finished run
python -m src.eval.benchmark --baseline-seeds 5 --iterations 25 --body-style fastback `
    --checkpoint runs/drivaernet-random/best.pt
```

The command writes `web/results/benchmark.json`. Runs are grouped into fair
comparisons: same body style, budget, surrogate and constraints. Within each
comparison it reports, per method, the median and middle-half spread of best
trusted Cd at every evaluation, as a change against the run's own unmodified
car.

Unfinished runs are listed and left out, never silently dropped. A second
baseline run with the same seed is left out too, because it is the same run.

`--checkpoint` scores the baseline seeds in-process with the checkpoint the
service serves, using the same gate thresholds and provenance. That avoids
the service's per-minute rate limit. Without it they score over HTTP, like the
agent. Agent runs cost money, so start them yourself with
`python -m src.agent.designer --model super --iterations 25`.

## What's in metrics.json

| Key | What | Measured on |
|---|---|---|
| `headline` | The numbers the README and video quote. `null` wherever nothing was measured. | |
| `in_distribution` | R², MAE, RMSE, max error, per body style, the reliability diagram, the gate | random split, test set |
| `out_of_distribution` | the same, for a body style the model never saw | held-out style, all of it |
| `ood_detection` | Does the gate flag the unseen? Covers the held-out style (all inputs, and the 16 aero dials alone), jointly extreme in-bounds corners, random designs in the box, and sigma alone for comparison | |
| `gate` (in each split) | Pass rate, with the error on what passes vs what's stopped, at the checkpoint's calibrated threshold and at the contract's fallback | |
| `references` (in each split) | the same numbers for the mean and a straight line | |
| `data_scaling` | R² against training-set size, on one fixed test set | |
| `field` | Surface-pressure error. Not available until stage 2. | |
| `speed` | ms per design, one at a time and batched | |
| `warnings` | anything that makes a number less trustworthy than it looks | |

R² is reported against the noise ceiling as well as 1.0. DrivAerNet's
`Std Cd` is the solver's own run-to-run spread, so even a perfect model of the
mean can't beat about 0.91.

## Reading the reliability diagram

- **coverage:** for each level p, the share of designs whose error falls
  inside the predicted p-interval. On the diagonal means the error bars are the
  right width.
- **sigma_bins:** designs sorted by predicted sigma into ten bins, with the
  error actually observed in each. Rising means sigma ranks designs by risk,
  which is what the gate relies on.
