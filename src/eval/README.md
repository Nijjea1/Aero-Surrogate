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
| `--out` | default `runs/metrics.json` |

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
