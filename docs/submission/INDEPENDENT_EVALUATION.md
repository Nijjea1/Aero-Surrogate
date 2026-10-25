# Independent final evaluation: 27 October 2026

Evaluated source: `43191bf2c9b9d87ffc313e63b3ff36ee127a0c92` (clean).
This report records a fresh local regrade, not a copy of the incoming reports.
The original bundle remains unchanged. Reports and checkpoint bytes stay outside Git.

## Inputs and reproduction

The incoming bundle's SHA256SUMS verified all 43 payloads. Run commands from the
repository root. Set `OMP_NUM_THREADS=2`, `MKL_NUM_THREADS=2` and
`OPENBLAS_NUM_THREADS=2`; device is CPU. On worktrees owned by another Windows
account, supply Git's per-process safe.directory exception so provenance resolves.

Copy the bundle's `data/drivaernet/parametric.csv` byte-for-byte to ignored
`data/drivaernet/parametric.csv` before training. `configs/scaling-final.yaml`
uses that portable location. The full training schedule is unchanged: 400 maximum
mean epochs with patience 40, followed by 300 maximum variance epochs with
patience 30; no epoch override. Model seed 0; nested subset seed 1000.

```sh
python -m src.eval.scaling train --config configs/scaling-final.yaml --out-dir runs/scaling-final --device cpu
python -m src.eval.harness --checkpoint runs/drivaernet-random/best.pt --ood-checkpoint BUNDLE/checkpoints/drivaernet-category/best.pt --data BUNDLE/data/drivaernet/parametric.csv --splits BUNDLE/splits/splits.drivaernet.json --scaling runs/scaling-final/scaling-005-random/best.pt runs/scaling-final/scaling-010-random/best.pt runs/scaling-final/scaling-025-random/best.pt runs/scaling-final/scaling-050-random/best.pt runs/scaling-final/scaling-100-random/best.pt --out runs/scalar-category.json --device cpu
python -m src.eval.fields --checkpoint BUNDLE/checkpoints/fields/best.pt --root BUNDLE/data/drivaerml/fields --metrics runs/scalar-category.json --device cpu --chunk-size 1024 --reference-workers 1
```

Replace `BUNDLE` with the extracted immutable bundle root. The published report
uses the same operations, with the separately produced field block merged into the
scalar report. Field evaluation source is recorded explicitly in the combined
report. Production weights were neither retrained nor replaced.

## Scalar and body-style holdout

625 unseen designs: R2 0.874971674, MAE 0.006160588; linear reference R2
0.835234104. One-sigma coverage is 0.6448 and nominal 90% coverage is 0.8656;
these intervals are too narrow. Sigma barely ranks error (Spearman approximately
0.066). Local warm prediction timing was approximately 0.17 ms per design;
this excludes preprocessing, requests and agent calls and is not an end-to-end
latency or a measured CFD speedup.

1,403 held-out notchbacks: R2 0.479211931, MAE approximately 0.010161.
The all-19-input detector has 100% detection/AUROC 1 because it observes an
unseen categorical style indicator. On the 16 aerodynamic parameters alone,
detection is 0.0263721 and AUROC 0.535656. This is not evidence of reliable
geometric-novelty detection. Sigma rises from 0.00810805 on familiar validation
styles to 0.00860770 on held-out styles, insufficient to track the error increase.
Solver-label variability is not a proven hard accuracy ceiling.

## Independently retrained scaling

Every checkpoint's exact validation and test IDs equal the incoming random split
(625 each). Every training-ID list equals its seed-1000 nested subset; no leakage
or resplitting. Training source recorded in all checkpoints: `7d69099`.
These are five separate runs, with their own validation-selected stopping points;
the full-data model is not the serving checkpoint. This is one seed, not a
multi-seed estimate of variability.

| Fraction | Train designs | Test R2 | MAE | Epochs run | Trainer wall seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5% | 146 | 0.778353948 | 0.008488172 | 127 | 7.30 |
| 10% | 292 | 0.809180227 | 0.007696925 | 114 | 6.31 |
| 25% | 729 | 0.849655691 | 0.006865906 | 86 | 8.15 |
| 50% | 1458 | 0.856522075 | 0.006565114 | 130 | 19.34 |
| 100% | 2915 | 0.878515693 | 0.005989044 | 126 | 27.51 |

Checkpoint SHA-256, in the same fraction order:

- 5%: `30ec68e8ff1686d37ecccdd885e49139831ea21184ec64e582a1e2790ce013b7`
- 10%: `1d6f3afb7faf1b83007e6fa9cd959c0f2cc3e50a662c2e919e9452b0dd6da22c`
- 25%: `4ee047c3ccf7cf9fa8e55471022c7f05d3ea24ea63c88c764ae802ae33024835`
- 50%: `7bc4fe5c9bcf67db52bdae91e1d61714562c497a19659db7d25cc3ebbad6e6f6`
- 100%: `b4b4c932e73a2836dc5efa36f99c553317d635d5fe5bb407c60e478aa488bd52`

## Surface pressure and pressure-only drag

All 500,000 stored query cells from five held-out runs (3, 11, 12, 22, 27)
were graded; these are extracted evaluation cells, not a claim of complete native
CFD surface quadrature. Reference inputs include all twenty training runs. The
pooled reference samples 20,000 points per training car, uses eight neighbours
in normalized position plus 0.25 times normal space, seed 0. Its JSON key
`nearest_training_car` is retained for compatibility; it does not retrieve a
single whole car.

Model Cp R2 0.5080304164, MAE 0.1455350373, one-sigma coverage 0.799376. The pooled nearest-training-point reference is better: R2 0.6905177845, MAE 0.1025096801. This comparison must accompany the model result.

Pressure-only integrated per-car Cd MAE 0.02184127671, mean absolute relative error 8.567636%. Runs 3, 11, 22 underpredict; 12 and 27 overpredict. The historical 0.0326087 MAE used a 4,096-point subsample and is a different scope, not the final stored-cell result.

The new field result agrees with the immutable incoming report within
0.000000262 Cp R2 and 0.000000215 pressure Cd MAE. Minor numerical differences
are consistent with changed inference chunking/platform floating-point arithmetic.
No lift or wall-shear result was independently recomputed. No generated viewer
mesh alignment, pressure projection or proposed-geometry CFD gain is validated.

## Evidence files and checkpoint identities

Coordinator-local outputs under ignored `.audit-track-b/final-evaluation/`:
`scalar-category.json`, `field-full.json`, `metrics-final.json`,
`scaling-verification.json`, and the full `scaling-train.log`.
`metrics-final.json` contains sanitized paths and source identities; raw incoming
reports must not be published. Root publication may reformat the report without
changing its measurements or provenance.

- Serving scalar: `49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`.
- Held-out-category evaluation only: `6b6033efbe90d01a10534f9cc0a93665e76227095da18fb47ca39f2d2db6848f`.
- Fields: `6a210c09b09205459f59d17b5ff913648587c9af086fc972e9090e3d69aa688c`.

Validation: four scaling tests passed; field tooling's 21 focused tests passed,
including fresh CLI output, direct/nested array roots and chunk equivalence.
The five complete training runs and fresh scalar/category/field grading exited 0.

DrivAerNet is CC BY-NC 4.0. DrivAerML is CC BY-SA 4.0; preserve each dataset's
attribution and licence independently. Derived software licensing does not
relicense the data.
