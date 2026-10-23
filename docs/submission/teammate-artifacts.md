# Missing evaluation artifacts — teammate handoff

Prepare the missing evaluation artifacts for Aero Surrogate. Preserve the
already qualified scalar serving checkpoint; no new training is required for
the current release. Send the local path of the completed bundle to the team.

1. Locate the held-out-category checkpoint. Expected SHA-256:
   `6b6033efbe90d01a10534f9cc0a93665e76227095da18fb47ca39f2d2db6848f`.
   Include config, seed, preprocessing and exact train/validation/test design
   IDs. Report any hash mismatch rather than silently substituting a model.
2. Supply the original extracted DrivAerML arrays for test runs **3, 11, 12,
   22, 27**, plus training arrays and metadata required by the independent
   evaluator's reference baselines. The checkpoint's recorded splits are the
   authority; do not split by points across cars.
3. Field checkpoint SHA-256:
   `6a210c09b09205459f59d17b5ff913648587c9af086fc972e9090e3d69aa688c`.
   Use the independent `src.eval.harness` and `src.eval.fields` tools after
   coordinating the current source revision with the team. Include exact
   commands, environment versions, raw reports and checksums. The harness now
   needs to support the complete handover split package, not only its inner
   `stage1.parameters` object.
4. Report ordinary R²/MAE, interval coverage, OOD results, Cp error and
   pressure-only integrated Cd error. Do not treat `Std Cd` as a proven hard
   R² ceiling. Disclose style-indicator novelty and distinguish fixed-query
   numerical parity from CFD accuracy on generated viewer geometry.
5. Package the artifacts outside Git with an inventory, checksums and dataset
   attribution/licence notices. Keep failed or unavailable results explicit.

The current release proceeds with the scalar model; category and full-test
field accuracy remain unavailable until these inputs are independently checked.
