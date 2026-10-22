# Tooling feedback — evidence-based draft

This feedback describes saved experiments, not current provider pricing or a new
production deployment. No AI Cloud or Serverless execution is asserted.

## What worked

NVIDIA Nemotron Super, served through Nebius Token Factory, supported typed tool calls
in a multi-step design workflow. The historical matched study includes five complete
25-evaluation Super runs on a real scalar checkpoint. Each run combined a scored
baseline with model proposals, gate feedback and an observed-result logbook.

The saved Super runs had a median logged cost of about **$0.0271 per run**, median
step latency of about **1.59 seconds**, and median sum of step times of about
**43.59 seconds**. About **99%** of step time was spent waiting on proposals, while
local surrogate scoring was much faster. These are application measurements from
the recorded experiment, not advertised service latency or present-day prices.

Nano was useful for inexpensive smoke checks and shorter runs. Tool-call results
could be fed back into the next proposal, making the agent responsive to rejected
designs and to hypotheses whose expected direction disagreed with the surrogate.

## Application work that mattered

Restricting each move to one or two parameters, refusing duplicate candidates,
remembering measured effects and checking expected direction against the observed
score improved the inspectability of the loop. These were our orchestration controls;
we do not attribute their absence to a platform defect.

We reserved estimated call cost before execution and maintained a persistent ledger.
Per-call, per-run and public daily limits made experimentation manageable. Billing can
be ambiguous after transport timeouts/retries, so this remains an application budget
guard rather than an absolute guarantee about provider billing.

## Actionable requests

1. Publish machine-readable model capability metadata for tool calling and thinking
   controls, with a small validated request/response example per model. This would
   reduce trial-and-error when choosing a low-latency model for a typed-tool workflow.
2. Expose stable request identifiers, retry semantics and billable-usage diagnostics
   that let a client reconcile a timed-out request against later usage. This would
   make conservative application cost accounting easier to implement correctly.
3. Provide an end-to-end example that reports proposal time, tool execution time,
   token usage and failure/correction frequency separately. Our surrogate was fast
   enough that model orchestration dominated, which a combined latency number hides.

These are requests based on integration needs. They do not assert an unverified
outage, incorrect charge or undocumented provider bug. Any such report should add
the actual model, request ID, timestamp, redacted payload and observed/expected result.

## NVIDIA scientific resources

DoMINO's global geometry encoding and multi-scale local neighbourhoods informed our
separate custom PyTorch pressure architecture. PhysicsNeMo external-aerodynamics
examples informed data preparation. We did not run a PhysicsNeMo-trained production
model or PhysicsNeMo-CFD benchmark and should not describe those as executed tooling.

## Before submission

- Attach the public matched protocol and run summaries.
- Preserve experiment timestamps and model IDs; do not relabel old runs as new tests.
- Include a concrete redacted failure/correction trace only if its exact cause is verified.
- Add hosting feedback only after the team deploys and has actual observations.
