# Remaining steps

The release is local-run: the application, with the real trained models, starts
from a fresh clone with the [Quick start](../../README.md#quick-start)
(`install_artifacts.py` plus `docker compose up --build`). That path was verified
from a fresh clone; see [compose-verification.json](evidence/compose-verification.json).
No public hosting is planned. Source, weights, thresholds and published metrics
are unchanged from the qualified release.

## Release and submission (team)

1. **GitHub release `v1.0.0`.** Attach `runs/release/aero-surrogate-models.zip`
   and its `.sha256` (archive SHA-256
   `70b2829ae2a5e0f1a8bc82f7d1f2321cad1d8fba62ff3f191c4a53b399834a93`). Keep the
   dataset licences and the note that MIT does not cover the weights in the
   release notes. Confirm a logged-out download installs with the Quick start.
2. **Video.** Record the local application with [video-script.md](video-script.md),
   including a genuine live Nemotron run. Export below three minutes, upload
   publicly to YouTube and check playback logged out.
3. **Links.** Put the video URL in the README and [submission.md](submission.md).
4. **Devpost.** Use [submission.md](submission.md) and
   [tooling-feedback.md](tooling-feedback.md). The release page is the test-build
   link; testing instructions are the Quick start, noting that live runs need the
   judge's own Token Factory key and that the video shows one. Submit (not only
   save) and have a teammate check every link.
5. **Spend.** Stop the local stack after recording (`docker compose down`) and
   reconcile the Token Factory account with the application ledger.

## Optional: public hosting

If the team later hosts the application, follow the [API runbook](../../src/service/DEPLOY.md)
and [web runbook](../../web/DEPLOY.md): one small CPU VM, one API worker,
persistent `/data`, Caddy for HTTPS, exact CORS origin, deliberate public budgets.
Seed the data volume with the release's historical run logs (the compose `seed`
service shows how) and mount the scalar and field checkpoints from separate
directories, because both files are named `best.pt`. Verify `/health`, the full
scalar hash `49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`,
baseline Cd about 0.22456 and `python -m src.service.qualify --service-url` from
outside before sharing the URL.

No further model training or new features are needed for this release.
