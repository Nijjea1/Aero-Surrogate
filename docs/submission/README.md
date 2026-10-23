# Local release package — deployment is the next external step

This release contains the working local application, independently graded scalar
model evidence, fifteen historical matched search trajectories, submission copy,
and checksum-verified local model packaging. Nothing here deploys the application,
uploads a video, publishes model weights or submits Devpost automatically.

## Start from the repository

Use Python 3.13 and an isolated environment. Install CPU Torch before the rest
when CUDA training is not needed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -rs
```

`constraints-release.txt` is a recorded tested environment, not a universal lock;
it is optional. [Reproduction tools](../../scripts/README.md) document dependency
groups, local verification and any installation limitations observed here.

The pages serve pinned three.js and IBM Plex fonts from `web/vendor/`, retaining
their licences. Rendering the designer does not require a CDN connection. Live
Nemotron proposals still need Token Factory connectivity and an operator key.

## Build or install the trusted models

The team can build the local ZIP from existing qualified artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/package_release.py --include-fields
```

Outputs:

- `runs/release/aero-surrogate-models.zip`
- `runs/release/aero-surrogate-models.zip.sha256`
- [release-manifest.json](release-manifest.json), identifying exact payload bytes

The archive includes the scalar model, optional experimental pressure model,
native references, frozen split package, metrics, benchmark, attribution notices,
and historical logs for the Runs page. It excludes `.env`, keys and the full raw
datasets. A separately published immutable download URL remains pending.

For another checkout, receive the ZIP plus its SHA-256 through a trusted channel,
then run:

```powershell
python scripts/install_artifacts.py --archive PATH_TO_ZIP --sha256 TRUSTED_ARCHIVE_SHA256 --destination .
```

An HTTPS archive URL may replace the local path after the team publishes it.
The installer authenticates the archive and verifies every declared payload
before changing destination files. It rejects traversal, symlinks, duplicate
names, unlisted files, excessive sizes and accidental overwrites. `--replace`
is an explicit trusted update option; do not use it on a live run-log directory.

Model files stay outside Git. [Dataset/model notices](DATA_AND_MODEL_NOTICES.md)
travel with the ZIP. The source-code MIT licence does not relicense datasets.

## Run and verify

In the API shell:

```powershell
$env:AERO_CHECKPOINT="runs/drivaernet-random/best.pt"
$env:AERO_FIELD_CHECKPOINT="runs/fields/best.pt"  # optional experimental overlay
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --port 8000 --workers 1 --no-proxy-headers
```

In a second shell:

```powershell
.\.venv\Scripts\python.exe web/serve.py
```

Open `http://localhost:5173`. `/health.synthetic` must be false. The full scalar
SHA-256 is in `/health.provenance.checkpoint_sha256` (the top-level hash is a
short display identifier). Expected scalar:
`49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`.

The default baseline fastback should return Cd about 0.22456. Pick **Edge of the
box** to see a genuine joint-extreme rejection, then reset. The field overlay
is experimental and its confidence does not feed the Cd gate. Cl is unavailable.

To enable a real agent, configure `NEBIUS_API_KEY` locally (or in the operator's
uncommitted `.env`), review `.env.example` budget settings and use the Runs page.
Never enter the operator/admin key into the webpage or publish it in recordings.
The archive's existing matched runs are historical records, not new provider calls.

Local HTTP verification is repeatable and makes no paid calls:

```powershell
.\.venv\Scripts\python.exe scripts/check_local.py --checkpoint runs/drivaernet-random/best.pt --field-checkpoint runs/fields/best.pt --out runs/local-wiring-new.json
```

The browser rehearsal records actual browser checks separately from the API-only
check. Install its optional dependencies and browser binaries:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-browser.txt
$env:PLAYWRIGHT_BROWSERS_PATH=".audit-track-b/browsers"
.\.venv\Scripts\python.exe -m playwright install chromium firefox
.\.venv\Scripts\python.exe scripts/rehearse_browser.py --project-root . --attempts 20 --out runs/browser-rehearsal-new
```

This default makes no provider calls. Only `--live-runs N` explicitly opts into
new paid runs under the isolated `--max-spend` application guard. See its `--help`
and the completion report for the command and scope actually verified here.

## Evidence and submission documents

- [Scalar and matched evidence](evidence/README.md): actual release-model grading,
  all fifteen matched trajectories and their original experiment identities.
- [Submission description](submission.md): copy ready for the final form, with
  pending external URLs explicitly marked.
- [Tooling feedback](tooling-feedback.md): measured observations and concrete requests.
- [Video script](video-script.md): under-three-minute recording plan; distinguish
  raw browser recordings from a finished narrated submission video.
- [Teammate artifact handoff](teammate-artifacts.md): missing category/full-field inputs.
- [Original audit](../REPO_AUDIT_2026-10-27.md): findings at the pre-fix revision.

The historical audit is preserved. It is not a claim that all listed findings
remain open after the release fixes; use the completion report for final status.

## Deployment handoff

The team must supply the public host/domain, provision the API and dynamic web
server, mount the selected models, persist logs and spend accounting, configure
HTTPS/CORS/proxy trust, and fund judge-period inference. Follow the [API
runbook](../../src/service/DEPLOY.md) and [web runbook](../../web/DEPLOY.md).

After deployment, repeat the judge walkthrough externally and verify model hashes
against the manifest. Replace pending URLs in the submission copy, record the
final deployed footage, upload the public YouTube video and submit Devpost. The
local browser tests do not establish public HTTPS availability or remote uptime.
