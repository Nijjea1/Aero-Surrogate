# Local reproduction tools

Run these commands from the repository root. Python 3.13 is the release target.
No script in this folder deploys the application or publishes a release.

## Install dependencies

Use an isolated virtual environment and its Python interpreter:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m pytest -q -rs
```

`requirements-service.txt` supports the labelled synthetic API and agent
orchestration without Torch. `requirements-model.txt` adds real scalar/field
inference; `requirements-web.txt` is the dynamic web/mesh server without Torch.
`requirements.txt` adds scalar training/evaluation dependencies. Raw DrivAerML
VTU/VTP extraction additionally requires `requirements-extract.txt` and the raw
dataset. No dataset or model weights are installed through pip.

`constraints-release.txt` records direct versions from the locally qualified
Windows/Python 3.13.14 environment. It is an environment snapshot, **not** a
complete platform-independent lock. Passing `-c constraints-release.txt` requests
those versions when available from your indexes; it is optional and is not used
to claim portability. The isolated pip installation attempted during release
work was blocked by network socket permissions to PyPI; the existing qualified
environment passed `pip check`. CI performs a clean CPU installation and tests
when run by GitHub Actions; adding the workflow does not establish that it has
already passed remotely.

Pytest is configured to collect only `tests/`, so old ignored runs, model data
and worktree directories cannot break ordinary `pytest -q` collection. Tests
needing unavailable datasets/checkpoints report skips rather than invent data.

## Install a trusted artifact bundle

Obtain the release ZIP and its archive SHA-256 through a trusted release channel:

```powershell
python scripts/install_artifacts.py --archive path/to/aero-release.zip --sha256 ARCHIVE_SHA256 --destination .
```

An HTTPS release URL can replace the local ZIP path. Do not substitute a hash
copied from an untrusted download response. This standard-library installer
authenticates the complete archive before reading its manifest, then checks
every declared member's size and SHA-256. It accepts only the manifest and
`runs/` payload paths, rejects traversal, links and undeclared/duplicate members,
and imposes archive/payload limits. It does not import, unpickle or run weights.

All validation happens before destination files are changed. Existing files
are refused unless `--replace` is explicitly passed for a trusted update.
An I/O failure during the final copy can leave a partial installation; rerun
the installer deliberately and requalify the model rather than starting the UI
against incomplete files.

The archive contract is `format: aero-surrogate-release/1` in root
`release-manifest.json`, with `files: [{path, bytes, sha256, ...}]` describing
every other member under `runs/`. Publish the bundle checksum separately.

## Check the local application

The default check uses the synthetic fixture. To check installed real artifacts:

```powershell
.\.venv\Scripts\python.exe scripts/check_local.py --checkpoint runs/drivaernet-random/best.pt --field-checkpoint runs/fields/best.pt --out runs/local-wiring.json
```

Omit `--field-checkpoint` for a scalar-only release. The report path must be new.
The check starts two loopback-only processes on temporary ports, verifies model
identity, readiness, baseline evaluation, joint-extreme rejection, dynamic mesh,
web/API configuration, optional field inference and a five-step **scripted**
agent/history flow. It disables provider credentials and anonymous paid runs,
uses temporary logs and terminates its processes. It does not validate browser
rendering, current Token Factory availability or aerodynamic search quality.

The real scalar+field local check passed during release work using checkpoint
hashes `49e5a8b14d5b…` and `6a210c09b092…`. This is local HTTP wiring evidence,
not deployment readiness or field calibration.
