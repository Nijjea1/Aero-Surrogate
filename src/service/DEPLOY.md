# Deploying the Aero Surrogate API

This covers the API in `src/service` only. The web UI (`web/`) is a separate
static site; it calls this API at the URL given in its `?api=` parameter.

All configuration is environment variables (table at the end). Nothing secret
is baked into the image or committed: secrets go in a host file passed with
`--env-file`.

## 1. Run locally

From the repository root (see `README.md` in this folder for setup):

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.service.app:app --host 127.0.0.1 --port 8000
```

On startup the service prints one line describing its configuration, for example:

```
aero-surrogate ready: predictor=stub schema=2.0.0 cors=http://localhost:3000,http://localhost:5173 admin_token=unset public_runs=on rate_limit=120/min trust_proxy=off max_body=65536B log_dir=C:\...\runs\logs
```

It never prints secret values; `admin_token` only says `set` or `unset`.
Startup fails with a clear message if the run log directory cannot be created
or written, if a setting is invalid, or if `AERO_CHECKPOINT` cannot be served.

## 2. Docker

```bash
docker build -t aero-surrogate .                                  # CPU torch included (~1.5 GB)
docker build -t aero-surrogate:stub --build-arg WITH_TORCH=0 .    # stub only, no torch
```

The image is `python:3.13-slim`, runs as the non-root user `aero` (uid 10001),
contains only `src/` and the requirements files (`.dockerignore` keeps out
`.env`, `.venv`, `runs/`, `data/`, `web/`, checkpoints and `*.pt`), and sets:

| Inside the image | Value |
|---|---|
| `AERO_RUNLOG_DIR` | `/data/logs` |
| `AERO_SPEND_LEDGER` | `/data/spend.jsonl` |
| Volume | `/data` |
| Port | `8000` |
| Health check | `GET /ready` every 30 s (Python `urllib`; slim has no curl) |
| Command | `uvicorn src.service.app:app --host 0.0.0.0 --port 8000 --workers 1 --no-proxy-headers` |

Run it with a named volume and an env file that lives outside the repository:

```bash
docker volume create aero-data
docker run -d --name aero \
  -p 127.0.0.1:8000:8000 \
  -v aero-data:/data \
  --env-file /srv/aero/aero.env \
  --restart unless-stopped \
  aero-surrogate
docker logs aero              # the startup line, then access logs
curl -s localhost:8000/ready  # {"ready":true}
```

Example `/srv/aero/aero.env` (`chmod 600`, never committed). Docker passes
values verbatim, so do **not** quote them:

```
AERO_CORS_ORIGINS=https://aero.example.com
AERO_ADMIN_TOKEN=<output of: python3 -c "import secrets; print(secrets.token_urlsafe(32))">
AERO_TRUST_PROXY=1
NEBIUS_API_KEY=<key>
AERO_SPEND_CAP_USD=5.0
```

To use a host directory instead of a named volume, make it writable by uid
10001 first: `sudo mkdir -p /srv/aero/data && sudo chown 10001:10001 /srv/aero/data`,
then `-v /srv/aero/data:/data`.

## 3. A small Nebius AI Cloud CPU VM

The API needs no GPU: the stub is pure Python and the params-v1 model is a
small MLP (one forward pass per `/evaluate`). A small CPU VM with 2 vCPU,
4-8 GB RAM and 30 GB disk is enough (the image is about 1.5 GB).

1. **Create the VM** in the Nebius console: a CPU-only platform, an Ubuntu
   image, a public IP and your SSH key. In its security group allow inbound
   TCP 22 (SSH, ideally only from your IP), 80 and 443. Do **not** open 8000.
2. **Install Docker** on the VM:
   ```bash
   curl -fsSL https://get.docker.com | sudo sh
   sudo usermod -aG docker $USER   # log out and back in
   ```
3. **Build and run** (clone the repo on the VM, or `docker save | ssh ... docker load`
   an image built elsewhere):
   ```bash
   git clone <repo-url> aero && cd aero
   docker build -t aero-surrogate .
   sudo mkdir -p /srv/aero && sudo nano /srv/aero/aero.env && sudo chmod 600 /srv/aero/aero.env
   docker volume create aero-data
   docker run -d --name aero --restart unless-stopped \
     -p 127.0.0.1:8000:8000 -v aero-data:/data --env-file /srv/aero/aero.env \
     aero-surrogate
   ```
   Publishing on `127.0.0.1` keeps port 8000 off the internet (Docker's
   published ports bypass `ufw`, so binding to loopback is the reliable way).
4. **Put HTTPS in front** with [Caddy](https://caddyserver.com/docs/install)
   on the host, which obtains and renews the certificate itself. Point a DNS
   name (say `api.aero.example.com`) at the VM's public IP, then
   `/etc/caddy/Caddyfile`:
   ```
   api.aero.example.com {
       reverse_proxy 127.0.0.1:8000
   }
   ```
   and `sudo systemctl reload caddy`.
5. **Settings behind the proxy:**
   - `AERO_CORS_ORIGINS` = the exact origin the frontend is served from
     (scheme + host + port, no path, no trailing slash), e.g.
     `https://aero.example.com`. `*` is refused.
   - `AERO_TRUST_PROXY=1`. Without it every request appears to come from the
     Docker gateway and all users share one rate-limit bucket. With it the
     service uses the **last** `X-Forwarded-For` entry, which Caddy sets to
     the real client address. Only set it when the API is reachable solely
     through the proxy (step 3 binds to loopback for exactly this reason);
     otherwise a client could pick its own IP.
6. **Point the frontend at it:** `https://<frontend>/?api=https://api.aero.example.com`.
   An HTTPS page cannot call a plain-HTTP API, which is another reason for
   step 4.

## 4. Serving a real checkpoint

Without `AERO_CHECKPOINT` the service serves the synthetic stub (`/health`
reports `"synthetic": true`). To serve a trained params-v1 checkpoint, use the
default (torch) image and mount the file read-only:

```bash
docker run -d --name aero --restart unless-stopped \
  -p 127.0.0.1:8000:8000 -v aero-data:/data \
  -v /srv/aero/checkpoints:/checkpoints:ro \
  -e AERO_CHECKPOINT=/checkpoints/best.pt \
  --env-file /srv/aero/aero.env aero-surrogate
```

A checkpoint that cannot be served fails startup; there is never a fallback
to the stub. `torch.load` runs code pickled in the file, so only mount
checkpoints a teammate produced with `src/train/train.py`. `/health` then
reports `model_version` and `checkpoint_sha256`. See "Serving a real
checkpoint" in `README.md` for what the loader accepts.

## 5. Operations

**`/health` vs `/ready`.** `/health` describes the service (predictor,
synthetic flag, schema, checkpoint hash) and answers 200 whenever the process
is up. `/ready` answers 200 only when the predictor is loaded and the run log
directory is writable, else 503 with `{"ready": false, "problems": [...]}`.
Point health checks and load balancers at `/ready`. Plain Docker only marks an
unhealthy container (`docker ps`); `--restart unless-stopped` restarts it when
the process exits, not when it is unhealthy.

**Where things live.** Everything persistent is under `/data` (the volume):

| Path | Contents |
|---|---|
| `/data/logs/<run_id>.jsonl` | scored attempts, served by `/history/{run_id}` |
| `/data/logs/events/<run_id>.jsonl` | run start/finish, refused proposals, provider errors, token usage |
| `/data/spend.jsonl` (+ `.lock` while writing) | the Token Factory spend ledger the cap is enforced against |
| `docker logs aero` | startup line and access log (stdout) |

Back up the volume with
`docker run --rm -v aero-data:/data -v "$PWD":/backup python:3.13-slim tar czf /backup/aero-data.tgz /data`.
**Losing the volume resets the spend ledger, and with it the cap's running total.**

**Rotating `AERO_ADMIN_TOKEN`.** Generate a new one (at least 16 characters;
`python3 -c "import secrets; print(secrets.token_urlsafe(32))"`), edit the env
file, then recreate the container: `docker rm -f aero` and the same
`docker run` as before. `docker restart` does **not** re-read `--env-file`.
The old token stops working as soon as the new container starts; agent runs in
progress are stopped by the restart. Unset the variable to disable admin
access entirely.

**Exactly one worker.** Run logs use process-local locks and exclusive run-ID
reservation, and the rate limiter and concurrent-run limit are in-memory
counters. Several workers (or several containers on one volume) would each
keep their own limits and could interleave writes to the same logs. The image
therefore runs `--workers 1`; scale up the VM rather than out. (The spend
ledger alone is safe across processes: it uses a lock file.)

**Cost safety.** Only agent runs cost money (Nemotron calls through Token
Factory); `/evaluate` and `/compare` never call the provider. Layers, from
narrowest to widest:
- per-run budget: `AERO_PUBLIC_RUN_BUDGET_USD` / `AERO_ADMIN_RUN_BUDGET_USD`;
- iteration ceilings: `AERO_PUBLIC_MAX_ITERATIONS` / `AERO_ADMIN_MAX_ITERATIONS`;
- `AERO_PUBLIC_DAILY_BUDGET_USD`: total public spend per UTC day;
- `AERO_MAX_CONCURRENT_RUNS`;
- `AERO_SPEND_CAP_USD`: a hard, lifetime cap on everything recorded in the
  ledger. Every call reserves its worst case first and is refused if the
  total would pass the cap, across restarts as long as `/data` persists.
Set `AERO_PUBLIC_RUNS=0` to stop anonymous runs without redeploying anything
else. The run and budget settings take effect through the agent-run (`/runs`)
endpoints; without those, the service spends nothing.

**What is public and what needs the token.**

| Endpoint | Access |
|---|---|
| `GET /health`, `/ready`, `/schema`, `/history/{run_id}` | public, not rate limited |
| `POST /evaluate`, `/compare` | public, rate limited per client IP, body-size limited |
| `GET /runs`, `GET /runs/{run_id}` | public, rate limited. They show every run's settings, cost, stop reason and error text, admin runs included; nothing secret is stored in them |
| `POST /runs` (agent runs, spend money) | rate limited; without an `Authorization` header the caller is *public* (allowed only when `AERO_PUBLIC_RUNS=1`, with public limits); `Authorization: Bearer <AERO_ADMIN_TOKEN>` gets admin limits; a wrong token is always refused, never downgraded to public |
| `POST /runs/{run_id}/cancel` | admin: any run this process owns; public: only a public run, with the `X-Cancel-Token` returned when it started |

Rate-limited requests get `429` with `Retry-After` (seconds); oversized bodies
get `413`. Both carry CORS headers so the browser can show the message.

## 6. Environment variables

Defaults are for local development. The service settings (`AERO_*` below,
except the spend ledger ones) are validated at startup; invalid values fail
startup rather than being guessed. `NEBIUS_*`, `NEMOTRON_MODEL`,
`AERO_SPEND_CAP_USD` and `AERO_SPEND_LEDGER` are read when an agent run
starts, so a bad value there shows up as a 503 from `POST /runs`, not at
startup. Check them with `python -m src.agent.provider --smoke` (two tiny
billed calls) after deploying.

| Variable | Default | Used by | Meaning |
|---|---|---|---|
| `AERO_CORS_ORIGINS` | `http://localhost:3000,http://localhost:5173` | service | comma-separated origins allowed to call the API; explicit `http(s)://` origins only, `*` refused |
| `AERO_ADMIN_TOKEN` | unset (no admin) | service | bearer token that lifts public limits; at least 16 characters |
| `AERO_PUBLIC_RUNS` | `1` | service | `1`/`0`: may callers without the token start agent runs |
| `AERO_MAX_CONCURRENT_RUNS` | `1` | service | agent runs executing at once (>= 1) |
| `AERO_PUBLIC_MAX_ITERATIONS` | `25` | service | evaluation ceiling for public runs |
| `AERO_ADMIN_MAX_ITERATIONS` | `50` | service | evaluation ceiling for admin runs |
| `AERO_PUBLIC_RUN_BUDGET_USD` | `0.10` | service | spend ceiling per public run |
| `AERO_ADMIN_RUN_BUDGET_USD` | `0.50` | service | spend ceiling per admin run |
| `AERO_PUBLIC_DAILY_BUDGET_USD` | `1.00` | service | total public spend per UTC day |
| `AERO_RATE_LIMIT_PER_MINUTE` | `120` | service | `POST /evaluate`, `/compare` and every `GET`/`POST` under `/runs`, per client IP per minute; `0` = off |
| `AERO_MAX_BODY_BYTES` | `65536` | service | largest accepted request body (>= 1024); larger gets 413 |
| `AERO_RUNLOG_DIR` | `runs/logs` (image: `/data/logs`) | service, agent | run logs; must be writable; same directory for service and agent |
| `AERO_TRUST_PROXY` | `0` | service | `1` = take the client IP from the last `X-Forwarded-For` entry; only behind your own reverse proxy |
| `AERO_CHECKPOINT` | empty (stub) | service | params-v1 checkpoint to serve; failure to load fails startup |
| `AERO_SPEND_LEDGER` | `<repo>/runs/spend.jsonl` (image: `/data/spend.jsonl`) | agent | spend ledger (JSONL) the cap is enforced against |
| `AERO_SPEND_CAP_USD` | `5.0` | agent | hard cap on total spend recorded in the ledger |
| `NEBIUS_API_KEY` | unset | agent | Token Factory key; needed only for live agent runs |
| `NEBIUS_BASE_URL` | `https://api.tokenfactory.nebius.com/v1` | agent | Token Factory endpoint |
| `NEMOTRON_MODEL` | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B` | agent | default model when a run names none |
| `AERO_ENV_FILE` | `<repo>/.env` | agent | `.env` file the agent loads (real environment wins); in Docker use `--env-file` instead |
| `AERO_SERVICE_URL` | `http://127.0.0.1:8000` | agent CLI | where the command-line designer finds the service |
