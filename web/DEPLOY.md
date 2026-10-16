# Deploying the web UI

The site is two containers on one VM behind one Caddy:

| | Image | Serves | Guide |
|---|---|---|---|
| `aero.example.com` | `aero-web` (`web/Dockerfile`) | the pages and `POST /api/mesh`, the 3D car | this file |
| `api.aero.example.com` | `aero-surrogate` (`Dockerfile`) | predictions, fields, agent runs | [src/service/DEPLOY.md](../src/service/DEPLOY.md) |

A plain static host is not enough: the designer and the race build every car
through `/api/mesh`, which needs the geometry module. That is why the web side
runs `web/serve.py` in its own small image (Python, numpy, trimesh, scipy;
about 470 MB, no torch).

## 1. Build and run

On the same VM as the API (after its steps 1–3), from the repository root:

```bash
docker build -f web/Dockerfile -t aero-web .
docker run -d --name aero-web --restart unless-stopped \
  -p 127.0.0.1:5173:5173 \
  -e AERO_WEB_API=https://api.aero.example.com \
  -e AERO_WEB_MESH_PER_MINUTE=60 \
  -e AERO_WEB_TRUST_PROXY=1 \
  aero-web
curl -s localhost:5173/healthz     # {"ok": true}
curl -s localhost:5173/config.js   # window.AERO_CONFIG = {"api": "https://api.aero.example.com"};
```

The build uses `web/Dockerfile.dockerignore`, so only `src/contracts.py`,
`src/geometry/` and `web/` go into the image: no data, checkpoints or secrets.
Publishing on `127.0.0.1` keeps the port off the internet; Caddy is the only
way in.

| Setting | Default | Meaning |
|---|---|---|
| `AERO_WEB_API` | unset: the pages call `http://127.0.0.1:8000` | the API origin the pages call, served as `/config.js` so a shared link needs no `?api=` |
| `AERO_WEB_MESH_PER_MINUTE` | `0` (no limit) | 3D models per visitor per minute; past it, `429` with `Retry-After` |
| `AERO_WEB_MESH_WORKERS` | `2` | models built at once; a request that waits 10 s for a free one gets `503` |
| `AERO_WEB_TRUST_PROXY` | off | take the visitor from the last `X-Forwarded-For` entry. Only behind your own proxy, as here; otherwise a client could pick its own address |
| `PORT` | `5173` | listening port inside the container |

`?api=<url>` on any page still overrides the configured API, for testing a
second service.

## 2. HTTPS for both

Point both DNS names at the VM, then one `/etc/caddy/Caddyfile`:

```
aero.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:5173
}

api.aero.example.com {
    encode zstd gzip
    reverse_proxy 127.0.0.1:8000
}
```

`sudo systemctl reload caddy`. `encode` compresses the API's `/fields`
responses (about 770 KB of JSON each) in transit with no change to the service;
the mesh endpoint already gzips its own replies, and Caddy leaves those alone.

## 3. Join them up

- On the API, `AERO_CORS_ORIGINS=https://aero.example.com` (exact origin, no
  trailing slash) and `AERO_TRUST_PROXY=1`.
- On the web container, `AERO_WEB_API=https://api.aero.example.com`.
- Open `https://aero.example.com`. The status in the top bar should name the
  model and schema; "Service unreachable" means the API address or CORS origin
  does not match.

## Updating

`docker build` again and `docker rm -f aero-web` then the `docker run` above.
The pages revalidate on every load, so visitors see the new version straight away.
Publishing new results is a file change: rebuild after updating
`web/results/metrics.json` or `web/results/benchmark.json`.
