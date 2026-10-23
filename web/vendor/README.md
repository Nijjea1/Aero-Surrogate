# Offline browser assets

The application serves its runtime graphics modules and fonts locally. Page
startup does not require jsDelivr or Google Fonts to be reachable.

- `three/`: three.js **0.160.0**, the dependency closure used by this application.
  Upstream MIT licence: `three/LICENSE`. `three/manifest.json` records each
  upstream URL, file size and SHA-256. Deliberate refresh command:
  `python scripts/vendor_web.py`.
- `fonts/`: IBM Plex Mono, Sans and Sans Condensed, served through local CSS.
  The adjacent OFL notices retain upstream SIL Open Font License 1.1 terms.
  `fonts/manifest.json` records the Google Fonts source URLs and file hashes.

The local server and Docker build include these files. They do not make API
or model inference available offline: live Nemotron proposals still require
the operator's Nebius Token Factory credentials and service connectivity.
