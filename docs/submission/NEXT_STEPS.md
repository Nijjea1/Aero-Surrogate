# Remaining work after local completion

All pre-deployment steps 1?9 are complete. Source is integrated on local `main`;
the serving weights and thresholds remain unchanged. See [COMPLETION.md](COMPLETION.md)
for measured results, current verification evidence and explicit environment limits.

1. **Synchronize and publish source from a network-enabled terminal.** This
   runner could not reach GitHub, so cached remote refs are not proof of freshness.

   ```powershell
   git switch main
   git fetch origin --prune
   git merge origin/main
   python -m pytest -q -rs
   git push origin main
   ```

   Resolve and test any new incoming changes before pushing. Do not force-push.
   If branch protection requires review, push a review branch and open a PR.
   Preserve the current source/evaluation/model identities; new runtime changes
   require relevant requalification and a rebuilt bundle.

2. **Transfer or publish the trusted serving bundle.** Use
   `runs/release/aero-surrogate-models.zip` and its adjacent `.zip.sha256`.
   Check [release-manifest.json](release-manifest.json) and
   [installation receipt](evidence/artifact-installation.json). The ZIP includes
   serving models, native references, frozen splits, complete metrics and the
   fifteen historical histories. The large raw evaluation bundle and category
   checkpoint are evaluation inputs, not requirements for serving.

   ```powershell
   python scripts/install_artifacts.py --archive PATH_TO_ZIP --sha256 TRUSTED_SHA256 --destination .
   ```

3. **Deploy the API and dynamic web server.** Follow the
   [API runbook](../../src/service/DEPLOY.md) and [web runbook](../../web/DEPLOY.md).
   Build/test the containers on the target host: Docker daemon access was denied
   here. Configure domain/HTTPS/CORS/proxy trust, one API worker, persistent
   histories/spend accounting, provider credentials and deliberate public budgets.
   A static-only web host cannot provide `/api/mesh`.

4. **Verify the actual hosted judge path.** Confirm `/health.synthetic=false`,
   the full scalar hash
   `49e5a8b14d5b21bc2fd3dedaeb2c997e8eb176da4bfb557b6d119deec473706a`,
   baseline Cd about 0.22456, slider changes, genuine joint-extreme rejection,
   Runs/logbooks and complete Results. Run a real Nemotron completion and
   cancellation/auth test with network access; verify persistence after restart
   and public budget limits. Repeat logged-out external Chromium/Firefox/mobile
   checks. Local Firefox passed twenty contexts; Chromium had seven passes and
   two preserved host resource failures, so hosted Chromium still needs checking.

5. **Record, upload and submit.** Use [video-script.md](video-script.md),
   [submission.md](submission.md) and [tooling-feedback.md](tooling-feedback.md).
   Record the verified deployed flow, produce the public video under three
   minutes, fill the real demo/repository/video/artifact URLs and submit Devpost.
   Confirm current deadline, eligibility, required fields and technology rules
   against the official portal before final submission. Submission copy must
   retain uncertainty/field/reference limitations and historical-run labels.

6. **Keep the submitted version available.** Assign ownership for hosting,
   monitoring and inference budget through the official judging period. Preserve
   submitted source/model/evidence identities and keep credentials private.

No further model training or new features are needed for this release.
