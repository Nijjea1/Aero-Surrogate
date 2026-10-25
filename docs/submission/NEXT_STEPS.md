# Updated plan after branch consolidation

Updated 27 October 2026. The working branch is **`finish/predeploy-complete`**. This supersedes
the branch-location statement in the earlier local-completion report; the
evaluation evidence and historical experiment identities remain unchanged.

## Integration status

- All branch tips already available in this checkout are ancestors of local
  `main`, including the three agent branches and the cached teammate branches.
- The one remaining teammate change, phone-layout commit `516c924`, was merged
  normally: whole-car framing, bounded labels, initially hidden phone labels and
  truncated service status.
- Agent branch commits were first verified patch-equivalent to the integration
  work, then their original history was recorded without overwriting later fixes.
- Existing feature/fix commits and original author/committer dates were retained.
  No backdating, fabricated activity, squashing or force-push was performed.
- A backup branch preserves the pre-consolidation state:
  `backup/pre-merge-20261027-5aa669f`.
- **Remote freshness and publication are unverified.** Fetch, pull and push all
  failed because this runner could not connect to `github.com:443`. Cached
  `origin/*` refs must not be described as the latest server state.

## Earlier post-merge checks (dated snapshot)

- Full suite: **1,021 passed, four expected skips, three warnings**, 99.61 s.
- Desktop Chromium completed the real-model walkthrough.
- Six responsive checks passed: Designer, Runs and Results at both **390 px**
  and **412 px**, with no document-level horizontal overflow. The real 3D car
  rendered, Cd matched the qualified baseline, and phone labels started hidden.
- Firefox launch/context setup crashed before page navigation, including an
  isolated retry. This is retained as a failed environment-level check, not a
  passing post-merge Firefox result. The previous release's twenty successful
  Chromium/Firefox contexts remain dated pre-merge evidence.
- No new provider calls or deployment occurred during branch consolidation.

## Remaining work, in order

1. **Synchronize with GitHub from a network-enabled terminal.** Local merging is
   complete, but fresh server changes may still exist. Preserve history:

   ```powershell
   git switch finish/predeploy-complete
   git fetch origin --prune
   git merge origin/main
   python -m pytest -q -rs
   git push -u origin finish/predeploy-complete
   ```

   Resolve any newly discovered conflicts before testing/pushing. Do not force
   push. If branch protection requires review, push a review branch and open a
   PR targeting `main` instead. Existing commits are already separated by their
   actual feature/fix work; rewriting their dates is unnecessary.

2. **Finish final local publication and qualification.** The incoming bundle is
   received and verified: both checkpoint hashes match, all thirty field arrays
   load and frozen splits reproduce. Category and all-stored-cell field grades
   are complete in staging. Publish their fresh sanitized reports, the stronger
   pooled-neighbour field reference and fresh scaling results when complete.
   Regenerate README/graphics, run integrated tests/model/HTTP/browser checks,
   rebuild the serving ZIP and verify installation/checksums. Keep the category
   model evaluation-only; do not replace the serving model or threshold.

3. **Deploy the API and dynamic web server.** Mount the verified model bundle,
   persist logs/spend, configure HTTPS/CORS/proxy trust and keep one API worker.
   Use the refreshed ZIP whose source revision, current report checksums and
   installation test agree. The large raw evaluation bundle stays separate.

4. **Verify the actual hosted judge path.** Check the full scalar hash and real
   baseline, parameter updates, genuine domain rejection, recorded runs and final
   results. Run real Nemotron completion and cancellation with network access.
   Repeat external logged-out browser checks, including Firefox/mobile. Local
   socket/daemon restrictions prevented provider/container certification here.

5. **Finish the video and submission.** Use [video-script.md](video-script.md),
   [submission.md](submission.md) and [tooling-feedback.md](tooling-feedback.md).
   Record working deployed footage, keep the public YouTube video under three
   minutes, fill the real URLs and submit Devpost. Target October 29; verify the
   official event deadline and judging availability before submission rather
   than relying on the planning document's dates.

6. **Keep the judged release available.** Assign hosting/monitoring/inference
   budget through the official judging period. Preserve the submitted source and
   evidence identities; avoid untracked model or gate changes.

The model architecture, parameter contract and gate remain frozen. No additional
datasets, frameworks, services or unvalidated physics claims are release blockers.
