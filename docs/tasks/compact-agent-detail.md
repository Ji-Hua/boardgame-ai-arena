# Compact Agent detail
Status: verified. Preview 8790 only.
Bounded searchable opponent list and adjacent match/training/config panels. Preserve opponent and panel in URL; default first recorded opponent. One training Run at a time, full comparison via existing workspace. Preserve original APIs, provenance and playback controls.
Acceptance: desktop/mobile, opponent filtering and selection, config, training/run navigation, replay links, persisted panel, existing UI checks. Backup gateway config and retain other container identities.

## Acceptance
Verified preview runtime source 20450f6 on feature/agent-ladder-preview; no GitHub push. Typecheck/build/diff checks and all 48 UI tests passed. Three browser scenarios passed (compact detail, replay transport, Agent-origin Play); final compact detail rerun passed after waiting for loaded matches before screenshots. Evidence: /home/jihua/quoridor-backups/compact-agent-final-20261030 and /home/jihua/quoridor-backups/compact-agent-20261030. Linux screenshot CJK font limitation remains as documented in dark-workbench.md.
Image: sha256:560d072bb5f7a1f9449cd0aa84e9bfa9884a1989e93c391d27c4764facb36fcd.
Rollback config: /home/jihua/quoridor-releases/agent-ladder-preview/ui-refinement-20261030T165355Z. Restore saved GATEWAY_IMAGE and run manage.sh up -d --no-deps gateway. Other running containers retained IDs/images/start times. Training 8741, MLflow 8742 and Dev 8765 returned HTTP 200. No production/backend/data changes.
