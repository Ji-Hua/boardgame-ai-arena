# Unified web implementation and preserved rollout
Status: in progress. User authorizes replacement of running components while preserving old code, configurations and data.

## Contract
One React application in apps/web, six workspaces, native React Lab without
iframe/external navigation, shared board source in packages/board-ui. Preserve
existing Play/Training/Arena/Replay/Models flows and immutable identity. Lab
supports presets, edited/imported positions, heuristic/neural analysis, branch
inspection, tensors, sensitivity and capability-gated search. Analysis never
writes to live gameplay. Live pause/resume remains a separate backend feature;
only explicitly labelled snapshot analysis may continue a running game.

## Acceptance and deployment
Run types, UI and domain tests, build immutable images, test real browser flows
and exact model inference. Capture private old container inspect/config/image
identities and service units before any switch. Preserve original repositories,
images, configs, catalogs, datasets and MLflow. Start candidates on isolated ports,
then change the user entry only after acceptance, retain rollback commands.
Public GitHub push remains blocked pending separate public-information consent;
private host-to-host Git synchronization may be used for authorized deployment.
