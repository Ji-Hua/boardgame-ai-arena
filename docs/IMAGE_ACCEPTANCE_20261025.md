# Independent image acceptance, 2026-10-25
Status: verified local candidates on Training; not a production release.

## Image identity
| Component | Local immutable image ID | Source revision |
|---|---|---|
| Runtime / Lab | `sha256:5746cf19c0c0eb126759cfe24a77d60953bdf6d4d64fa3c06409db5055a4ee7f` | `fae4fa19837634f68a592c2a36cf121c4e6bb4e2` |
| Unified Research | `sha256:ae9acf2c462a78e26c4a253c178d87fe6a0970083e75719cbc2a50ece82354f2` | `b412b7b09d5b6935c221d778cbcb00bd18c92a42` |
Research uses the accepted Runtime image as its dependency/engine base. No agent,
engine or Lab source changed between these revisions. Its Node build preserves
existing shared board imports from frontend/src. Application directory relocation
is deferred to a behavior-preserving change.
Images currently exist only in Training's Docker store; local RepoDigests do not
mean they have been uploaded to a registry. Do not use them as a cross-host release.

## Evidence
Evidence root: `/home/jihua/quoridor-backups/migration-evidence-20261025`.
Build logs, image inspect JSON, resolved Python dependency list, test logs and
acceptance JSON are external. Python 3.12 base resolved to
`sha256:a0f57064cbdcd6285fa45f51237ab0b35ea8ebaed06a1f5973e89e19dfb86704`.
Rust wheel was freshly compiled using the committed Cargo.lock; no host editable
engine or host venv was mounted. CPU torch is 2.11.0+cpu.
Direct Python dependencies are pinned; transitive versions are recorded in the
image. Base tags/apt/transitive resolution are not yet a hermetic rebuild lock.

## Observed checks
- Runtime image: 173 pytest passed, 1 optional external-model test skipped.
  Test-only pytest/pytest-asyncio were installed in a disposable container.
- Research image: 124 unittest tests run, 1 skipped, successful, without network.
- React: 24 tests, TypeScript and Vite production build passed inside image build.
- Runtime preview listed 13 built-in agent types (not the production external roster).
- Real Lab model cnn_random_2k SHA `bad4bfeb424967cbfa17712cf8058c0f91816c1c369d50569d48c5dbee8080e8`
  matched before and after inference; 131 legal candidates, a legal selection.
  Response semantics remain action_values; no policy-logit relabeling.
- Unified platform indexed 92 models. This is static discovery, not proof of 92
  successful model loads. SPA HTML routes and actual Lab proxy analysis passed.
- Previews used loopback random ports, non-root images, read-only filesystem/data,
  bounded CPU/memory and no source mounts. Created containers/networks were removed.

## Limits / next gates
No interactive browser or full human-game acceptance was performed here.
Play navigation HTML is not a verified playable session. The Gameplay image,
research capability keys/catalog, cross-host gateway and runtime session linkage
still need a separate isolated acceptance. GPU training images, MLflow database
migration, durable games and registry publication are not delivered.
Before live rollout: choose distribution/registry, record per-component provenance
and model roster, verify isolated cross-host games, capture current rollback
configuration, and obtain the user's concrete service-switch approval.
