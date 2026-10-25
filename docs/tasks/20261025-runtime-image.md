# Independent CPU runtime image
Status: implementation and acceptance in progress.

## Scope
Build Runtime and Decision Lab from the migration worktree, including a freshly
compiled Rust engine and CPU neural dependencies. No existing venv or source
mount may supply runtime code. No live port, roster or database changes.

## Acceptance
Image imports engine, Runtime and Lab; independent unit suites pass. Start
short-lived loopback-only previews using immutable local image ID, read-only
filesystem and data mounts. Verify health, Lab UI, model identity and neural
analysis against an existing checkpoint without writing data. Stop only the
preview containers created for this task. Record image/source/dependency evidence.
Registry publication and production release remain separate gates.
