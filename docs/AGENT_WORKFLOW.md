# Agent development workflow
Read AGENTS.md, the accepted design, MIGRATION_STATUS.md and the relevant module contract before editing.
For a nontrivial change, write a task record using docs/tasks/TEMPLATE.md before implementation. Record scope, behavior, acceptance, tests and unresolved decisions. Keep status factual.

## Authority order
Explicit current user decisions -> accepted monorepo design -> current domain contracts -> operational records -> historical drafts.
If a user has already decided a conflict, document that decision and proceed. Otherwise isolate uncertain work and record it for the consolidated decision review.

## Source rules
Use named branches and focused commits. Never overwrite another worktree's dirty files. Do not copy whole source trees to synchronize hosts. Commit source/tests/config templates/docs only; generated outputs stay external.
No silent fallback to a different model. No duplicated rule implementations. No environment-specific branches.
Changes to public schemas must include compatibility expectations and consumer tests.

## Commands
python3 scripts/project.py check
python3 -m unittest discover -s tests/migration -v
python3 scripts/project.py release-env RELEASE.json --role dev
python3 scripts/project.py release-env RELEASE.json --role training
The last commands validate and print nonsecret variables; they do not deploy.
Existing domain test commands remain in component documentation; do not infer GPU support from CPU tests.

## Handoff
State branch and commit, exact checks and outcomes, current deployed versions if checked, external artifact paths, deferred questions and rollback impact. An implemented design and a deployed service are different statuses.
