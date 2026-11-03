# Optional OpenSpiel research profile

This CPU backend supports the registered Kuhn Poker rules only. Rules,
tabular Q-learning, CFR, and exact exploitability come from OpenSpiel 2.0.2.
CFR is a solver reference, not an RL algorithm. Independent Q-learning is a
functional baseline, with no convergence or competitive-strength claim.

The original M3 RLlib suggestion is deferred; these small existing algorithms
exercise project/profile selection, jobs, metrics and policy export first.
Neural policies, recurrent memory and distributed rollout remain future work.

Build deploy/Dockerfile.research-openspiel and set JOBS_OPENSPIEL_IMAGE for a
controller using a separate data root. Without it the entrypoint is disabled.
Submission resolves the configured image to its local immutable Docker ID.
Existing JOBS_IMAGE and Quoridor jobs retain their behavior. This is not a
deployment instruction for the live controller.

Create a Kuhn Poker project, then submit train_openspiel with:
- algorithm: cfr or tabular_q_learning
- iterations: 1..100000 (CFR updates or Q-learning episodes)
- eval_every: positive integer; final policy is always evaluated
- seed: unsigned 32-bit integer

Q-learning uses upstream default step size 0.1, discount 1 and fixed epsilon
0.2. Each role gets its own learner and terminal reward. Chance sampling has a
seeded sampler; the upstream learner uses NumPy global RNG in the isolated
single-job worker. Evaluation preserves that RNG state. The exported Q policy
is greedy (ties shared), whereas CFR exports the average policy.

Artifacts in the job run directory:
- research-config.json: source, image, config hash, game/project, package versions
- research-metrics.jsonl: exact exploitability and NashConv versus progress
- policy.json: hash-verified information-state action probabilities for both roles
- research-artifacts.json: artifact byte SHA and explicit output semantics
- policy-smoke.json: exported-policy matches as each role against random

The worker reloads exported policy and exercises the common match runner.
Controller verifies artifact hash and provenance before marking success.
No JSON file is registered as a legacy .pt model or published to the ladder.
Successful training additionally copies policy bytes into the immutable
<data root>/policies/<sha>.json store and records project/job provenance in
policies/catalog.sqlite. This remains separate from legacy Torch model catalogs.
The project research-policy page lists identities and links to evaluation Jobs.
Policy download requires the existing platform token. Project scoping is a data
compatibility boundary, not a new per-user authorization model.
Historical completed jobs are not automatically backfilled by this change.
A neural-policy catalog and live multi-game Play UI remain future work.
No hidden-state replay is exposed through Jobs.

Reproduce container acceptance from a clean committed checkout:
PYTHONPATH=. python scripts/verify_openspiel_jobs.py --source <commit>
  --image <built image> --data-root <new external directory>
This creates two bounded training jobs, recreates their controller to verify
restart reconciliation, evaluates a registered policy against random and another
registered policy, and cancels a long-running training job.

## Registered-policy evaluation
evaluate_research_policy takes candidate_sha, optional opponent_sha (null means
random), pairs (1..1000), and seed. Both identities must be registered in the
selected project and compatible with its exact GameRef. Submission and worker
both verify policy bytes. No arbitrary path or external policy upload is accepted.

Protocol kuhn-policy-paired-v1:
- For each pair, swap candidate roles using the same environment seed.
- Agent seeds follow candidate/opponent identity when roles swap.
- Report both role returns and the mean of paired returns.
- Standard error uses independent pair means, not individual games; null for
  one pair. This is a sampling error estimate, not a competitive-strength claim.
- Candidate exploitability uses its complete two-role policy and OpenSpiel exact
  best responses. It is independent of the sampled opponent.
- Store project/game, input SHAs, seed/config hash, source/image and report digest.
- Keep full replay records in private-replays.jsonl; no public replay endpoint.
  The job API exposes the aggregate report only.

A completed evaluation does not create a Quoridor ladder season or promotion.


Upstream algorithm sources:
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/cfr.py
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/tabular_qlearner.py
https://github.com/google-deepmind/open_spiel/blob/master/open_spiel/python/algorithms/exploitability.py
