# Unified research image
Status: implementation and acceptance in progress.
The old Dashboard image omits platform_entry and platform_play. Build the React
SPA in a Node stage and serve it with the existing unified Python application.
Reuse the accepted CPU image as the engine/dependency base; no source mounts.
Keep the Lab proxy allowlist: additionally accept only the fixed Compose DNS
endpoint http://lab:8744; arbitrary hosts, credentials and ports stay rejected.
Acceptance: frontend build/tests, backend tests, loopback-only container preview,
SPA navigation, health, model listing and real Lab proxy. No MLflow writes,
production switch, public preview or gameplay promotion in this task.
