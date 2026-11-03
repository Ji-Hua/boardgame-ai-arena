"""Versioned research identity, independent of legacy AgentSpec hashes."""
import re
from agent_system.games.contracts import stable_hash
from agent_system.games.registry import create_game,validate_ref

def policy_spec(bundle,sha,role):
    ref=validate_ref(bundle["game_ref"])
    if role not in create_game(ref.wire()).roles:raise ValueError("Unsupported role")
    if not re.fullmatch(r"[0-9a-f]{64}",sha):raise ValueError("Invalid policy identity")
    if bundle.get("format")!="mrjack_mlp_tanh_v1":raise ValueError("Unsupported research policy")
    value=dict(schema_version=2,namespace="research-agent-v2",game_ref=ref.wire(),
        role=role,policy_sha256=sha,runtime="numeric_mlp_v1",
        encoder=bundle.get("encoder","mrjack_visible_summary_v1"),
        memory_mode=bundle.get("memory_mode","visible_summary"),
        output_semantics=bundle["output_semantics"],selection="sample_legal_probabilities")
    return dict(value,hash=stable_hash(value))
