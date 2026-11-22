# TEST_CLASSIFICATION: SPECIFIED
# Spec: documents/agent/analysis/decision-analysis-interface.md
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from fastapi.testclient import TestClient

from agent_system.analysis import neural
from agent_system.analysis.app import app
from agent_system.analysis.state import ENGINE, Move, Position, Wall, build_state
from agent_system.runtime.inference.dqn import DQNInference
from agent_system.training.dqn.checkpoint import save_checkpoint
from agent_system.training.dqn.model import CNNQNetwork, QNetwork


@pytest.fixture(params=["cnn", "mlp", "cnn_policy", "cnn_ppo"])
def checkpoint(tmp_path, monkeypatch, request):
    arch = "cnn" if request.param in {"cnn_policy", "cnn_ppo"} else request.param
    with torch.random.fork_rng():
        torch.manual_seed(34)
        network = CNNQNetwork(cnn_channels=[2, 3]) if arch == "cnn" else QNetwork(hidden_layers=[16, 8])
    path = tmp_path / "model.pt"
    algorithm = {"cnn_policy": "policy_distillation", "cnn_ppo": "ppo"}.get(request.param, "dqn")
    save_checkpoint(path, network, model_arch=arch, algorithm=algorithm)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {"id": "test_model", "checkpoint": str(path), "sha256": digest, "metadata": {"model_config": {"model_arch": arch, "algorithm": algorithm}}}
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"models": [entry]}))
    monkeypatch.setenv("LAB_MODEL_CATALOG", str(catalog))
    neural.cached_runner.cache_clear()
    yield entry, network, catalog
    neural.cached_runner.cache_clear()


def wire_state(state):
    pos = neural.to_position(state)
    return {"current_player": pos["turn"], "pawns": {str(i+1): {"row": xy[0], "col": xy[1]} for i, xy in enumerate(pos["pawns"])},
            "walls_remaining": {str(i+1): n for i, n in enumerate(pos["walls_remaining"])},
            "wall_state": {key: getattr(state, key) for key in ("horizontal_edges", "vertical_edges", "horizontal_heads", "vertical_heads")}}


@pytest.mark.parametrize("position", [Position(), Position(turn=2), Position(pawns=((4, 3), (4, 4)),
                         walls_remaining=(9, 10), walls=[Wall(x=3, y=4, orientation="horizontal")])])
def test_diagnostics_match_runtime_q_mask_and_choice(checkpoint, position):
    entry, _, _ = checkpoint
    result = neural.analyze(position, entry["id"])
    serving = DQNInference(entry["checkpoint"], entry["sha256"])
    scores = serving.score(wire_state(build_state(position)))
    assert [(r["action"], r["value"]) for r in result["candidates"]] == scores
    assert result["selected_id"] == neural.action_id(max(scores, key=lambda item: item[1])[0])
    assert len(result["neural"]["outputs"]) == 209
    assert sum(row["legal"] for row in result["neural"]["outputs"]) == len(scores)
    assert all(row["masked_value"] == "-inf" for row in result["neural"]["outputs"] if not row["legal"])
    assert result["provenance"]["checkpoint_sha256"] == entry["sha256"]
    assert result["neural"]["model"]["output_semantics"] == ("policy_logits" if entry["metadata"]["model_config"]["algorithm"] in {"policy_distillation", "ppo"} else "action_values")
    # Strict JSON includes every diagnostic, even when masking uses -infinity.
    json.dumps(result, allow_nan=False)


def test_mask_and_ties_match_serving_even_when_illegal_output_is_largest(checkpoint):
    entry, network, catalog = checkpoint
    with torch.no_grad():
        for p in network.parameters():
            p.zero_()
        list(network.modules())[-1].bias[0] = 100
    save_checkpoint(entry["checkpoint"], network, model_arch=entry["metadata"]["model_config"]["model_arch"])
    entry["sha256"] = hashlib.sha256(open(entry["checkpoint"], "rb").read()).hexdigest()
    catalog.write_text(json.dumps({"models": [entry]}))
    result = neural.analyze(Position(), entry["id"])
    assert result["neural"]["raw_best"]["action_index"] == 0
    assert not result["neural"]["raw_best"]["legal"]
    assert result["neural"]["illegal_above_selected"] == 1
    assert result["selected_id"] == min(result["candidates"], key=lambda r: r["action_index"])["id"]
    assert result["neural"]["legal_margin"] == 0


def test_gradients_match_finite_differences_and_leave_network_unchanged(checkpoint):
    entry, _, _ = checkpoint
    position = Position(turn=2)
    result = neural.analyze(position, entry["id"])
    chosen = next(row for row in result["candidates"] if row["id"] == result["selected_id"])
    _, runner = neural.get_runner(entry["id"])
    before = {k: v.clone() for k, v in runner.network.state_dict().items()}
    detail = neural.branch(position, Move.model_validate(chosen["action"]), entry["id"])
    assert detail["value"] == chosen["value"]
    assert detail["after"]["position"]["turn"] == 1
    x = torch.tensor(runner.inference.encoder(build_state(position)), dtype=torch.float32)
    for sensitivity in detail["neural"]["sensitivities"]:
        gradient = torch.tensor(sensitivity["gradient"])
        index = gradient.abs().argmax().item()
        plus, minus = x.clone(), x.clone()
        plus.reshape(-1)[index] += .001
        minus.reshape(-1)[index] -= .001
        with torch.no_grad():
            delta = (runner.network(plus)-runner.network(minus))/.002
        finite_difference = delta[chosen["action_index"]]
        ref = sensitivity["reference_index"]
        if ref is not None:
            finite_difference -= delta[ref]
        assert gradient.reshape(-1)[index].item() == pytest.approx(finite_difference.item(), rel=.02, abs=2e-5)
    assert all(torch.equal(value, before[key]) for key, value in runner.network.state_dict().items())
    assert all(p.grad is None for p in runner.network.parameters())
    assert all(not module._forward_hooks for module in runner.modules.values())
    json.dumps(detail, allow_nan=False)


def test_input_axes_layers_and_weight_windows(checkpoint):
    entry, _, _ = checkpoint
    position = Position(turn=2)
    result = neural.analyze(position, entry["id"])
    views = result["neural"]["input"]["views"]
    assert views[0]["values"][4][8] == 1  # P2 is current player; no board flip.
    assert views[1]["values"][4][0] == 1
    _, runner = neural.get_runner(entry["id"])
    cnn = entry["metadata"]["model_config"]["model_arch"] == "cnn"
    if cnn:
        assert all(views[6]["values"][x][0] == 1 for x in range(9))
        layer, channel = "conv.0", 1
        expected_weight = runner.network.conv[0].weight[1, 2]
    else:
        assert len(views) == 6  # No invented goal-row feature for legacy MLP.
        layer, channel = "net.0", 1
        expected_weight = runner.network.net[0].weight[1, 256:]
    weights = neural.tensor(position, entry["id"], "weight", layer, channel, 2 if cnn else 0, 0 if cnn else 256)
    flattened = [n for row in weights["values"] for n in row]
    assert flattened == expected_weight.detach().reshape(-1).tolist()
    assert sum(weights["stats"]["histogram"]["counts"]) == weights["stats"]["count"]
    activation = neural.tensor(position, entry["id"], "activation", layer, channel if cnn else 0)
    assert activation["layout"] == ("board_xy" if cnn else "vector")
    x = torch.tensor(runner.inference.encoder(build_state(position)))
    with torch.no_grad():
        expected_activation = runner.network.conv[0](x.unsqueeze(0))[0, channel] if cnn else runner.network.net[0](x)
    assert [n for row in activation["values"] for n in row] == expected_activation.reshape(-1).tolist()
    with pytest.raises(ValueError, match="越界"):
        neural.tensor(position, entry["id"], "weight", layer, 99999)
    with pytest.raises(ValueError, match="输入通道"):
        neural.tensor(position, entry["id"], "activation", layer, 0, 1)


def test_catalog_identity_errors_terminal_and_http_contract(checkpoint):
    entry, _, catalog = checkpoint
    client = TestClient(app)
    assert client.get("/api/models").json()["models"][0]["id"] == entry["id"]
    body = {"position": Position().model_dump(mode="json"), "algorithm": "dqn", "model_id": entry["id"]}
    response = client.post("/api/analyze", json=body)
    assert response.status_code == 200
    result = response.json()
    action = next(r["action"] for r in result["candidates"] if r["id"] == result["selected_id"])
    assert client.post("/api/branch", json={**body, "action": action}).status_code == 200
    assert client.post("/api/neural/tensor", json={"position": body["position"], "model_id": entry["id"], "kind": "activation", "layer": result["neural"]["layers"][0]["name"]}).status_code == 200
    assert client.post("/api/analyze", json={**body, "model_id": "/tmp/model.pt"}).status_code == 422
    assert client.post("/api/analyze", json={**body, "model_id": None}).status_code == 422
    terminal = neural.analyze(Position(pawns=((4, 8), (4, 4))), entry["id"])
    assert terminal["selected_id"] is None and terminal["candidates"] == []
    assert not any(r["legal"] for r in terminal["neural"]["outputs"])
    entry["sha256"] = "0"*64
    catalog.write_text(json.dumps({"models": [entry]}))
    assert "SHA256" in client.post("/api/analyze", json=body).json()["detail"]


def test_missing_catalog_keeps_heuristics_available(monkeypatch, tmp_path):
    monkeypatch.setenv("LAB_MODEL_CATALOG", str(tmp_path/"missing.json"))
    client = TestClient(app)
    assert client.get("/api/models").json()["models"] == []
    assert client.post("/api/analyze", json={"position": {}}).status_code == 200


def test_parallel_observers_are_isolated_and_hooks_cleaned_on_failure(checkpoint):
    entry, _, _ = checkpoint
    _, runner = neural.get_runner(entry["id"])
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda p: neural.analyze(p, entry["id"]), [Position(), Position(turn=2)]))
    assert [r["neural"]["input"]["perspective"] for r in results] == [1, 2]
    assert all(len(r["neural"]["layers"]) == len(runner.modules) for r in results)
    def fail(*args):
        raise RuntimeError("test-only hook failure")
    handle = list(runner.modules.values())[-1].register_forward_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="hook failure"):
            runner.forward(build_state(Position()), capture=True)
    finally:
        handle.remove()
    assert all(not m._forward_hooks for m in runner.modules.values())


def test_nonfinite_checkpoint_weights_rejected(checkpoint):
    entry, network, catalog = checkpoint
    with torch.no_grad():
        next(network.parameters()).reshape(-1)[0] = float("nan")
    save_checkpoint(entry["checkpoint"], network, model_arch=entry["metadata"]["model_config"]["model_arch"])
    entry["sha256"] = hashlib.sha256(open(entry["checkpoint"], "rb").read()).hexdigest()
    catalog.write_text(json.dumps({"models": [entry]}))
    with pytest.raises(ValueError, match="非有限"):
        neural.analyze(Position(), entry["id"])
