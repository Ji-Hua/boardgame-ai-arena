"""Decision Lab / Arena share the search, checkpoint and outcome perspective."""
import hashlib
import json

import pytest
import torch
from fastapi.testclient import TestClient

from agent_system.analysis import neural
from agent_system.analysis.app import app
from agent_system.analysis.state import Position, build_state
from agent_system.evaluation.arena.agents.mcts_scorer import MCTSScorer
from agent_system.evaluation.arena.matrix import play_block, standard_opening, summarize
from agent_system.training.dqn.checkpoint import save_checkpoint
from agent_system.training.dqn.model import CNNQNetwork
from backend.adapters.engine_adapter import _serialize_state


@pytest.fixture
def model(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    network = CNNQNetwork(cnn_channels=[2], dense_width=8, value_head=True)
    with torch.no_grad():
        network.value_head.bias.fill_(.3)
    path = tmp_path / 'model.pt'
    save_checkpoint(path, network, model_arch='cnn', algorithm='value_calibration',
        eval_summary={'output_semantics': 'policy_logits', 'value_head_status': 'calibrated'})
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = {'id': 'test_az', 'checkpoint': str(path), 'sha256': digest, 'metadata': {}}
    catalog = tmp_path / 'catalog.json'
    catalog.write_text(json.dumps({'models': [entry]}))
    monkeypatch.setenv('LAB_MODEL_CATALOG', str(catalog))
    neural.cached_runner.cache_clear()
    yield entry, network
    neural.cached_runner.cache_clear()


@pytest.mark.parametrize('seat', [1, 2])
def test_search_api_matches_arena_and_exposes_value_head(model, seat):
    entry, network = model
    position = Position(turn=seat)
    client = TestClient(app)
    base = {'position': position.model_dump(), 'model_id': entry['id']}
    analysis = client.post('/api/analyze', json={**base, 'algorithm': 'dqn'}).json()
    assert analysis['neural']['value'] == pytest.approx(torch.tanh(torch.tensor(.3)).item())
    assert any(l['name'] == 'value_head' for l in analysis['neural']['layers'])
    response = client.post('/api/neural/search', json={**base, 'simulations': 9})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['perspective'] == seat and sum(a['visits'] for a in data['actions']) == 9
    scorer = MCTSScorer(entry['checkpoint'], entry['sha256'], simulations=9)
    candidates = scorer.score(_serialize_state(build_state(position)))
    chosen = max(candidates, key=lambda v: v[1])[0]
    assert chosen == next(r['action'] for r in data['actions'] if r['action_index'] == data['selected_action_index'])
    tensor = client.post('/api/neural/tensor', json={**base, 'kind': 'activation', 'layer': 'value_head'})
    assert tensor.status_code == 200, tensor.text
    assert client.post('/api/neural/search', json={**base, 'simulations': 201}).status_code == 422
    assert hashlib.sha256(open(entry['checkpoint'], 'rb').read()).hexdigest() == entry['sha256']


def test_uncalibrated_checkpoint_rejected_and_arena_cost_counts(model, tmp_path):
    entry, network = model
    path = tmp_path / 'uncalibrated.pt'
    save_checkpoint(path, network, model_arch='cnn', algorithm='alphazero_lite')
    with pytest.raises(ValueError, match='calibration'):
        MCTSScorer(str(path), hashlib.sha256(path.read_bytes()).hexdigest())
    spec = {'id': entry['id'], 'algo': {'type': 'mcts', 'params': {
        'checkpoint': entry['checkpoint'], 'sha256': entry['sha256'], 'simulations': 4}},
        'policy': {'type': 'top_k', 'k': 1}, 'deterministic': True}
    greedy = {'id': 'greedy', 'algo': {'type': 'greedy', 'params': {}}, 'policy': {'type': 'top_k', 'k': 1}}
    opening = standard_opening()
    opening['id'] = 0
    block = play_block(spec, greedy, opening, max_plies=6)
    for game in block['games']:
        assert game['error'] is None
        assert game['decision_diagnostics']['test_az']['simulations'] == 4 * game['decisions']['test_az']
        assert len(game['decision_milliseconds']['test_az']) == game['decisions']['test_az']
    report = summarize([spec, greedy], [block], 1)
    assert report['status'] == 'complete'
    assert report['matrix']['test_az']['greedy']['decision_diagnostics']['simulations'] == 24
    json.dumps(report, allow_nan=False)
