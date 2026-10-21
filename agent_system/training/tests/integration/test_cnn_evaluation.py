"""Selection is locked before independent matches, including repeated seeds."""
import hashlib
import json
import sys
import torch
import pytest

from scripts import evaluate_cnn_repair as evaluation


def test_seed_models_are_locked_before_holdout_and_references_share_book(tmp_path, monkeypatch):
    def record(identifier, score):
        path = tmp_path / (identifier + '.pt')
        path.write_bytes(identifier.encode())
        return {'model_id': identifier, 'checkpoint': str(path),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'score': score, 'tactical_wins': 18, 'max_abs_legal_q': 1.}
    suite = tmp_path / 'suite'; suite.mkdir()
    baseline = suite / 'baseline'; baseline.mkdir()
    (baseline / 'diagnosis.json').write_text(json.dumps(record('original_2k', 0.)))
    development_book = evaluation.make_openings(1, 440)
    for seed, scores in ((42, (.4, .3)), (43, (.2, .35))):
        run = suite / str(seed); run.mkdir()
        (run / 'result.json').write_text('{}')
        (run / 'book.json').write_text(json.dumps(development_book))
        rows = [record(f'seed{seed}_update{i}', score) for i, score in enumerate(scores)]
        (run / 'diagnoses.json').write_text(json.dumps(rows))
    reference = tmp_path / 'reference.json'
    reference.write_text(json.dumps(record('previous_repair', .36)))
    from agent_system.evaluation.arena.agents.scorers import _build_state
    from agent_system.training.dqn.observation_cnn import encode_observation_cnn
    trained_opening = evaluation.make_openings(1, 2026102101)[0]
    trained_obs = torch.tensor([encode_observation_cnn(_build_state(trained_opening['state']))])
    dataset = tmp_path / 'demonstrations.pt'
    torch.save({'train': {'obs': trained_obs}, 'validation': {'obs': trained_obs}}, dataset)
    from agent_system.training.dqn.demonstrations import symmetry_key
    online_opening = evaluation.make_openings(1, 2026102102)[0]
    online_obs = torch.tensor(encode_observation_cnn(_build_state(online_opening['state'])))
    visited = tmp_path / 'training-state-keys.json'
    visited.write_text(json.dumps([symmetry_key(online_obs.numpy())]))
    seen = []

    def diagnose(path, identifier, book, output):
        selection = json.loads((suite / 'holdout/selection.json').read_text())
        assert selection['candidate_id'] == 'seed42_update0'
        assert {m['model_id'] for m in selection['replicate_selections']} == {'seed42_update0', 'seed43_update1'}
        assert {m['model_id'] for m in selection['method_selections']} == {'seed43_update1'}
        if not output.name.endswith('_standard'):
            assert book == selection['book']
            assert len({o['state_hash'] for o in book}) == 3
            assert development_book[0]['state_hash'] not in {o['state_hash'] for o in book}
            assert trained_opening['state_hash'] not in {o['state_hash'] for o in book}
            assert online_opening['state_hash'] not in {o['state_hash'] for o in book}
            assert selection['excluded_dataset_symmetry_groups'] == 1
            assert selection['excluded_symmetry_groups'] == 2
            seen.append(identifier)
        # A different seed wins holdout; that must not change the frozen primary.
        return {'score': .9 if identifier.startswith('seed43') else .1}

    monkeypatch.setattr(evaluation, 'diagnose_checkpoint', diagnose)
    monkeypatch.setattr(sys, 'argv', ['evaluate', str(suite), '--pairs', '3',
        '--replicate-prefix', 'seed42', '--replicate-prefix', 'seed43',
        '--comparison-prefix', 'seed43',
        '--reference-diagnosis', str(reference), '--exclude-dataset', str(dataset),
        '--exclude-state-keys', str(visited)])
    evaluation.main()
    assert sorted(seen) == ['original_2k', 'previous_repair', 'seed42_update0', 'seed43_update1']
    report = json.loads((suite / 'holdout/comparison.json').read_text())
    assert report['selection']['candidate_id'] == 'seed42_update0'
    assert len(report['replicate_comparisons']) == 2
    assert report['replicate_comparisons'][1]['score_improvement'] == .8
    assert {m['model_id'] for m in report['method_comparisons']} == {'seed42_update0', 'seed43_update1'}
    assert len(report['method_comparisons']) == 2


def test_fixed_selection_cannot_silently_pick_best_of_multiple_checkpoints(tmp_path, monkeypatch):
    run = tmp_path / 'run'; run.mkdir()
    (run / 'result.json').write_text('{}')
    (run / 'diagnoses.json').write_text(json.dumps([
        {'model_id': 'd1_ep00500'}, {'model_id': 'd1_ep02000'}]))
    monkeypatch.setattr(sys, 'argv', ['evaluate', str(tmp_path),
        '--candidate-prefix', 'd1', '--fixed-selection'])
    with pytest.raises(ValueError, match='exactly one checkpoint'):
        evaluation.main()
    assert not (tmp_path / 'holdout').exists()
