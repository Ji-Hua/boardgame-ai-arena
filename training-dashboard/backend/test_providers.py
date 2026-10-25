"""SPECIFIED: provider contracts/evidence security in the unified dashboard design.

Tests use stdlib fixtures/mocks; originals remain read-only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend import mlflow
from backend.app import _find_run_dir, create_app
from backend.providers import CompositeAdapter, LegacyAdapter, LocalEvidence, MLflowAdapter, UnifiedRunService, metric_metadata


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def registration(run_id="registered", source=None, sha=hashlib.sha256(b"").hexdigest(), **extra):
    return dict(run_id=run_id, run_name="same name", status="FINISHED", start_time=1000, end_time=2000,
                params={"source_path": str(source)} if source else {},
                tags={"algorithm": "policy_distillation", "checkpoint_sha256": sha}, metrics={}, **extra)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = patch.dict(os.environ, {"QUORIDOR_DATA_ROOT": str(self.root),
                                          "TRAINING_DASHBOARD_ARTIFACT_ROOTS": str(self.root / "legacy")})
        self.env.start()
        self.source = self.root / "experiments" / "comparison" / "matrix" / "C" / "seed-42"
        self.checkpoint = self.source / "checkpoints" / "update012000.pt"
        self.checkpoint.parent.mkdir(parents=True)
        self.checkpoint.touch()
        self.sha = hashlib.sha256(b"").hexdigest()
        write(self.source / "experiment.json", {"training": {"algorithm": "policy_distillation"}})
        (self.source / "distillation-metrics.jsonl").write_text(
            '\n'.join(json.dumps({"update": u, "policy_cross_entropy": 1 / u, "training_batch_accuracy": .5}) for u in (200, 400)))
        write(self.source / "heldout-d2.json", {"strict_action_agreement": .4, "cross_entropy": 2.1})
        write(self.root / "experiments/comparison/architecture-catalog.json", {"models": [{"source_path": str(self.checkpoint),
            "sha256": self.sha, "historical_training_commit": "immutable-commit", "metadata": {"model_config": {"algorithm": "policy_distillation", "model_arch": "cnn"}}}]})
        self.rows = [registration(source=self.checkpoint)]
        self.exp = patch.object(mlflow, "experiments", return_value=[{"id": "9", "name": "Architecture"}])
        self.runs = patch.object(mlflow, "runs", side_effect=lambda _: self.rows)
        self.exp_mock, self.runs_mock = self.exp.start(), self.runs.start()
        self.addCleanup(self.temp.cleanup)
        self.addCleanup(self.env.stop)
        self.addCleanup(self.exp.stop)
        self.addCleanup(self.runs.stop)

    def arena(self, wrong_sha=False, missing_actions=False):
        root = self.root / "experiments/comparison/arena-v2"
        manifest = {"protocol": "paired-v1", "roster": {"agents": [{"id": "candidate", "algo": {"params": {
            "checkpoint": str(self.checkpoint), "sha256": "b" * 64 if wrong_sha else self.sha}}}]},
            "book": [{"id": 0, "state_hash": "opening-hash", "actions": [], "state": {}}]}
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write(root / "manifest.json", manifest)
        write(root / "report.json", {"run_id": "arena-v2", "manifest_sha256": manifest["manifest_sha256"], "matrix": {"candidate": {
            "minimax_d2": {"games": 4, "wins": 1, "draws": 2, "losses": 1, "score": .5}, "unplayed": {"games": 0}}},
            "standard_controls": [{"a": "candidate", "b": "minimax_d2", "winners": [None, "minimax_d2"]}]})
        game = {"p1": "candidate", "p2": "minimax_d2", "winner": None, "steps": 1,
                "actions": [] if missing_actions else [{"player": 1, "type": "pawn", "target": [4, 1]}]}
        block = {"block_id": "block", "a": "candidate", "b": "minimax_d2", "opening_id": 0,
                 "opening_hash": "opening-hash", "manifest_sha256": manifest["manifest_sha256"], "control": False, "games": [game]}
        (root / "blocks").mkdir()
        with gzip.open(root / "blocks/block.json.gz", "wt") as handle:
            json.dump(block, handle)
        return root, manifest, block

    def test_local_curve_has_recorded_axis_only_and_heldout_snapshot(self):
        p = MLflowAdapter()
        metrics = p.metrics("registered")
        self.assertEqual(len(metrics["points"]), 2)
        self.assertEqual(metrics["points"][0]["update"], 200)
        self.assertIsNone(metrics["points"][0]["timestamp"])
        self.assertIsNone(metrics["points"][0]["episode"])
        self.assertIsNone(metrics["points"][0]["step"])
        self.assertEqual(metrics["metadata"]["policy_cross_entropy"]["semantic"], "policy_cross_entropy")
        self.assertEqual(len(metrics["snapshots"]), 1)
        self.assertIn("No MLflow metric history", metrics["provenance"]["reason"])
        detail = p.detail("registered")
        self.assertEqual(detail["checkpoints"][0]["step"], 12000)
        self.assertIsNone(detail["checkpoints"][0]["episode"])
        self.assertEqual(detail["source"], "mlflow")

    def test_empty_history_does_not_invent_curve(self):
        (self.source / "distillation-metrics.jsonl").unlink()
        metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(metrics["points"], [])
        self.assertEqual(metrics["axes"], [])
        self.assertEqual(len(metrics["snapshots"]), 1)

    def test_mlflow_history_timestamps_and_unknown_semantics(self):
        self.rows[0]["metrics"] = {"unfamiliar": 3.0}
        with patch.object(mlflow, "metric_history", return_value=[{"step": 7, "timestamp": 123, "value": 3.0}]):
            metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(metrics["points"][0]["timestamp"], 123)
        self.assertEqual(metrics["points"][0]["step"], 7)
        self.assertIsNone(metrics["points"][0]["update"])
        self.assertEqual(metrics["metadata"]["unfamiliar"]["semantic"], "policy_distillation:unfamiliar")
        self.assertEqual(metric_metadata("loss", "DoubleDQN", "env_steps")["semantic"], "td_loss")

    def test_duplicate_registrations_and_evaluations_not_trainings(self):
        self.rows += [registration("second", self.checkpoint),
                      {**registration("evaluation", self.checkpoint), "tags": {"checkpoint_sha256": self.sha, "benchmark_run_id": "arena-v2"}},
                      {"run_id": "container", "tags": {"protocol": "paired"}, "params": {}, "metrics": {}}]
        p = MLflowAdapter()
        runs, _ = p.listing()
        self.assertEqual(len(runs), 1)
        self.assertIn("evaluation", runs[0]["provenance"]["related_evaluation_runs"])
        self.assertEqual(p.resolve("second")[0], "registered")
        with self.assertRaises(KeyError):
            p.resolve("evaluation")

    def test_same_names_are_not_identity(self):
        self.rows.append(registration("different", self.root / "other/checkpoints/update2.pt", sha="b" * 64))
        self.assertEqual(len(MLflowAdapter().listing()[0]), 2)

    def test_parent_training_and_checkpoint_child_coalesce(self):
        parent = {**registration("parent"), "params": {}, "tags": {}, "metrics": {"loss": 1}}
        self.rows = [parent, {**registration(source=self.checkpoint), "tags": {"mlflow.parentRunId": "parent", "checkpoint_sha256": self.sha}}]
        p = MLflowAdapter()
        self.assertEqual(len(p.listing()[0]), 1)
        self.assertEqual(p.resolve("registered")[0], "parent")

    def test_listing_cached_with_failures_retained(self):
        self.exp_mock.side_effect = RuntimeError("offline")
        p = MLflowAdapter()
        self.assertIn("offline", p.listing()[1][0])
        self.assertIn("offline", p.listing()[1][0])
        self.assertEqual(self.exp_mock.call_count, 1)

    def test_successful_listing_cached(self):
        p = MLflowAdapter()
        self.assertEqual(p.listing(), p.listing())
        self.assertEqual(self.exp_mock.call_count, 1)
        self.assertEqual(self.runs_mock.call_count, 1)

    def test_evidence_only_resolves_catalog_sources_for_matching_sha(self):
        write(self.root / "unrelated-catalog.json", {"models": [
            {"sha256": f"{index:064x}", "source_path": str(self.root / "models" / f"{index}.pt")}
            for index in range(40)
        ]})
        provider = MLflowAdapter()
        provider._refresh()
        with patch.object(provider, "_catalog_source", wraps=provider._catalog_source) as source:
            evidence = provider._evidence("registered")
        self.assertEqual(evidence["checkpoints"][0]["sha256"], self.sha)
        self.assertTrue(source.called)
        self.assertTrue(all(call.args[0]["sha256"] == self.sha for call in source.call_args_list))

    def test_result_json_curve_is_not_heldout_curve(self):
        (self.source / "distillation-metrics.jsonl").unlink()
        write(self.source / "result.json", {"metrics": [{"update": 12000, "policy_cross_entropy": .4}]})
        m = MLflowAdapter().metrics("registered")
        self.assertEqual(len(m["points"]), 1)
        self.assertEqual(m["points"][0]["update"], 12000)
        self.assertIsNone(m["points"][0]["timestamp"])
        self.assertEqual(len(m["snapshots"]), 1)

    def test_child_checkpoint_metrics_are_snapshots(self):
        self.rows = [registration("parent", self.checkpoint),
                     {**registration("child", self.checkpoint), "tags": {"mlflow.parentRunId": "parent", "checkpoint_sha256": self.sha},
                      "metrics": {"score": .2}}]
        with patch.object(mlflow, "metric_history") as history:
            metrics = MLflowAdapter().metrics("parent")
        history.assert_not_called()
        self.assertEqual(len(metrics["points"]), 2)
        self.assertNotIn("score", metrics["columns"])
        self.assertTrue(any(s["scope"] == "checkpoint" for s in metrics["snapshots"]))

    def test_secondary_training_owner_history_independent_of_canonical_id(self):
        self.rows = [registration("a-registration", self.checkpoint),
                     {**registration("z-trainer", self.source), "tags": {"algorithm": "DoubleDQN", "step_type": "env_steps"},
                      "metrics": {"loss": .2}}]
        with patch.object(mlflow, "metric_history", return_value=[
                {"step": 10, "timestamp": 100, "value": .4},
                {"step": 20, "timestamp": 200, "value": .2}]) as history:
            p = MLflowAdapter()
            self.assertEqual(p.resolve("z-trainer")[0], "a-registration")
            metrics = p.metrics("a-registration")
        history.assert_called_once_with("z-trainer", "loss")
        self.assertEqual(len(metrics["points"]), 2)
        self.assertEqual(metrics["metadata"]["loss"]["step_type"], "env_steps")
        self.assertEqual(metrics["metadata"]["loss"]["semantic"], "td_loss")
        self.assertEqual(metrics["metadata"]["loss"]["origin"]["run_id"], "z-trainer")
        self.assertFalse(any(s.get("run_id") == "z-trainer" for s in metrics["snapshots"]))

    def test_explicit_training_child_and_conflicting_owners_remain_separate(self):
        self.rows = [registration("parent", self.source),
                     {**registration("trainer", self.checkpoint), "tags": {
                         "mlflow.parentRunId": "parent", "run_type": "training", "step_type": "env_steps", "algorithm": "DoubleDQN"},
                      "metrics": {"loss": .5}},
                     {**registration("other", self.source), "tags": {"step_type": "gradient_updates", "algorithm": "ppo"},
                      "metrics": {"loss": .1}}]
        with patch.object(mlflow, "metric_history", side_effect=lambda run, key: [
                {"step": 1, "timestamp": 2, "value": .5 if run == "trainer" else .1}]) as history:
            metrics = MLflowAdapter().metrics("parent")
        self.assertEqual(history.call_count, 2)
        self.assertEqual(set(metrics["columns"]), {"loss::mlflow:trainer", "loss::mlflow:other"})
        self.assertEqual(metrics["metadata"]["loss::mlflow:trainer"]["step_type"], "env_steps")
        self.assertEqual(metrics["metadata"]["loss::mlflow:other"]["step_type"], "gradient_updates")
        self.assertNotEqual(metrics["metadata"]["loss::mlflow:trainer"]["semantic"],
                            metrics["metadata"]["loss::mlflow:other"]["semantic"])
        self.assertTrue(any("Multiple metric owners" in w for w in metrics["warnings"]))

    def test_evaluation_descendants_classified_before_grouping(self):
        self.rows += [{**registration("eval", self.checkpoint), "tags": {"run_type": "evaluation"}},
                      {**registration("eval-child", self.checkpoint), "tags": {"mlflow.parentRunId": "eval"}, "metrics": {"score": .5}},
                      {**registration("eval-grandchild", self.checkpoint), "tags": {"mlflow.parentRunId": "eval-child"}, "metrics": {"score": .6}},
                      {**registration("explicit-trainer", self.root / "training"), "tags": {
                          "mlflow.parentRunId": "eval", "run_type": "training"}, "metrics": {"loss": .2}}]
        p = MLflowAdapter()
        self.assertEqual(len(p.listing()[0]), 2)
        for run in ("eval", "eval-child", "eval-grandchild"):
            with self.assertRaises(KeyError):
                p.resolve(run)
        self.assertEqual([r["run_id"] for r in p.resolve("registered")[1]], ["registered"])
        p.resolve("explicit-trainer")

    def test_sha_only_catalog_recovers_exact_source_without_name_guessing(self):
        self.rows = [registration()]
        p = MLflowAdapter()
        self.assertEqual(p._evidence("registered")["source"], self.source)
        self.assertEqual(len(p.metrics("registered")["points"]), 2)
        write(self.root / "additional-catalog.json", {"models": [{"sha256": self.sha,
              "training_experiment": str(self.root / "different-training")} ]})
        p = MLflowAdapter()
        self.assertIsNone(p._evidence("registered")["source"])
        self.assertEqual(p.metrics("registered")["points"], [])
        self.assertTrue(any("Ambiguous catalog source" in w for w in p.metrics("registered")["warnings"]))

    def test_catalog_training_experiment_preferred_and_contained(self):
        self.rows = [registration()]
        for path in self.root.rglob("*catalog*.json"):
            path.unlink()
        write(self.root / "catalog.json", {"models": [{"sha256": self.sha,
              "source_path": str(self.root / "models" / f"{self.sha}.pt"),
              "training_experiment": str(self.source)}]})
        self.assertEqual(len(MLflowAdapter().metrics("registered")["points"]), 2)
        write(self.root / "catalog.json", {"models": [{"sha256": self.sha, "training_experiment": "/etc"}]})
        self.assertIsNone(MLflowAdapter()._evidence("registered")["source"])

    def test_local_checkpoint_sha_conflict_blocks_fallback_with_warning(self):
        self.checkpoint.write_bytes(b"changed weights")
        metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(metrics["points"], [])
        self.assertTrue(any("SHA conflict" in w for w in metrics["warnings"]))
        self.assertTrue(any("fallback blocked" in w for w in metrics["warnings"]))

    def test_catalog_tag_sha_conflict_blocks_fallback_even_without_file(self):
        self.checkpoint.unlink()
        self.rows[0]["tags"]["checkpoint_sha256"] = "b" * 64
        metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(metrics["points"], [])
        self.assertTrue(any("SHA conflict" in w for w in metrics["warnings"]))

    def test_failed_history_metadata_does_not_override_local_curve_semantics(self):
        (self.source / "distillation-metrics.jsonl").write_text(json.dumps({"step": 15, "loss": .2}))
        self.rows[0]["metrics"] = {"loss": 1}
        self.rows[0]["tags"].update(step_type="gradient_updates", algorithm="DoubleDQN")
        with patch.object(mlflow, "metric_history", side_effect=RuntimeError("unavailable")):
            metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(metrics["metadata"]["loss"]["step_type"], "env_steps")
        self.assertEqual(metrics["metadata"]["loss"]["origin"]["source"], "local_training_files")

    def test_catalog_sha_source_disagreement_blocks_explicit_fallback(self):
        write(self.root / "catalog.json", {"models": [{"sha256": self.sha,
              "training_experiment": str(self.root / "other-training")} ]})
        metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(metrics["points"], [])
        self.assertTrue(any("Source provenance conflict" in w for w in metrics["warnings"]))

    def test_metric_failure_warning_survives_source_fallback(self):
        self.rows[0]["metrics"] = {"loss": 1}
        with patch.object(mlflow, "metric_history", side_effect=RuntimeError("history offline")):
            metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(len(metrics["points"]), 2)
        self.assertTrue(any("history offline" in w for w in metrics["warnings"]))

    def test_benchmarks_win_score_controls_and_sha_association(self):
        self.arena()
        b = MLflowAdapter().benchmarks("registered")["benchmarks"]
        self.assertEqual(len(b), 2)
        ranked = next(x for x in b if x["protocol"] == "arena_paired")
        self.assertEqual(ranked["win_rate"], .25)
        self.assertEqual(ranked["score_rate"], .5)
        self.assertEqual(ranked["checkpoint_sha"], self.sha)
        self.assertEqual(ranked["game_type"], "evaluation")
        self.assertTrue(ranked["source_manifest"]["sha256"])

    def test_wrong_sha_does_not_associate_arena(self):
        self.arena(wrong_sha=True)
        self.assertEqual(MLflowAdapter().benchmarks("registered")["count"], 0)

    def test_tampered_manifest_is_not_associated(self):
        root, manifest, _ = self.arena()
        manifest["protocol"] = "tampered"
        write(root / "manifest.json", manifest)
        p = MLflowAdapter()
        self.assertEqual(p.benchmarks("registered")["count"], 0)
        self.assertTrue(any("hash invalid" in w for w in p._evidence("registered")["warnings"]))

    def test_mirrored_evaluation_deduplicated_by_manifest(self):
        root, manifest, _ = self.arena()
        mirror = self.root / "arena/mirror"
        write(mirror / "manifest.json", manifest)
        write(mirror / "report.json", json.loads((root / "report.json").read_text()))
        self.assertEqual(len(MLflowAdapter()._arenas("registered")), 1)

    def test_replay_index_and_lazy_converter_use_original_block(self):
        _, manifest, block = self.arena()
        p = MLflowAdapter()
        row = p.replays("registered")["replays"][0]
        self.assertTrue(row["replay_available"])
        self.assertEqual(row["game_id"], "block:0")
        self.assertIsNone(row["episode"])
        self.assertEqual(row["step"], 12000)
        converter = types.ModuleType("backend.replay")
        from unittest.mock import Mock
        converter.arena_replay = Mock(return_value={"states": []})
        with patch.dict(sys.modules, {"backend.replay": converter}):
            self.assertEqual(p.replay("registered", row["replay_path"]), {"states": []})
        args = converter.arena_replay.call_args.args
        self.assertEqual(args[:5], (manifest, block, 0, "candidate", "registered"))
        with self.assertRaises(KeyError):
            p.replay("registered", "../../experiment.json")

    def test_missing_trajectory_is_visible_not_replayed(self):
        self.arena(missing_actions=True)
        p = MLflowAdapter()
        row = p.replays("registered")["replays"][0]
        self.assertFalse(row["replay_available"])
        with self.assertRaises(ValueError):
            p.replay("registered", row["replay_path"])

    def test_same_training_checkpoint_match_replays_have_distinct_perspectives(self):
        root, manifest, block = self.arena()
        second = self.source / "checkpoints/update024000.pt"
        second.write_bytes(b"second checkpoint")
        second_sha = hashlib.sha256(second.read_bytes()).hexdigest()
        self.rows.append(registration("second", second, sha=second_sha))
        manifest.pop("manifest_sha256")
        manifest["roster"]["agents"].append({"id": "candidate-two", "algo": {"params": {"sha256": second_sha}}})
        manifest["manifest_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        write(root / "manifest.json", manifest)
        write(root / "report.json", {"run_id": "arena-v2", "manifest_sha256": manifest["manifest_sha256"]})
        block.update(b="candidate-two", manifest_sha256=manifest["manifest_sha256"])
        block["games"][0].update(p2="candidate-two", winner="candidate")
        with gzip.open(root / "blocks/block.json.gz", "wt") as handle:
            json.dump(block, handle)
        p = MLflowAdapter()
        rows = p.replays("registered")["replays"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(len({r["replay_path"] for r in rows}), 2)
        self.assertEqual({r["checkpoint_sha"] for r in rows}, {self.sha, second_sha})
        converter = types.ModuleType("backend.replay")
        from unittest.mock import Mock
        converter.arena_replay = Mock(side_effect=lambda *args: {"agent": args[3], "sha": args[5]["sha256"]})
        with patch.dict(sys.modules, {"backend.replay": converter}):
            actual = [p.replay("registered", r["replay_path"]) for r in rows]
        self.assertEqual({(r["agent"], r["sha"]) for r in actual},
                         {("candidate", self.sha), ("candidate-two", second_sha)})

    def test_block_manifest_and_opening_guards(self):
        _, manifest, block = self.arena()
        block["opening_hash"] = "changed"
        self.assertIn("Opening", MLflowAdapter._trajectory_reason(manifest, block, block["games"][0]))
        block["manifest_sha256"] = "changed"
        self.assertIn("manifest", MLflowAdapter._trajectory_reason(manifest, block, block["games"][0]))

    def test_safe_historical_mapping_and_symlink_uri_rejection(self):
        local = LocalEvidence()
        self.assertEqual(local.path("/home/jihua/quoridor-data/experiments/a"), self.root / "experiments/a")
        for value in ("https://host/data", "file:///etc/passwd", "/etc/passwd", str(self.root / "../escape")):
            with self.assertRaises(ValueError):
                local.path(value)
        (self.root / "escape").symlink_to("/etc")
        with self.assertRaises(ValueError):
            local.path(self.root / "escape/passwd")

    def legacy(self):
        run = self.root / "legacy/old"
        write(run / "metrics.jsonl", {"episode": 1, "step": 9, "loss": .5, "timestamp": 99})
        (run / "checkpoints").mkdir()
        write(run / "eval_replays/game.json", self.legacy_trajectory())
        (run / "eval_replays.jsonl").write_text(json.dumps({"replay_path": "eval_replays/game.json"}) + "\n")
        return LegacyAdapter([str(run.parent)])

    @staticmethod
    def legacy_trajectory():
        state = {"p1": [4, 0], "p2": [4, 8], "h_walls": [], "v_walls": [],
                 "walls_remaining_p1": 10, "walls_remaining_p2": 10}
        return {"game_length": 1, "states": [state, {**state, "p1": [4, 1]}],
                "actions": [{"player_id": "P1", "action_type": "pawn", "x": 4, "y": 1}]}

    def legacy_evaluation(self, p, checkpoints=("ep00500_step154339.pt",), replay_checkpoint="ep00500_step154339.pt"):
        run = p.require("old")
        for checkpoint in checkpoints:
            (run / "checkpoints" / checkpoint).touch()
        write(run / "eval_results.jsonl", {"episode": 500, "step": 154339,
              "opponent_name": "minimax_d1", "num_games": 2, "wins": 1})
        write(run / "eval_replays.jsonl", {"episode": 500, "checkpoint": replay_checkpoint,
              "opponent_name": "minimax_d1", "game_length": 1, "replay_path": "eval_replays/game.json"})

    def test_legacy_exact_checkpoint_groups_benchmark_and_replay(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        replay = p.replays("old")["replays"][0]
        benchmark = p.benchmarks("old")["benchmarks"][0]
        self.assertEqual((replay["episode"], replay["step"], replay["progress_type"]), (500, 154339, "env_steps"))
        self.assertEqual(benchmark["checkpoint"], replay["checkpoint"])
        self.assertTrue(benchmark["replay_available"])

    def test_legacy_unique_replay_checkpoint_without_local_weights(self):
        p = self.legacy()
        self.legacy_evaluation(p, checkpoints=())
        self.assertEqual(p.benchmarks("old")["benchmarks"][0]["checkpoint"], "ep00500_step154339.pt")
        run = p.require("old")
        entry = json.loads((run / "eval_replays.jsonl").read_text())
        entry["episode"] = None
        write(run / "eval_replays.jsonl", entry)
        self.assertEqual(p.replays("old")["replays"][0]["episode"], 500)

    def test_legacy_missing_ambiguous_and_conflicting_identity_not_joined(self):
        for checkpoints, name, episode, step in (
                ((), "arbitrary-ep00500_step154339.pt", 500, 154339),
                (("ep00500_step154339.pt", "ep500_step154339.pt"), "ep00500_step154339.pt", 500, 154339),
                ((), "ep00500_step154340.pt", 500, 154339),
                ((), "ep00500_step154339.pt", 501, 154339),
                ((), "ep00500_step154339.pt", 500, None)):
            with self.subTest(checkpoints=checkpoints, name=name, episode=episode, step=step):
                p = self.legacy()
                self.legacy_evaluation(p, checkpoints, name)
                run = p.require("old")
                entry = json.loads((run / "eval_replays.jsonl").read_text())
                entry["episode"] = episode
                write(run / "eval_replays.jsonl", entry)
                write(run / "eval_results.jsonl", {"episode": 500, "step": step, "opponent_name": "minimax_d1"})
                benchmark = p.benchmarks("old")["benchmarks"][0]
                self.assertIsNone(benchmark["checkpoint"])
                self.assertFalse(benchmark["replay_available"])
                # Reuse the fixture root without leaving checkpoint evidence.
                for checkpoint in (run / "checkpoints").iterdir():
                    checkpoint.unlink()
                (run / "checkpoints").rmdir()

    def test_legacy_stale_index_display_metadata_uses_complete_payload(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        run = p.require("old")
        index_path = run / "eval_replays.jsonl"
        replay_path = run / "eval_replays/game.json"
        index = json.loads(index_path.read_text())
        index.update(game_index=0, game_length=236, winner="P2",
                     result_from_dqn_perspective="loss", dqn_player_id="P1")
        write(index_path, index)
        payload = {**self.legacy_trajectory(), "run_id": "old", "episode": 500,
                   "checkpoint": index["checkpoint"], "game_index": 0,
                   "eval_opponent": {"name": "minimax_d1"}, "winner": "P1",
                   "result_from_dqn_perspective": "loss", "dqn_player_id": "P2"}
        # Cover both length-only staleness and changed outcome/seat metadata.
        for length, result in ((246, "loss"), (194, "win")):
            with self.subTest(length=length):
                payload.update(game_length=length, result_from_dqn_perspective=result,
                               actions=self.legacy_trajectory()["actions"] * length,
                               states=[self.legacy_trajectory()["states"][0]] * (length + 1))
                write(replay_path, payload)
                originals = (index_path.read_bytes(), replay_path.read_bytes())
                response = p.replays("old")
                row = response["replays"][0]
                self.assertTrue(row["replay_available"])
                for key in ("game_length", "winner", "result_from_dqn_perspective", "dqn_player_id"):
                    self.assertEqual(row[key], payload[key])
                self.assertTrue(any("Stale replay index metadata" in w and "game_length" in w
                                    for w in response["warnings"]))
                detail = p.replay("old", row["replay_path"])
                self.assertEqual(detail["game_length"], length)
                self.assertEqual(detail["warnings"], response["warnings"])
                self.assertTrue(p.benchmarks("old")["benchmarks"][0]["replay_available"])
                self.assertEqual(originals, (index_path.read_bytes(), replay_path.read_bytes()))

    def test_legacy_config_authorized_dqn_writer_identity_reconciles_metadata(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        run = p.require("old")
        write(run / "config.yaml", {"agent_id": "dqn_old", "algorithm": "double_dqn"})
        payload = {**self.legacy_trajectory(), "run_id": "dqn_old", "agent_id": "dqn_old",
                   "checkpoint": "ep00500_step154339.pt", "episode": 500, "game_length": 2}
        payload["actions"] *= 2
        payload["states"].append(payload["states"][-1])
        write(run / "eval_replays/game.json", payload)
        originals = {path: path.read_bytes() for path in
                     (run / "config.yaml", run / "eval_replays.jsonl", run / "eval_replays/game.json")}
        for identity in ("old", "dqn_old"):
            with self.subTest(identity=identity):
                payload["run_id"] = identity
                write(run / "eval_replays/game.json", payload)
                row = p.replays("old")["replays"][0]
                self.assertTrue(row["replay_available"])
                self.assertEqual(row["game_length"], 2)
                detail = p.replay("old", row["replay_path"])
                self.assertEqual(detail["run_id"], identity)
                self.assertTrue(any("game_length" in warning for warning in detail["warnings"]))
                self.assertTrue(p.benchmarks("old")["benchmarks"][0]["replay_available"])
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})

    def test_legacy_dqn_writer_identity_requires_precise_config_and_agent(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        run = p.require("old")
        valid_config = {"agent_id": "dqn_old", "algorithm": "double_dqn"}
        valid = {**self.legacy_trajectory(), "run_id": "dqn_old", "agent_id": "dqn_old"}
        for config, changes in (
                (None, {}), ({}, {}), ([], {}), ("invalid: [", {}),
                ({"algorithm": "double_dqn"}, {}),
                ({"agent_id": "dqn_old"}, {}),
                ({**valid_config, "algorithm": "ppo"}, {}),
                ({**valid_config, "agent_id": "dqn_other"}, {}),
                (valid_config, {"agent_id": None}),
                (valid_config, {"agent_id": "dqn_other"}),
                (valid_config, {"run_id": "dqn_dqn_old"}),
                (valid_config, {"run_id": "other_old"}),
                ({**valid_config, "agent_id": "other_old"},
                 {"run_id": "other_old", "agent_id": "other_old"})):
            with self.subTest(config=config, changes=changes):
                if config is None:
                    (run / "config.yaml").unlink(missing_ok=True)
                elif isinstance(config, str):
                    (run / "config.yaml").write_text(config)
                else:
                    write(run / "config.yaml", config)
                write(run / "eval_replays/game.json", {**valid, **changes})
                row = p.replays("old")["replays"][0]
                self.assertFalse(row["replay_available"])
                self.assertEqual(row["replay_unavailable_reason"], "Replay identity mismatch: run_id")
                with self.assertRaisesRegex(ValueError, "identity mismatch: run_id"):
                    p.replay("old", row["replay_path"])

    def test_legacy_authorized_dqn_identity_still_rejects_overwritten_checkpoint(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        run = p.require("old")
        write(run / "config.yaml", {"agent_id": "dqn_old", "algorithm": "double_dqn"})
        for checkpoint in ("ep00500_step189992.pt", "ep500_step154339.pt"):
            with self.subTest(checkpoint=checkpoint):
                write(run / "eval_replays/game.json", {**self.legacy_trajectory(),
                      "run_id": "dqn_old", "agent_id": "dqn_old", "episode": 500,
                      "checkpoint": checkpoint, "game_length": 1})
                row = p.replays("old")["replays"][0]
                self.assertFalse(row["replay_available"])
                self.assertEqual(row["checkpoint"], "ep00500_step154339.pt")
                self.assertEqual(row["replay_unavailable_reason"], "Replay identity mismatch: checkpoint")
                self.assertFalse(p.benchmarks("old")["benchmarks"][0]["replay_available"])
                with self.assertRaisesRegex(ValueError, "identity mismatch: checkpoint"):
                    p.replay("old", row["replay_path"])

    def test_legacy_dqn_reused_path_does_not_reassociate_stale_index_row(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        run = p.require("old")
        write(run / "config.yaml", {"agent_id": "dqn_old", "algorithm": "double_dqn"})
        stale = json.loads((run / "eval_replays.jsonl").read_text())
        current = {**stale, "checkpoint": "ep00500_step189992.pt"}
        (run / "eval_replays.jsonl").write_text(
            "\n".join(json.dumps(row) for row in (stale, current)) + "\n")
        write(run / "eval_replays/game.json", {**self.legacy_trajectory(),
              "run_id": "dqn_old", "agent_id": "dqn_old", "episode": 500,
              "checkpoint": current["checkpoint"]})
        rows = p.replays("old")["replays"]
        self.assertFalse(rows[0]["replay_available"])
        self.assertEqual(rows[0]["checkpoint"], stale["checkpoint"])
        self.assertEqual(rows[0]["replay_unavailable_reason"], "Replay identity mismatch: checkpoint")
        self.assertTrue(rows[1]["replay_available"])
        self.assertEqual(p.replay("old", current["replay_path"])["checkpoint"], current["checkpoint"])
        self.assertFalse(p.benchmarks("old")["benchmarks"][0]["replay_available"])

    def test_legacy_stale_index_cannot_mask_incomplete_payload(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        run = p.require("old")
        # Index and arrays say one ply, but the payload declares two: incomplete.
        write(run / "eval_replays/game.json", {**self.legacy_trajectory(), "game_length": 2})
        response = p.replays("old")
        row = response["replays"][0]
        self.assertFalse(row["replay_available"])
        self.assertIn("Incomplete replay trajectory", row["replay_unavailable_reason"])
        self.assertEqual(row["game_length"], 1)
        with self.assertRaisesRegex(ValueError, "Incomplete replay trajectory"):
            p.replay("old", row["replay_path"])

    def test_legacy_payload_identity_conflicts_rejected_without_reassociation(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        run = p.require("old")
        index = json.loads((run / "eval_replays.jsonl").read_text())
        index.update(game_index=0, opponent_type="minimax", opponent_depth=1)
        write(run / "eval_replays.jsonl", index)
        valid = {**self.legacy_trajectory(), "run_id": "old", "episode": 500,
                 "checkpoint": index["checkpoint"], "game_index": 0,
                 "eval_opponent": {"name": "minimax_d1", "type": "minimax", "depth": 1}}
        for changes in ({"run_id": "another-run"}, {"checkpoint": "ep00500_step189992.pt"},
                        {"episode": 501}, {"game_index": 100},
                        {"eval_opponent": {"name": "pawn_random"}},
                        {"eval_opponent": {"type": "random"}},
                        {"eval_opponent": {"depth": 2}}):
            with self.subTest(changes=changes):
                write(run / "eval_replays/game.json", {**valid, **changes})
                row = p.replays("old")["replays"][0]
                self.assertFalse(row["replay_available"])
                self.assertIn("identity mismatch", row["replay_unavailable_reason"])
                self.assertEqual(row["checkpoint"], index["checkpoint"])
                self.assertFalse(p.benchmarks("old")["benchmarks"][0]["replay_available"])
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    p.replay("old", row["replay_path"])

    def test_legacy_invalid_trajectories_are_explicitly_unavailable(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        path = p.require("old") / "eval_replays/game.json"
        valid = self.legacy_trajectory()
        for payload in (None, "{bad json", [], {"states": []},
                        {**valid, "actions": []}, {**valid, "states": valid["states"][:1]},
                        {**valid, "game_length": 2}, {**valid, "states": [{}, {}]},
                        {**valid, "states": [{**valid["states"][0], "p1": None}, valid["states"][1]]},
                        {**valid, "states": [{**valid["states"][0], "h_walls": None}, valid["states"][1]]},
                        {**valid, "actions": [{}]}):
            with self.subTest(payload=payload):
                if payload is None:
                    path.unlink(missing_ok=True)
                elif isinstance(payload, str):
                    path.write_text(payload)
                else:
                    write(path, payload)
                row = p.replays("old")["replays"][0]
                self.assertFalse(row["replay_available"])
                self.assertTrue(row["replay_unavailable_reason"])
                self.assertFalse(p.benchmarks("old")["benchmarks"][0]["replay_available"])
                with self.assertRaises(ValueError):
                    p.replay("old", row["replay_path"])

    def test_legacy_recorded_current_player_preserved_without_guessing(self):
        p = self.legacy()
        trajectory = self.legacy_trajectory()
        trajectory["states"][0]["current_player"] = "P2"
        trajectory["states"][1]["current_player"] = 1
        write(p.require("old") / "eval_replays/game.json", trajectory)
        states = p.replay("old", "eval_replays/game.json")["states"]
        self.assertEqual([s["current_player"] for s in states], ["P2", "P1"])

    def test_legacy_optimizer_axis_and_explicit_reward_semantics(self):
        p = self.legacy()
        run = p.require("old")
        write(run / "metrics.jsonl", {"episode": 1, "step": 9, "opt_steps": 4, "avg_reward": .2})
        (run / "train.log").write_text("algorithm=DoubleDQN reward_mode=terminal\n")
        metrics = p.metrics("old")
        self.assertEqual(metrics["points"][0]["gradient_updates"], 4)
        self.assertIn("gradient_updates", [axis["key"] for axis in metrics["axes"]])
        self.assertEqual(metrics["metadata"]["avg_reward"]["reward_mode"], "terminal")
        semantic = metrics["metadata"]["avg_reward"]["semantic"]
        (run / "train.log").write_text("algorithm=DoubleDQN reward_mode=distance_delta\n")
        self.assertNotEqual(semantic, p.metrics("old")["metadata"]["avg_reward"]["semantic"])
        (run / "train.log").write_text("algorithm=DoubleDQN\n")
        unknown = p.metrics("old")["metadata"]["avg_reward"]
        self.assertIsNone(unknown["reward_mode"])
        self.assertIn("doubledqn:old", unknown["semantic"])
        # A suggestive directory name is not explicit reward provenance.
        run.rename(run.parent / "terminal_other")
        other = p.metrics("terminal_other")["metadata"]["avg_reward"]
        self.assertIsNone(other["reward_mode"])
        self.assertNotEqual(unknown["semantic"], other["semantic"])

    def test_cross_entropy_curve_scope_and_heldout_override(self):
        (self.source / "distillation-metrics.jsonl").write_text(json.dumps({"update": 1, "cross_entropy": .2}))
        metrics = MLflowAdapter().metrics("registered")
        self.assertEqual(metrics["metadata"]["cross_entropy"]["scope"], "training_batch")
        self.assertEqual(metrics["snapshots"][0]["metadata"]["cross_entropy"]["scope"], "heldout")

    def test_legacy_list_does_not_read_checkpoint_contents(self):
        p = self.legacy()
        self.legacy_evaluation(p)
        original = Path.open
        def guard(path, *args, **kwargs):
            self.assertNotEqual(path.suffix, ".pt", "Listing must not load/hash checkpoint weights")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guard):
            self.assertEqual(len(p.listing()[0]), 1)

    def test_legacy_schema_timestamp_metadata_and_replay_security(self):
        p = self.legacy()
        detail = p.detail("old")
        self.assertIn("summary_stats", detail)
        self.assertEqual(detail["source"], "legacy")
        metrics = p.metrics("old")
        self.assertEqual(metrics["points"][0]["timestamp"], 99)
        self.assertEqual(metrics["metadata"]["loss"]["label"], "TD loss")
        self.assertEqual(p.replay("old", "eval_replays/game.json"), self.legacy_trajectory())
        with self.assertRaises(KeyError):
            p.replay("old", "metrics.jsonl")
        for run_id in ("..", "../old", "/etc", "old/..", "old\\.."):
            self.assertIsNone(p.find(run_id))
            self.assertIsNone(_find_run_dir(run_id))

    def test_legacy_symlink_escape_not_read(self):
        p = self.legacy()
        run = self.root / "legacy/old"
        (run / "notes.md").symlink_to("/etc/passwd")
        self.assertIsNone(p.find("old"))
        self.assertEqual(p.listing()[0], [])

    def test_cross_provider_dedup_requires_exact_source_path(self):
        legacy = self.legacy()
        self.legacy_evaluation(legacy)
        run = legacy.require("old")
        (run / "notes.md").write_text("Unique legacy research notes")
        (run / "config.yaml").write_text("algorithm: DoubleDQN\nlegacy_only: retained\ntraining:\n  seed: 42\n")
        write(run / "experiment.json", {"training": {"algorithm": "policy_distillation"}})
        checkpoint = run / "checkpoints/ep00500_step154339.pt"
        (self.root / "experiments/comparison/architecture-catalog.json").unlink()
        self.rows = [registration(source=checkpoint)]
        service = UnifiedRunService()
        result = service.listing()
        self.assertEqual(result["count"], 1)
        tracked = result["runs"][0]
        self.assertEqual(tracked["run_id"], "registered")
        self.assertEqual(tracked["source"], "mlflow")
        self.assertEqual(tracked["provenance"]["sources"], ["legacy", "mlflow"])
        self.assertEqual(tracked["provenance"]["legacy_run_ids"], ["old"])
        for flag in ("has_notes", "has_metrics", "has_benchmarks", "has_eval_replays", "has_checkpoints"):
            self.assertTrue(tracked[flag], flag)
        provider = service.provider_for("registered")
        self.assertIsInstance(provider, CompositeAdapter)
        detail = provider.detail("registered")
        self.assertIn("Unique legacy", detail["notes"])
        self.assertEqual(detail["config"]["legacy_only"], "retained")
        self.assertEqual(detail["config"]["training"], {"algorithm": "policy_distillation", "seed": 42})
        self.assertEqual(detail["provenance"]["legacy_config"]["algorithm"], "DoubleDQN")
        self.assertEqual(detail["provenance"]["mlflow_run_ids"], ["registered"])
        self.assertEqual(len(detail["checkpoints"]), 1)
        self.assertEqual(detail["checkpoints"][0]["sha256"], self.sha)
        self.assertTrue(detail["checkpoints"][0]["has_benchmark"])
        metrics = provider.metrics("registered")
        self.assertEqual(metrics["run_id"], "registered")
        self.assertEqual(metrics["points"][0]["values"]["loss"], .5)
        benchmark = provider.benchmarks("registered")["benchmarks"][0]
        replay = provider.replays("registered")["replays"][0]
        self.assertEqual(benchmark["source"], "legacy")
        self.assertIsNone(benchmark["protocol"])
        self.assertEqual(benchmark["evaluation_id"], replay["evaluation_id"])
        self.assertEqual(replay["replay_path"], "legacy/old/eval_replays/game.json")
        self.assertEqual(provider.replay("registered", replay["replay_path"])["states"], self.legacy_trajectory()["states"])
        self.assertIn("Unique legacy", service.provider_for("old").detail("old")["notes"])
        self.assertEqual(service.provider_for("old").replays("old")["count"], 1)
        self.assertIsInstance(service.provider_for("old"), LegacyAdapter)
        self.assertEqual(service.listing(source="legacy")["count"], 0)
        self.assertEqual(service.listing(source="mlflow")["count"], 1)

    def test_composite_does_not_match_names_or_parent_directories(self):
        legacy = self.legacy()
        for source in (None, self.source, legacy.require("old").parent,
                       self.root / "elsewhere/old/checkpoints/update1.pt"):
            with self.subTest(source=source):
                self.rows = [registration(source=source)]
                self.rows[0]["run_name"] = "old"
                service = UnifiedRunService()
                self.assertEqual(service.listing()["count"], 2)
                self.assertIsInstance(service.provider_for("registered"), MLflowAdapter)

    def test_composite_resolves_translated_exact_source_and_mlflow_alias(self):
        legacy = self.legacy()
        source = "/home/jihua/quoridor-data/legacy/old/checkpoints/update1.pt"
        self.rows = [registration(source=source), registration("second", source=source)]
        service = UnifiedRunService()
        self.assertEqual(service.listing()["count"], 1)
        provider = service.provider_for("second")
        self.assertIsInstance(provider, CompositeAdapter)
        self.assertEqual(provider.detail("second")["run_id"], "registered")
        self.assertEqual(provider.source_path, legacy.require("old"))

    def test_composite_curve_priority_fallback_and_conflict_safeguard(self):
        legacy = self.legacy()
        self.legacy_evaluation(legacy)
        checkpoint = legacy.require("old") / "checkpoints/ep00500_step154339.pt"
        (self.root / "experiments/comparison/architecture-catalog.json").unlink()
        self.rows = [registration(source=checkpoint)]
        self.rows[0]["metrics"] = {"loss": .1}
        # Force a true composite fallback, rather than the tracked adapter's own local parser.
        service = UnifiedRunService()
        provider = service.provider_for("registered")
        with patch.object(service.mlflow, "_metric_files", return_value=[]), patch.object(
                mlflow, "metric_history", return_value=[]):
            metrics = provider.metrics("registered")
        self.assertEqual(metrics["points"][0]["values"]["loss"], .5)
        self.assertEqual(metrics["provenance"]["source"], "legacy")
        self.assertEqual(metrics["metadata"]["loss"]["origin"]["run_id"], "old")
        with patch.object(mlflow, "metric_history", return_value=[{"step": 9, "value": .1}]):
            metrics = provider.metrics("registered")
        self.assertEqual(len(metrics["points"]), 1)
        self.assertEqual(metrics["points"][0]["values"]["loss"], .1)
        checkpoint.write_bytes(b"changed")
        with patch.object(mlflow, "metric_history", return_value=[]):
            metrics = provider.metrics("registered")
        self.assertEqual(metrics["points"], [])
        self.assertTrue(any("fallback blocked" in w for w in metrics["warnings"]))

    def test_composite_checkpoint_identity_is_not_filename_or_progress(self):
        legacy = self.legacy()
        self.legacy_evaluation(legacy)
        checkpoint = legacy.require("old") / "checkpoints/ep00500_step154339.pt"
        self.rows = [registration(source=checkpoint)]
        provider = UnifiedRunService().provider_for("registered")
        tracked = {"filename": checkpoint.name, "path": str(checkpoint), "sha256": self.sha,
                   "episode": 500, "step": 154339, "has_benchmark": False}
        local = {**tracked, "sha256": None, "has_benchmark": True}
        self.assertEqual(len(provider._checkpoints([tracked], [local])), 1)
        self.assertTrue(provider._checkpoints([tracked], [local])[0]["has_benchmark"])
        other_path = {**local, "path": str(legacy.require("old") / "different" / checkpoint.name)}
        self.assertEqual(len(provider._checkpoints([tracked], [other_path])), 2)
        conflict = {**local, "sha256": "b" * 64}
        self.assertEqual(len(provider._checkpoints([tracked], [conflict])), 2)

    def test_composite_app_methods_preserve_both_evidence_sources(self):
        legacy = self.legacy()
        self.legacy_evaluation(legacy)
        self.checkpoint = legacy.require("old") / "checkpoints/ep00500_step154339.pt"
        (self.root / "experiments/comparison/architecture-catalog.json").unlink()
        arena, _, _ = self.arena()
        (self.root / "arena").mkdir()
        arena.rename(self.root / "arena/composite")
        self.rows = [registration(source=self.checkpoint)]
        service = UnifiedRunService()
        canonical = "registered"
        app = create_app()
        app.state.runs = service
        routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
        self.assertEqual(routes["/api/runs"]()["count"], 1)
        self.assertEqual(routes["/api/summary"]()["total_runs"], 1)
        detail = routes["/api/runs/{run_id}"](canonical)
        self.assertEqual(detail["provenance"]["sources"], ["legacy", "mlflow"])
        for method in ("metrics", "benchmarks", "diagnosis"):
            self.assertEqual(routes[f"/api/runs/{{run_id}}/{method}"](canonical)["run_id"], canonical)
        benchmarks = routes["/api/runs/{run_id}/benchmarks"](canonical)["benchmarks"]
        self.assertEqual(len(benchmarks), 3)
        self.assertEqual({b["protocol"] for b in benchmarks}, {"arena_paired", "control", None})
        replays = routes["/api/runs/{run_id}/eval-replays"](canonical)["replays"]
        self.assertEqual(len(replays), 2)
        self.assertEqual(len({r["replay_path"] for r in replays}), 2)
        local = next(r for r in replays if r.get("source") == "legacy")
        endpoint = routes["/api/runs/{run_id}/eval-replays/{replay_path:path}"]
        self.assertEqual(endpoint(canonical, local["replay_path"])["run_id"], canonical)
        self.assertEqual(endpoint("old", "eval_replays/game.json"), self.legacy_trajectory())
        tracked = next(r for r in replays if r["replay_path"].startswith("arena/"))
        converter = types.ModuleType("backend.replay")
        from unittest.mock import Mock
        converter.arena_replay = Mock(return_value={"run_id": canonical, "states": []})
        with patch.dict(sys.modules, {"backend.replay": converter}):
            self.assertEqual(endpoint(canonical, tracked["replay_path"])["run_id"], canonical)
        converter.arena_replay.assert_called_once()
        from fastapi import HTTPException
        for path in ("legacy/old/metrics.jsonl", "legacy/other/eval_replays/game.json",
                     "legacy/old/../old/eval_replays/game.json", "eval_replays/game.json"):
            with self.subTest(path=path), self.assertRaises(HTTPException) as error:
                endpoint(canonical, path)
            self.assertEqual(error.exception.status_code, 404)

    def test_filters_and_summary_share_unified_list_with_offline_warning(self):
        self.legacy()
        service = UnifiedRunService()
        self.assertEqual(service.listing(source="mlflow", experiment="9", status="finished")["count"], 1)
        self.assertEqual(service.listing(source="legacy")["count"], 1)
        self.assertEqual(service.listing(algorithm="nonexistent")["count"], 0)
        self.exp_mock.side_effect = RuntimeError("offline")
        service = UnifiedRunService()
        result = service.listing()
        self.assertEqual(result["count"], 1)
        self.assertTrue(any("offline" in x for x in result["warnings"]))
        app = create_app()
        app.state.runs = service
        routes = {r.path: r.endpoint for r in app.routes if hasattr(r, "endpoint")}
        self.assertEqual(routes["/api/summary"]()["total_runs"], result["count"])
        self.assertEqual(routes["/api/runs/{run_id}/metrics"]("old")["run_id"], "old")


class PaginationTests(unittest.TestCase):
    def test_all_search_pages_are_read(self):
        for func, args, key, row in ((mlflow.experiments, (), "experiments", {"experiment_id": "9", "name": "x", "lifecycle_stage": "active"}),
                                    (mlflow.runs, ("9",), "runs", {"info": {"run_id": "x", "status": "FINISHED"}}),
                                    (mlflow.metric_history, ("x", "loss"), "metrics", {"value": .1})):
            with self.subTest(key=key), patch.object(mlflow, "_request", side_effect=[{key: [row], "next_page_token": "next"}, {key: [row]}]) as request:
                self.assertEqual(len(func(*args)), 2)
                self.assertIn("next", str(request.call_args))

    def test_repeated_cursor_is_not_silent_truncation(self):
        with patch.object(mlflow, "_request", return_value={"metrics": [], "next_page_token": "loop"}):
            with self.assertRaises(RuntimeError):
                mlflow.metric_history("x", "loss")


if __name__ == "__main__":
    unittest.main()