"""Unit tests for the read-only MLflow Tracking adapter."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import mlflow


class MLflowAdapterTests(unittest.TestCase):
    def test_normalizes_tracking_run(self):
        payload = {
            "runs": [{
                "info": {"run_id": "run-1", "run_name": "candidate", "status": "FINISHED"},
                "data": {
                    "params": [{"key": "algorithm", "value": "policy_distillation"}],
                    "tags": [{"key": "checkpoint_sha256", "value": "abc"}],
                    "metrics": [{"key": "cross_entropy", "value": 0.2}],
                },
            }],
        }
        with patch.object(mlflow, "_request", return_value=payload):
            self.assertEqual(mlflow.runs("9"), [{
                "run_id": "run-1", "run_name": "candidate", "status": "FINISHED",
                "start_time": None, "end_time": None, "artifact_uri": None,
                "params": {"algorithm": "policy_distillation"},
                "tags": {"checkpoint_sha256": "abc"},
                "metrics": {"cross_entropy": 0.2},
            }])

    def test_metric_history_uses_requested_metric(self):
        with patch.object(mlflow, "_request", return_value={"metrics": [{"step": 12, "value": 0.2}]}) as request:
            self.assertEqual(mlflow.metric_history("run-1", "cross_entropy"), [{"step": 12, "value": 0.2}])
        self.assertIn("run_id=run-1", request.call_args.args[0])
        self.assertIn("metric_key=cross_entropy", request.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
