"""Unit tests for resolve_device() in scripts/train_dqn.py.

These tests run without CUDA — CUDA-specific branches are guarded by
torch.cuda.is_available().
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

# Make sure repo root is on path so the script can be imported without
# executing main().
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from scripts.train_dqn import resolve_device


class TestResolveDeviceCpu:
    def test_cpu_returns_cpu_device(self):
        d = resolve_device("cpu")
        assert d == torch.device("cpu")

    def test_cpu_uppercase_still_works(self):
        d = resolve_device("CPU")
        assert d == torch.device("cpu")

    def test_cpu_with_whitespace(self):
        d = resolve_device("  cpu  ")
        assert d == torch.device("cpu")


class TestResolveDeviceAuto:
    def test_auto_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            d = resolve_device("auto")
        assert d.type == "cuda"

    def test_auto_returns_cpu_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            d = resolve_device("auto")
        assert d == torch.device("cpu")


class TestResolveDeviceCuda:
    def test_cuda_raises_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError, match="cuda.is_available"):
                resolve_device("cuda")

    def test_cuda_returns_cuda_device_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            d = resolve_device("cuda")
        assert d.type == "cuda"

    def test_cuda_index_raises_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            with pytest.raises(RuntimeError):
                resolve_device("cuda:0")

    def test_cuda_index_ok_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            d = resolve_device("cuda:0")
        assert d.type == "cuda"


class TestResolveDeviceInvalid:
    def test_invalid_string_raises_value_error(self):
        with pytest.raises(ValueError, match="Unrecognised"):
            resolve_device("tpu")

    def test_empty_string_raises_value_error(self):
        with pytest.raises(ValueError):
            resolve_device("")
