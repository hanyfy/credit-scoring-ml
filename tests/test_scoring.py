import numpy as np
import pytest

from scoring.monitoring import psi
from scoring.train import expected_cost, ks_stat


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=2000)
    assert psi(x, x) == pytest.approx(0.0, abs=1e-6)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.25


def test_ks_perfect_and_random():
    y = np.array([0, 0, 1, 1])
    assert ks_stat(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert ks_stat(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0


def test_expected_cost_matrix():
    y = np.array([1, 0])
    assert expected_cost(y, np.array([0.1, 0.1]), 0.5) == 5 / 2   # mauvais payeur accepté : coût 5
    assert expected_cost(y, np.array([0.9, 0.9]), 0.5) == 1 / 2   # bon payeur refusé : coût 1
