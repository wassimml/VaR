import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must come before pyplot import

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def mock_display(monkeypatch):
    """Suppress all plot display and file writes for every test."""
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "savefig", lambda *a, **kw: None)
    monkeypatch.setattr(Figure, "savefig", lambda *a, **kw: None)


@pytest.fixture(scope="session")
def sample_returns() -> pd.DataFrame:
    """500 days × 5 independent MVN assets (μ=0, σ=1%/day). Truly MVN so
    parametric and Monte Carlo VaR should agree with enough simulations."""
    rng = np.random.default_rng(42)
    data = rng.multivariate_normal(
        np.zeros(5),
        np.eye(5) * (0.01 ** 2),
        500,
    )
    index = pd.date_range("2019-01-01", periods=500, freq="B")
    return pd.DataFrame(data, columns=list("ABCDE"), index=index)


@pytest.fixture(scope="session")
def weights() -> list:
    return [0.2, 0.2, 0.2, 0.2, 0.2]


@pytest.fixture(scope="session")
def portfolio_value() -> float:
    return 1_000_000.0


@pytest.fixture(scope="session")
def calc(sample_returns, weights, portfolio_value):
    from var_models import VaRCalculator
    return VaRCalculator(sample_returns, weights, portfolio_value)


@pytest.fixture(scope="session")
def backtester(sample_returns, weights, portfolio_value):
    from backtest import Backtester
    return Backtester(sample_returns, weights, portfolio_value, alpha=0.99, window_size=100)


@pytest.fixture(scope="session")
def stress():
    from stress_test import StressTest
    return StressTest(initial_value=1_000_000.0, var_1j=40_284.0)
