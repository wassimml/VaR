import numpy as np
import pandas as pd
import pytest
from backtest import Backtester


class TestBacktesterInit:
    def test_alpha_stored(self, backtester):
        assert backtester.alpha == 0.99

    def test_window_size_stored(self, backtester):
        assert backtester.window_size == 100

    def test_portfolio_value_stored(self, backtester):
        assert backtester.portfolio_value == 1_000_000.0

    def test_weights_converted_to_numpy(self, backtester):
        assert isinstance(backtester.weights, np.ndarray)


class TestBacktesterRun:
    def test_smoke(self, backtester, capsys):
        backtester.run()
        out = capsys.readouterr().out
        assert "Violations" in out
        assert "Kupiec" in out

    def test_prints_observed_rate(self, backtester, capsys):
        backtester.run()
        out = capsys.readouterr().out
        assert "Taux observé" in out

    def test_zero_violations_when_portfolio_always_gains(self, weights, portfolio_value, capsys):
        # All daily returns = +1 % → rolling VaR is negative → no violation possible
        n = 400
        index = pd.date_range("2019-01-01", periods=n, freq="B")
        returns = pd.DataFrame(np.full((n, 5), 0.01), columns=list("ABCDE"), index=index)
        bt = Backtester(returns, weights, portfolio_value, alpha=0.99, window_size=100)
        bt.run()
        out = capsys.readouterr().out
        assert "Violations      : 0" in out
