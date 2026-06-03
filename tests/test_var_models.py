import numpy as np
import pandas as pd
import pytest
from var_models import VaRCalculator


class TestHistoricalVaR:
    def test_positive(self, calc):
        assert calc.historical(0.95) > 0

    def test_monotone_in_alpha(self, calc):
        assert calc.historical(0.99) > calc.historical(0.95)

    def test_scales_with_portfolio_value(self, sample_returns, weights):
        c1 = VaRCalculator(sample_returns, weights, 1_000_000)
        c2 = VaRCalculator(sample_returns, weights, 2_000_000)
        assert pytest.approx(c2.historical(0.95), rel=1e-9) == 2 * c1.historical(0.95)

    def test_alpha_above_one_raises(self, calc):
        with pytest.raises(Exception):
            calc.historical(1.1)

    def test_alpha_below_zero_raises(self, calc):
        with pytest.raises(Exception):
            calc.historical(-0.1)


class TestRollingVaR:
    def test_returns_series(self, calc):
        assert isinstance(calc.rolling(0.95, 50), pd.Series)

    def test_first_window_minus_one_are_nan(self, calc):
        window = 50
        result = calc.rolling(0.95, window)
        assert result.iloc[: window - 1].isna().all()

    def test_values_after_warmup_are_positive(self, calc):
        assert (calc.rolling(0.95, 50).dropna() > 0).all()

    def test_monotone_in_alpha(self, calc):
        r95 = calc.rolling(0.95, 50).dropna()
        r99 = calc.rolling(0.99, 50).dropna()
        assert (r99 >= r95).all()

    def test_output_length_equals_input_length(self, calc, sample_returns):
        result = calc.rolling(0.95, 50)
        assert len(result) == len(sample_returns)


class TestCVaR:
    def test_positive(self, calc):
        assert calc.cvar(0.95) > 0

    def test_cvar_geq_var_at_95(self, calc):
        assert calc.cvar(0.95) >= calc.historical(0.95)

    def test_cvar_geq_var_at_99(self, calc):
        assert calc.cvar(0.99) >= calc.historical(0.99)

    def test_monotone_in_alpha(self, calc):
        assert calc.cvar(0.99) > calc.cvar(0.95)


class TestParametricVaR:
    def test_positive(self, calc):
        assert calc.parametric(0.95) > 0

    def test_monotone_in_alpha(self, calc):
        assert calc.parametric(0.99) > calc.parametric(0.95)

    def test_scales_with_portfolio_value(self, sample_returns, weights):
        c1 = VaRCalculator(sample_returns, weights, 1_000_000)
        c2 = VaRCalculator(sample_returns, weights, 2_000_000)
        assert pytest.approx(c2.parametric(0.95), rel=1e-9) == 2 * c1.parametric(0.95)


class TestMonteCarloVaR:
    def test_positive(self, calc):
        np.random.seed(0)
        assert calc.montecarlo(0.95) > 0

    def test_reproducible_with_same_seed(self, calc):
        np.random.seed(7)
        r1 = calc.montecarlo(0.95)
        np.random.seed(7)
        r2 = calc.montecarlo(0.95)
        assert r1 == r2

    def test_monotone_in_alpha(self, calc):
        np.random.seed(0)
        mc95 = calc.montecarlo(0.95, n_simulations=50_000)
        np.random.seed(0)
        mc99 = calc.montecarlo(0.99, n_simulations=50_000)
        assert mc99 > mc95

    def test_converges_to_parametric(self, calc):
        # Fixture data is MVN → MC (also MVN) should match parametric closely
        param = calc.parametric(0.95)
        np.random.seed(0)
        mc = calc.montecarlo(0.95, n_simulations=300_000)
        assert abs(mc - param) / param < 0.02

    def test_zero_simulations_raises(self, calc):
        with pytest.raises(Exception):
            calc.montecarlo(0.95, n_simulations=0)

    def test_negative_simulations_raises(self, calc):
        with pytest.raises(Exception):
            calc.montecarlo(0.95, n_simulations=-1)


class TestCompareModels:
    def test_smoke(self, calc):
        calc.compare_models()
