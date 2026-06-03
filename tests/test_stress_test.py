import numpy as np
import pytest
from stress_test import StressTest

WEIGHTS = {"tech": 0.45, "value": 0.35, "energy": 0.20}


class TestSimulateLehman:
    def test_output_length(self, stress):
        result = stress.simulate_lehman(daily_loss=0.05, days=10, correlation=0.3)
        assert len(result) == 10

    def test_reproducible(self, stress):
        np.random.seed(42)
        r1 = stress.simulate_lehman(0.05, 10, 0.3)
        np.random.seed(42)
        r2 = stress.simulate_lehman(0.05, 10, 0.3)
        np.testing.assert_array_equal(r1, r2)

    def test_mean_final_value_below_initial_with_positive_loss(self, stress):
        # E[portfolio_values[-1]] = initial - daily_loss_dollars * days = 500 000
        results = np.array([
            stress.simulate_lehman(daily_loss=0.05, days=10, correlation=0.0)[-1]
            for _ in range(2_000)
        ])
        expected_mean = stress.initial_value - 0.05 * stress.initial_value * 10
        np.testing.assert_allclose(results.mean(), expected_mean, rtol=0.05)


class TestSimulateCovid:
    def test_output_length(self, stress):
        result = stress.simulate_covid(shock_loss=0.30, volatility_multiplier=3, days=20)
        assert len(result) == 20

    def test_reproducible(self, stress):
        np.random.seed(42)
        r1 = stress.simulate_covid(0.30, 3, 20)
        np.random.seed(42)
        r2 = stress.simulate_covid(0.30, 3, 20)
        np.testing.assert_array_equal(r1, r2)

    def test_zero_shock_mean_stays_at_initial(self, stress):
        # With shock_loss=0 the trajectory is centered on initial_value
        first_values = [
            stress.simulate_covid(shock_loss=0.0, volatility_multiplier=1, days=1)[0]
            for _ in range(2_000)
        ]
        np.testing.assert_allclose(
            np.mean(first_values), stress.initial_value, rtol=0.02
        )

    def test_full_shock_wipes_portfolio(self, stress):
        # With shock_loss=1.0 the portfolio value is identically zero
        result = stress.simulate_covid(shock_loss=1.0, volatility_multiplier=1, days=5)
        np.testing.assert_array_equal(result, np.zeros(5))


class TestSimulateInterestRate:
    def test_output_length(self, stress):
        result = stress.simulate_interest_rate(
            WEIGHTS, tech_loss=0.25, value_loss=0.10, energy_gain=0.15, days=20
        )
        assert len(result) == 20

    def test_reproducible(self, stress):
        np.random.seed(42)
        r1 = stress.simulate_interest_rate(WEIGHTS, 0.25, 0.10, 0.15, 20)
        np.random.seed(42)
        r2 = stress.simulate_interest_rate(WEIGHTS, 0.25, 0.10, 0.15, 20)
        np.testing.assert_array_equal(r1, r2)

    def test_deterministic_initial_portfolio_value(self, stress):
        # The starting level (before random P&L) is fully determined by shocks
        tech_v  = stress.initial_value * WEIGHTS["tech"]   * (1 - 0.25)
        value_v = stress.initial_value * WEIGHTS["value"]  * (1 - 0.10)
        energy_v = stress.initial_value * WEIGHTS["energy"] * (1 + 0.15)
        expected_level = tech_v + value_v + energy_v  # 882 500

        # Mean of first portfolio value over many sims ≈ expected_level (noise has mean 0)
        first_values = [
            stress.simulate_interest_rate(WEIGHTS, 0.25, 0.10, 0.15, 1)[0]
            for _ in range(2_000)
        ]
        np.testing.assert_allclose(np.mean(first_values), expected_level, rtol=0.02)
