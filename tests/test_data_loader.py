import numpy as np
import pandas as pd
import pytest
from data_loader import DataLoader


def make_price_df(prices):
    """Build the minimal DataFrame that DataLoader.log_returns expects."""
    return pd.DataFrame({"Adj Close": prices})


class TestLogReturns:
    def test_positive_returns_for_rising_prices(self):
        data = make_price_df([100.0, 110.0, 121.0])
        result = DataLoader.log_returns(data)
        assert (result > 0).all()

    def test_negative_returns_for_falling_prices(self):
        data = make_price_df([121.0, 110.0, 100.0])
        result = DataLoader.log_returns(data)
        assert (result < 0).all()

    def test_zero_returns_for_flat_prices(self):
        data = make_price_df([100.0, 100.0, 100.0])
        result = DataLoader.log_returns(data)
        np.testing.assert_allclose(result.values, 0.0)

    def test_output_length_is_n_minus_one(self):
        n = 10
        prices = np.linspace(100.0, 200.0, n).tolist()
        result = DataLoader.log_returns(make_price_df(prices))
        assert len(result) == n - 1

    def test_log_return_approximates_simple_return_for_small_move(self):
        # log(1 + r) ≈ r for small r
        data = make_price_df([100.0, 101.0])
        result = DataLoader.log_returns(data)
        np.testing.assert_allclose(result.values, [0.01], rtol=5e-3)

    def test_symmetry_up_down(self):
        # Prices end below start (99 < 100) → net log-return must be negative
        data = make_price_df([100.0, 110.0, 99.0])
        result = DataLoader.log_returns(data)
        assert result.iloc[0] > 0   # up leg
        assert result.iloc[1] < 0   # down leg
        assert result.sum() < 0     # net: 99 < 100

    def test_returns_pandas_series(self):
        data = make_price_df([100.0, 110.0, 121.0])
        result = DataLoader.log_returns(data)
        assert isinstance(result, pd.Series)
