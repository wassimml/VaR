import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import chi2

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class Backtester:
    def __init__(self, returns, weights, portfolio_value: float, alpha: float, window_size: int):
        self.returns = returns
        self.weights = np.array(weights)
        self.portfolio_value = portfolio_value
        self.alpha = alpha
        self.window_size = window_size

    def run(self):
        portfolio_returns = self.returns.dot(self.weights)
        actual_pnl = portfolio_returns * self.portfolio_value

        rolling_historical_var = (
            -portfolio_returns.rolling(window=self.window_size).quantile(1 - self.alpha)
            * self.portfolio_value
        )

        var_forecast = rolling_historical_var.shift(1).dropna()
        pnl_aligned = actual_pnl.loc[var_forecast.index]

        is_violation = pnl_aligned < -var_forecast
        n_violations = int(is_violation.sum())
        violation_rate = n_violations / len(var_forecast)
        expected_rate = 1 - self.alpha

        print(f"\nBacktesting VaR {self.alpha*100:.0f}% (fenêtre {self.window_size}j)")
        print(f"Période         : {var_forecast.index[0].date()} → {var_forecast.index[-1].date()}")
        print(f"Jours testés    : {len(var_forecast)}")
        print(f"Violations      : {n_violations}")
        print(f"Taux observé    : {violation_rate:.2%}  (attendu : {expected_rate:.2%})")

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(pnl_aligned.index, pnl_aligned.values,
                color='steelblue', linewidth=0.8, label='P&L réel (J)')
        ax.plot(var_forecast.index, -var_forecast.values,
                color='orange', linewidth=1.5, linestyle='--',
                label=f'−VaR prévue J−1 ({self.alpha*100:.0f}%)')

        violation_dates = pnl_aligned.index[is_violation]
        ax.scatter(violation_dates, pnl_aligned.loc[violation_dates],
                   color='red', zorder=5, s=20, label=f'Violations ({n_violations})')

        ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
        ax.set_title(
            f'Backtesting VaR historique {self.alpha*100:.0f}% — P&L réel vs VaR prévue (J−1)',
            fontsize=13,
        )
        ax.set_xlabel('Date')
        ax.set_ylabel('P&L ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / f'backtest_var_{int(self.alpha*100)}.png', dpi=150, bbox_inches='tight')
        plt.show()

        n = len(var_forecast)
        x = n_violations
        p = expected_rate
        ph = violation_rate
        L0 = x * np.log(p) + (n - x) * np.log(1 - p)
        L1 = x * np.log(ph) + (n - x) * np.log(1 - ph)
        LR = -2 * (L0 - L1)
        p_value = 1 - chi2.cdf(LR, df=1)
        print(f"Test de Kupiec : LR={LR:.2f}, p-value={p_value:.4f}")


if __name__ == "__main__":
    portfolio_value = 1_000_000

    weights = {
        'AAPL': 0.25,
        'MSFT': 0.20,
        'JPM':  0.20,
        'XOM':  0.20,
        'JNJ':  0.15,
    }

    data = yf.download(list(weights.keys()), start="2019-01-01", end="2024-01-01", auto_adjust=False)
    daily_returns = np.log(data['Adj Close'].pct_change().dropna() + 1)

    backtester = Backtester(
        returns=daily_returns,
        weights=list(weights.values()),
        portfolio_value=portfolio_value,
        alpha=0.99,
        window_size=252,
    )
    backtester.run()
