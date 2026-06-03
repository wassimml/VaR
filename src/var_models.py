import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class VaRCalculator:
    def __init__(self, returns, weights, portfolio_value: float):
        self.returns = returns
        self.weights = np.array(weights)
        self.portfolio_value = portfolio_value

    def historical(self, alpha: float) -> float:
        portfolio_returns = self.returns.dot(self.weights)
        return -np.percentile(portfolio_returns, (1 - alpha) * 100) * self.portfolio_value

    def rolling(self, alpha: float, window: int):
        portfolio_returns = self.returns.dot(self.weights)
        return (
            -portfolio_returns.rolling(window=window).quantile(1 - alpha)
            * self.portfolio_value
        )

    def cvar(self, alpha: float) -> float:
        portfolio_returns = self.returns.dot(self.weights)
        var_return = -np.percentile(portfolio_returns, (1 - alpha) * 100)
        return -portfolio_returns[portfolio_returns < -var_return].mean() * self.portfolio_value

    def parametric(self, alpha: float) -> float:
        cov_matrix = self.returns.cov()
        mu_p = self.weights @ self.returns.mean()
        sigma_p = np.sqrt(self.weights @ cov_matrix @ self.weights)
        z_score = abs(stats.norm.ppf(1 - alpha))
        return -(mu_p - z_score * sigma_p) * self.portfolio_value

    def montecarlo(self, alpha: float, n_simulations: int = 10000, graph: bool = False) -> float:
        mu_p = self.returns.mean().values
        cov = self.returns.cov().values
        simulated_returns = np.random.multivariate_normal(mu_p, cov, n_simulations)
        portfolio_pnl = simulated_returns @ self.weights * self.portfolio_value
        var = -np.percentile(portfolio_pnl, (1 - alpha) * 100)

        if graph:
            plt.figure(figsize=(10, 6))
            plt.hist(portfolio_pnl, bins=50, density=True, alpha=0.6, color='b')
            plt.title('Histogram of Simulated Portfolio P&L')
            plt.xlabel('P&L')
            plt.ylabel('Density')
            plt.gca().text(0.02, 0.97, f'Alpha : {alpha}', transform=plt.gca().transAxes,
                           fontsize=9, verticalalignment='top', horizontalalignment='left',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
            plt.axvline(-var, color='r', linestyle='dashed', linewidth=2,
                        label=f'VaR at {alpha*100}% confidence level')
            plt.legend()
            plt.grid()
            plt.savefig(REPORTS_DIR / f'montecarlo_var_histogram_{alpha}_Q2.png')
            plt.show()

        return var

    def compare_models(self):
        portfolio_returns = self.returns.dot(self.weights)
        alphas = [0.95, 0.99]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for ax, alpha in zip(axes, alphas):
            vh = self.historical(alpha)
            vp = self.parametric(alpha)
            vm = self.montecarlo(alpha)

            pnl_hist = portfolio_returns * self.portfolio_value
            ax.hist(pnl_hist, bins=80, density=True, alpha=0.4, color='steelblue',
                    label='Distribution historique')

            mu = float(portfolio_returns.mean()) * self.portfolio_value
            sigma = float(portfolio_returns.std()) * self.portfolio_value
            x = np.linspace(float(pnl_hist.min()), float(pnl_hist.max()), 500)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), color='darkorange', linewidth=2,
                    label='Distribution normale (paramétrique)')

            simulated_returns = np.random.multivariate_normal(
                self.returns.mean().values, self.returns.cov().values, 100000
            )
            pnl_mc = simulated_returns @ self.weights * self.portfolio_value
            ax.hist(pnl_mc, bins=80, density=True, alpha=0.3, color='green',
                    label='Distribution Monte Carlo')

            ax.axvline(-vh, color='steelblue', linestyle='--', linewidth=1.5,
                       label=f'VaR Hist.  = {vh:,.0f} $')
            ax.axvline(-vp, color='darkorange', linestyle='--', linewidth=1.5,
                       label=f'VaR Param. = {vp:,.0f} $')
            ax.axvline(-vm, color='green', linestyle='--', linewidth=1.5,
                       label=f'VaR MC     = {vm:,.0f} $')

            ax.set_title(f'Comparaison des distributions — VaR {int(alpha*100)}%', fontsize=13)
            ax.set_xlabel('P&L ($)')
            ax.set_ylabel('Densité')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle('Comparaison des trois méthodes VaR — Portefeuille 1 000 000 $',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'comparison_var_distributions.png', dpi=150, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    portfolio_value = 1_000_000

    weights = {
        'AAPL': 0.25,
        'MSFT': 0.20,
        'JPM':  0.20,
        'XOM':  0.20,
        'JNJ':  0.15,
    }
    seuil = [0.95, 0.99]

    data = yf.download(list(weights.keys()), start="2019-01-01", end="2024-01-01", auto_adjust=False)
    daily_returns = np.log(data['Adj Close'].pct_change().dropna() + 1)

    calc = VaRCalculator(daily_returns, list(weights.values()), portfolio_value)

    var1 = calc.historical(seuil[0])
    var2 = calc.historical(seuil[1])
    print(f"Historical VaR at {seuil[0]*100}% confidence level: {var1:.2f} $")
    print(f"Historical VaR at {seuil[1]*100}% confidence level: {var2:.2f} $")

    cvar1 = calc.cvar(seuil[0])
    cvar2 = calc.cvar(seuil[1])
    print(f"Historical CVaR at {seuil[0]*100}% confidence level: {cvar1:.2f} $")
    print(f"Historical CVaR at {seuil[1]*100}% confidence level: {cvar2:.2f} $")

    rolling_day = 100
    rolling_var1 = calc.rolling(seuil[0], rolling_day)
    rolling_var2 = calc.rolling(seuil[1], rolling_day)

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_var1, label=f'Rolling VaR at {seuil[0]*100}% confidence level')
    plt.plot(rolling_var2, label=f'Rolling VaR at {seuil[1]*100}% confidence level')
    plt.gca().text(0.02, 0.97, f'Days : {rolling_day}', transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.title('Rolling Historical VaR of the Portfolio')
    plt.xlabel('Date')
    plt.ylabel('VaR')
    plt.savefig(REPORTS_DIR / f'rolling_var_days_{rolling_day}_Q2.png')
    plt.legend()
    plt.show()

    pvar1 = calc.parametric(seuil[0])
    pvar2 = calc.parametric(seuil[1])
    print(f"Parametric VaR at {seuil[0]*100}% confidence level: {pvar1:.2f} $")
    print(f"Parametric VaR at {seuil[1]*100}% confidence level: {pvar2:.2f} $")

    mc_var1 = calc.montecarlo(seuil[0], graph=True)
    mc_var2 = calc.montecarlo(seuil[1], graph=True)
    print(f"Monte Carlo VaR at {seuil[0]*100}% confidence level: {mc_var1:.2f} $")
    print(f"Monte Carlo VaR at {seuil[1]*100}% confidence level: {mc_var2:.2f} $")

    mc_stability = [calc.montecarlo(seuil[0]) for _ in range(10)]
    print(f"Monte Carlo variance over 10 runs at {seuil[0]*100}% confidence level: {np.var(mc_stability):.2f}")

    simulation_counts = list(range(1000, 100_001, 2000))
    mc_convergence = [calc.montecarlo(seuil[0], n_simulations=n) for n in simulation_counts]

    plt.figure(figsize=(10, 6))
    plt.plot(simulation_counts, mc_convergence, marker='o')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Monte Carlo VaR')
    plt.title('Convergence of Monte Carlo VaR')
    plt.savefig(REPORTS_DIR / 'montecarlo_var_convergence_Q2.png')
    plt.show()

    calc.compare_models()
