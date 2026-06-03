import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class StressTest:
    def __init__(self, initial_value: float, var_1j: float):
        self.initial_value = initial_value
        self.var_1j = var_1j

    def simulate_lehman(self, daily_loss: float, days: int, correlation: float) -> np.ndarray:
        daily_loss_dollars = daily_loss * self.initial_value
        mean = np.full(days, -daily_loss_dollars)
        cov = np.full((days, days), correlation * daily_loss_dollars ** 2)
        np.fill_diagonal(cov, daily_loss_dollars ** 2)
        losses = np.random.multivariate_normal(mean, cov)
        return self.initial_value + np.cumsum(losses)

    def simulate_covid(self, shock_loss: float, volatility_multiplier: float, days: int) -> np.ndarray:
        shock_value = self.initial_value * (1 - shock_loss)
        daily_volatility = 0.02 * volatility_multiplier * shock_value
        daily_losses = np.random.normal(0, daily_volatility, size=days)
        return shock_value + np.cumsum(daily_losses)

    def simulate_interest_rate(
        self,
        weights: dict,
        tech_loss: float,
        value_loss: float,
        energy_gain: float,
        days: int,
    ) -> np.ndarray:
        tech_value = self.initial_value * weights['tech'] * (1 - tech_loss)
        value_value = self.initial_value * weights['value'] * (1 - value_loss)
        energy_value = self.initial_value * weights['energy'] * (1 + energy_gain)

        daily_pnl = (
            np.random.normal(0, 0.02, size=days) * tech_value
            + np.random.normal(0, 0.01, size=days) * value_value
            + np.random.normal(0, 0.03, size=days) * energy_value
        )
        return tech_value + value_value + energy_value + np.cumsum(daily_pnl)


if __name__ == "__main__":
    initial_value = 1_000_000
    var_1j = 40_284

    st = StressTest(initial_value=initial_value, var_1j=var_1j)
    weights = {'tech': 0.45, 'value': 0.35, 'energy': 0.20}

    lehman_values = st.simulate_lehman(daily_loss=0.05, days=10, correlation=0.3)
    covid_values = st.simulate_covid(shock_loss=0.30, volatility_multiplier=3, days=20)
    interest_rate_values = st.simulate_interest_rate(
        weights, tech_loss=0.25, value_loss=0.10, energy_gain=0.15, days=20
    )

    portfolio_loss_lehman = initial_value - lehman_values[-1]
    portfolio_loss_covid = initial_value - covid_values[-1]
    portfolio_loss_interest_rate = initial_value - interest_rate_values[-1]

    first_day_loss_lehman = initial_value - lehman_values[0]
    first_day_loss_covid = initial_value * 0.30
    first_day_loss_ir = initial_value - (
        initial_value * weights['tech']   * (1 - 0.25)
        + initial_value * weights['value']  * (1 - 0.10)
        + initial_value * weights['energy'] * (1 + 0.15)
    )

    print("\n=== Option 1 — Perte J1 vs VaR journalière 99% ===")
    print(f"{'':25s} {'Perte J1':>12s}   {'VaR 1j':>12s}   {'Ratio':>6s}   Décision")
    for name, loss in [("Lehman 2008", first_day_loss_lehman),
                       ("COVID-19 2020", first_day_loss_covid),
                       ("Hausse des taux 2022", first_day_loss_ir)]:
        ratio = loss / var_1j
        decision = "DEPASSE VaR" if loss > var_1j else "Couverte"
        print(f"  {name:23s} {loss:>12,.0f} $   {var_1j:>12,.0f} $   {ratio:>5.2f}x   {decision}")

    var_10j = var_1j * np.sqrt(10)
    var_20j = var_1j * np.sqrt(20)

    print("\n=== Option 2 — Perte cumulée vs VaR N jours (√t) ===")
    print(f"{'':25s} {'Perte cum.':>12s}   {'VaR Nj':>12s}   {'Ratio':>6s}")
    for name, loss, var_nj in [
        ("Lehman 2008 (10j)",     portfolio_loss_lehman,       var_10j),
        ("COVID-19 2020 (20j)",   portfolio_loss_covid,         var_20j),
        ("Hausse des taux (20j)", portfolio_loss_interest_rate, var_20j),
    ]:
        ratio = loss / var_nj
        print(f"  {name:23s} {loss:>12,.0f} $   {var_nj:>12,.0f} $   {ratio:>5.2f}x")

    plt.figure(figsize=(12, 6))
    plt.plot(lehman_values, label='Lehman 2008', color='red')
    plt.plot(covid_values, label='COVID-19 2020', color='orange')
    plt.plot(interest_rate_values, label='Hausse des taux 2022', color='green')
    plt.axhline(initial_value, color='gray', linewidth=0.8, linestyle='--', label='Valeur initiale')
    plt.title('Valeur du portefeuille sous différents scénarios de stress')
    plt.xlabel('Jours')
    plt.ylabel('Valeur ($)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'stress_test_scenarios_Q6.png')
    plt.show()

    labels = ['Lehman\n(10j)', 'COVID\n(20j)', 'Hausse taux\n(20j)']
    losses = [portfolio_loss_lehman, portfolio_loss_covid, portfolio_loss_interest_rate]
    var_nj_list = [var_10j, var_20j, var_20j]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, losses, width, label='Perte cumulée',
           color=['red', 'orange', 'green'], alpha=0.8)
    ax.bar(x + width / 2, var_nj_list, width, label='VaR N jours (√t)', color='steelblue', alpha=0.8)
    ax.set_title('Perte cumulée vs VaR N jours — Comparaison des scénarios de stress')
    ax.set_ylabel('Montant ($)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / 'loss_stress_scenarios_Q6.png')
    plt.show()
