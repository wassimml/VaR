import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats


### Scenario A: Lehman 2008 
### Actions : -5% par jour pendant 10 jours  |  Corrélations : +0.3 (hausse des corrélations en crise) 

def simulate_lehman_scenario(initial_value, daily_loss, days, correlation):
    """
    simulate_lehman_scenario simulates a portfolio's value over a specified number of days, 
    given an initial value, daily loss percentage, and correlation between assets.

    Parameters:
    initial_value (float): The initial value of the portfolio. 
    daily_loss (float): The percentage loss per day (e.g., 0.05 for 5%).
    days (int): The number of days to simulate.
    correlation (float): The correlation between daily losses (e.g., 0.3 for 30%).
    """

    daily_loss_dollars = daily_loss * initial_value  # perte journalière en dollars

    mean = np.full(days, -daily_loss_dollars)
    cov  = np.full((days, days), correlation * daily_loss_dollars**2)
    np.fill_diagonal(cov, daily_loss_dollars**2)

    losses = np.random.multivariate_normal(mean, cov)
    portfolio_values = initial_value + np.cumsum(losses)

    return portfolio_values

# Scenario B : COVID-19 2020
### Actions : choc ponctuel de -30%  | Volatilité x3 pendant 20 jours

def simulate_covid_scenario(initial_value, shock_loss, volatility_multiplier, days):
    """
    simulate_covid_scenario simulates a portfolio's value over a specified number of days, 
    given an initial value, shock loss percentage, volatility multiplier, and number of days.

    Parameters:
    initial_value (float): The initial value of the portfolio.
    shock_loss (float): The percentage loss due to the shock (e.g., 0.30 for 30%).
    volatility_multiplier (float): The factor by which volatility is increased (e.g., 3 for tripled volatility).
    days (int): The number of days to simulate after the shock.

    Hypothesis : 
    - Variabilité des pertes quotidiennes avant le choc : 2% (0.02)
    """

    shock_value = initial_value * (1 - shock_loss)
    daily_volatility = 0.02 * volatility_multiplier * shock_value
    daily_losses = np.random.normal(0, daily_volatility, size=days) 
    portfolio_values = shock_value + np.cumsum(daily_losses)

    return portfolio_values

# Scenario C : Hausse des taux d’intérêt 2022
### Actions tech de -25% | Action value : -10%  | Energie +15% 

def simulate_interest_rate_scenario(portfolio_value, weights, tech_loss, value_loss, energy_gain, days):
    """
    simulate_interest_rate_scenario simulates a portfolio's value over a specified number of days, 
    given an initial value, losses for tech and value stocks, gain for energy stocks, and number of days.

    Parameters:
    portfolio_value (float): The initial value of the portfolio.
    weights (dict): The keys are 'tech', 'value', and 'energy', and the values are the respective weights in the portfolio (e.g., {'tech': 0.4, 'value': 0.4, 'energy': 0.2}).
    tech_loss (float): The percentage loss for tech stocks (e.g., 0.25 for 25%).
    value_loss (float): The percentage loss for value stocks (e.g., 0.10 for 10%).
    energy_gain (float): The percentage gain for energy stocks (e.g., 0.15 for 15%).
    days (int): The number of days to simulate.

    Hypothesis :
    - Choc ponctuel de -25% pour les tech, -10% pour les value et +15% pour les energy
    - Variabilité des pertes quotidiennes pour les tech : 2% (0.02)
    - Variabilité des pertes quotidiennes pour les value : 1% (0.01
    - Variabilité des gains quotidiens pour les energy : 3% (0.03)
    """

    tech_value = portfolio_value * weights['tech'] * (1 - tech_loss)
    value_value = portfolio_value * weights['value'] * (1 - value_loss)
    energy_value = portfolio_value * weights['energy'] * (1 + energy_gain)

    daily_tech_losses = np.random.normal(0, 0.02, size=days) * tech_value
    daily_value_losses = np.random.normal(0, 0.01, size=days) * value_value
    daily_energy_gains = np.random.normal(0, 0.03, size=days) * energy_value

    portfolio_values = tech_value + value_value + energy_value + np.cumsum(daily_tech_losses + daily_value_losses + daily_energy_gains)

    return portfolio_values

if __name__ == "__main__":
    initial_value = 1000000  # 1 million dollars
    var_1j = 40284            # VaR historique 99% journalière

    # --- Simulations ---
    lehman_values = simulate_lehman_scenario(initial_value, daily_loss=0.05, days=10, correlation=0.3)
    portfolio_loss_lehman = initial_value - lehman_values[-1]

    covid_values = simulate_covid_scenario(initial_value, shock_loss=0.30, volatility_multiplier=3, days=20)
    portfolio_loss_covid = initial_value - covid_values[-1]

    weights = {'tech': 0.45, 'value': 0.35, 'energy': 0.20}
    interest_rate_values = simulate_interest_rate_scenario(initial_value, weights, tech_loss=0.25, value_loss=0.10, energy_gain=0.15, days=20)
    portfolio_loss_interest_rate = initial_value - interest_rate_values[-1]

    # --- Option 1 : perte du 1er jour vs VaR journalière 99% ---
    first_day_loss_lehman = initial_value - lehman_values[0]
    first_day_loss_covid  = initial_value * 0.30   # choc instantané J0
    first_day_loss_ir     = initial_value - (
        initial_value * weights['tech']   * (1 - 0.25) +
        initial_value * weights['value']  * (1 - 0.10) +
        initial_value * weights['energy'] * (1 + 0.15)
    )

    print("\n=== Option 1 — Perte J1 vs VaR journalière 99% ===")
    print(f"{'':25s} {'Perte J1':>12s}   {'VaR 1j':>12s}   {'Ratio':>6s}   Décision")
    for name, loss in [("Lehman 2008", first_day_loss_lehman),
                       ("COVID-19 2020", first_day_loss_covid),
                       ("Hausse des taux 2022", first_day_loss_ir)]:
        ratio = loss / var_1j
        decision = "DEPASSE VaR" if loss > var_1j else "Couverte"
        print(f"  {name:23s} {loss:>12,.0f} $   {var_1j:>12,.0f} $   {ratio:>5.2f}x   {decision}")

    # --- Option 2 : perte cumulée vs VaR N jours (règle de la racine du temps) ---
    var_10j = var_1j * np.sqrt(10)
    var_20j = var_1j * np.sqrt(20)

    print("\n=== Option 2 — Perte cumulée vs VaR N jours (√t) ===")
    print(f"{'':25s} {'Perte cum.':>12s}   {'VaR Nj':>12s}   {'Ratio':>6s}")
    for name, loss, var_nj in [("Lehman 2008 (10j)",       portfolio_loss_lehman,          var_10j),
                                ("COVID-19 2020 (20j)",     portfolio_loss_covid,            var_20j),
                                ("Hausse des taux (20j)",   portfolio_loss_interest_rate,    var_20j)]:
        ratio = loss / var_nj
        print(f"  {name:23s} {loss:>12,.0f} $   {var_nj:>12,.0f} $   {ratio:>5.2f}x")

    # --- Graphique trajectoires ---
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
    plt.savefig('reports/stress_test_scenarios_Q6.png')
    plt.show()

    # --- Graphique comparaison pertes cumulées vs VaR N jours ---
    labels = ['Lehman\n(10j)', 'COVID\n(20j)', 'Hausse taux\n(20j)']
    losses = [portfolio_loss_lehman, portfolio_loss_covid, portfolio_loss_interest_rate]
    var_nj = [var_10j, var_20j, var_20j]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, losses, width, label='Perte cumulée', color=['red', 'orange', 'green'], alpha=0.8)
    ax.bar(x + width/2, var_nj,  width, label='VaR N jours (√t)', color='steelblue', alpha=0.8)
    ax.set_title('Perte cumulée vs VaR N jours — Comparaison des scénarios de stress')
    ax.set_ylabel('Montant ($)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig('reports/loss_stress_scenarios_Q6.png')
    plt.show()