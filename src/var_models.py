import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats

### Variance Historique 

def historical_var(returns, weights, alpha, portfolio_value) :
    """
    Calculate the historical Value at Risk (VaR) of a portfolio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of the assets in the portfolio.
    weights (list): A list of weights corresponding to each asset in the portfolio.
    alpha (float): The confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
    portfolio_value (float): The total value of the portfolio.

    Returns:
    float: The historical VaR of the portfolio.
    """
    # Calculate the portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate the historical VaR
    var = -np.percentile(portfolio_returns, (1 - alpha) * 100) * portfolio_value
    
    return var

def rolling_var(returns, weights, alpha, portfolio_value, window) :
    """
    Calculate the rolling historical Value at Risk (VaR) of a portfolio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of the assets in the portfolio.
    weights (list): A list of weights corresponding to each asset in the portfolio.
    alpha (float): The confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
    portfolio_value (float): The total value of the portfolio.
    window (int): The size of the rolling window in days.

    Returns:
    pd.Series: A Series containing the rolling historical VaR of the portfolio.
    """
    # Calculate the portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate the rolling historical VaR
    result = -portfolio_returns.rolling(window=window).quantile(1 - alpha)*portfolio_value

    return result

def CVaR(returns, weights, alpha, portfolio_value) :
    """
    Calculate the historical Conditional Value at Risk (CVaR) of a portfolio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of the assets in the portfolio.
    weights (list): A list of weights corresponding to each asset in the portfolio.
    alpha (float): The confidence level for CVaR calculation (e.g., 0.95 for 95% confidence).
    portfolio_value (float): The total value of the portfolio.

    Returns:
    float: The historical CVaR of the portfolio.
    """
    # Calculate the portfolio returns
    portfolio_returns = returns.dot(weights)
    
    # Calculate the historical VaR
    var = -np.percentile(portfolio_returns, (1 - alpha) * 100)
    
    # Calculate the CVaR, here var is already a return, so we take the mean of the returns that are less than var
    cvar = -portfolio_returns[portfolio_returns < -var].mean() * portfolio_value
    
    return cvar


def parametric_var(returns, weights, alpha, portfolio_value) :
    """
    Calculate the parametric Value at Risk (VaR) of a portfolio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of the assets in the portfolio.
    weights (list): A list of weights corresponding to each asset in the portfolio.
    alpha (float): The confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
    portfolio_value (float): The total value of the portfolio.

    Returns:
    float: The parametric VaR of the portfolio.
    """
    
    # Calculate the covariance matrix of the returns
    cov_matrix = returns.cov()
    
    # Calculate the mean and standard deviation of the portfolio returns
    weights = np.array(weights)
    mu_p = weights @ returns.mean()          # scalar: weighted portfolio mean
    sigma_p = np.sqrt(weights @ cov_matrix @ weights)  # scalar: portfolio std
    z_score = abs(stats.norm.ppf(1 - alpha))

    # Calculate the parametric VaR using the normal distribution
    var = -(mu_p - z_score * sigma_p) * portfolio_value
    
    return var


def montecarlo_var(returns, weights, alpha, portfolio_value, n_simulations=10000,graph=False) :
    """
    Calculate the Monte Carlo Value at Risk (VaR) of a portfolio.

    Parameters:
    returns (pd.DataFrame): A DataFrame containing the historical returns of the assets in the portfolio.
    weights (list): A list of weights corresponding to each asset in the portfolio.
    alpha (float): The confidence level for VaR calculation (e.g., 0.95 for 95% confidence).
    portfolio_value (float): The total value of the portfolio.
    n_simulations (int): The number of Monte Carlo simulations to perform.

    Returns:
    float: The Monte Carlo VaR of the portfolio.
    """
    
    # Calculate the covariance matrix of the returns
    cov_matrix = returns.cov()
    
    weights = np.array(weights)
    mu_p = returns.mean().values        # vecteur des moyennes par actif
    cov = cov_matrix.values             # matrice de covariance 

    # Simulate portfolio returns using Monte Carlo method
    simulated_returns = np.random.multivariate_normal(mu_p, cov, n_simulations)
    
    # Calculate the P&L of the portfolio for each simulation
    portfolio_pnl = simulated_returns @ weights * portfolio_value

    # Calculate the Monte Carlo VaR
    var = -np.percentile(portfolio_pnl, (1 - alpha) * 100)

    if graph :
        # Plot the histogram of the simulated P&L with VaR (red line)
        plt.figure(figsize=(10, 6))
        plt.hist(portfolio_pnl, bins=50, density=True, alpha=0.6, color='b')
        plt.title('Histogram of Simulated Portfolio P&L')
        plt.xlabel('P&L')
        plt.ylabel('Density')
        plt.gca().text(0.02, 0.97, f'Alpha : {alpha}', transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )
        plt.axvline(-var, color='r', linestyle='dashed', linewidth=2, label=f'VaR at {alpha*100}% confidence level')
        plt.legend()

        plt.grid()
        plt.savefig(f'reports/montecarlo_var_histogram_{alpha}_Q2.png') 
        plt.show()
    
    return var


def compare_var_models(portfolio_value, weights, returns) :

    weights = np.array(weights)
    portfolio_returns = returns.dot(weights)   # 1D series of portfolio returns

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    alphas = [0.95, 0.99]
    var_hist   = [historical_var(returns, weights, alphas[0], portfolio_value),
                  historical_var(returns, weights, alphas[1], portfolio_value)]
    var_param  = [parametric_var(returns, weights, alphas[0], portfolio_value),
                  parametric_var(returns, weights, alphas[1], portfolio_value)]
    var_mc     = [montecarlo_var(returns, weights, alphas[0], portfolio_value),
                  montecarlo_var(returns, weights, alphas[1], portfolio_value)]

    for i, (alpha, vh, vp, vm) in enumerate(zip(alphas, var_hist, var_param, var_mc)):
        ax = axes[i]

        # --- Historique : distribution des P&L réels ---
        pnl_hist = portfolio_returns * portfolio_value
        ax.hist(pnl_hist, bins=80, density=True, alpha=0.4, color='steelblue', label='Distribution historique')

        # --- Paramétrique : courbe normale ---
        mu    = float(portfolio_returns.mean()) * portfolio_value
        sigma = float(portfolio_returns.std())  * portfolio_value
        x = np.linspace(float(pnl_hist.min()), float(pnl_hist.max()), 500)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), color='darkorange', linewidth=2, label='Distribution normale (paramétrique)')

        # --- Monte Carlo : KDE des P&L simulés ---
        simulated_returns = np.random.multivariate_normal(
            returns.mean().values, returns.cov().values, 100000
        )
        pnl_mc = simulated_returns @ weights * portfolio_value
        ax.hist(pnl_mc, bins=80, density=True, alpha=0.3, color='green', label='Distribution Monte Carlo')

        # --- Lignes VaR ---
        ax.axvline(-vh, color='steelblue',   linestyle='--', linewidth=1.5, label=f'VaR Hist.  = {vh:,.0f} $')
        ax.axvline(-vp, color='darkorange',  linestyle='--', linewidth=1.5, label=f'VaR Param. = {vp:,.0f} $')
        ax.axvline(-vm, color='green',       linestyle='--', linewidth=1.5, label=f'VaR MC     = {vm:,.0f} $')

        ax.set_title(f'Comparaison des distributions — VaR {int(alpha*100)}%', fontsize=13)
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Densité')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Comparaison des trois méthodes VaR — Portefeuille 1 000 000 $', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/comparison_var_distributions.png', dpi=150, bbox_inches='tight')
    plt.show()  


if __name__ == "__main__":

    portfolio_value = 1000000  # 1 million $

    weights = {
        'AAPL': 0.25,
        'MSFT': 0.20,
        'JPM':  0.20,
        'XOM':  0.20,
        'JNJ':  0.15
    }

    seuil = [0.95, 0.99]

    data = yf.download(list(weights.keys()), start="2019-01-01", end="2024-01-01", auto_adjust=False)
    daily_returns = np.log(data['Adj Close'].pct_change().dropna() + 1) # pct_change() Calculate daily returns and drop the first NaN value
    var1 = historical_var(daily_returns, list(weights.values()), seuil[0], portfolio_value)
    var2 = historical_var(daily_returns, list(weights.values()), seuil[1], portfolio_value)

    print(f"Historical VaR at {seuil[0]*100}% confidence level: {var1:.2f} $")
    print(f"Historical VaR at {seuil[1]*100}% confidence level: {var2:.2f} $")

    cvar1 = CVaR(daily_returns, list(weights.values()), seuil[0], portfolio_value)
    cvar2 = CVaR(daily_returns, list(weights.values()), seuil[1], portfolio_value)

    print(f"Historical CVaR at {seuil[0]*100}% confidence level: {cvar1:.2f} $")
    print(f"Historical CVaR at {seuil[1]*100}% confidence level: {cvar2:.2f} $")

    rolling_day = 100
    rolling_var1 = rolling_var(daily_returns, list(weights.values()), seuil[0], portfolio_value, window=rolling_day)
    rolling_var2 = rolling_var(daily_returns, list(weights.values()), seuil[1], portfolio_value, window=rolling_day)
    
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_var1, label=f'Rolling VaR at {seuil[0]*100}% confidence level')
    plt.plot(rolling_var2, label=f'Rolling VaR at {seuil[1]*100}% confidence level')
    plt.gca().text(0.02, 0.97, f'Days : {rolling_day}', transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )
    plt.title('Rolling Historical VaR of the Portfolio')
    plt.xlabel('Date')
    plt.ylabel('VaR')
    plt.savefig(f'reports/rolling_var_days_{rolling_day}_Q2.png')  # Save the plot as an image file
    plt.legend()
    plt.show()


    Parametric_VaR1 = parametric_var(daily_returns, list(weights.values()), seuil[0], portfolio_value)
    Parametric_VaR2 = parametric_var(daily_returns, list(weights.values()), seuil[1], portfolio_value)

    print(f"Parametric VaR at {seuil[0]*100}% confidence level: {Parametric_VaR1:.2f} $")
    print(f"Parametric VaR at {seuil[1]*100}% confidence level: {Parametric_VaR2:.2f} $")


    MonteCarlo_VaR1 = montecarlo_var(daily_returns, list(weights.values()), seuil[0], portfolio_value,graph=True)
    MonteCarlo_VaR2 = montecarlo_var(daily_returns, list(weights.values()), seuil[1], portfolio_value,graph=True)
    
    print(f"Monte Carlo VaR at {seuil[0]*100}% confidence level: {MonteCarlo_VaR1:.2f} $")
    print(f"Monte Carlo VaR at {seuil[1]*100}% confidence level: {MonteCarlo_VaR2:.2f} $")

    # Test the stability of the Monte Carlo VaR by running it multiple times
    montecarlo_var_values = []
    for i in range(10):
        var = montecarlo_var(daily_returns, list(weights.values()), seuil[0], portfolio_value)
        montecarlo_var_values.append(var)

    print(f"Monte Carlo variance over 10 runs at {seuil[0]*100}% confidence level: {np.var(montecarlo_var_values):.2f}")


    # Progressively increase the number of simulations and observe the convergence of the Monte Carlo VaR
    simulation_counts = [i for i in range(1000, 100001, 2000)]
    montecarlo_var_convergence = []
    for n in simulation_counts:
        var = montecarlo_var(daily_returns, list(weights.values()), seuil[0], portfolio_value, n_simulations=n)
        montecarlo_var_convergence.append(var)
    
    plt.figure(figsize=(10, 6))
    plt.plot(simulation_counts, montecarlo_var_convergence, marker='o')
    plt.xlabel('Number of Simulations')
    plt.ylabel('Monte Carlo VaR')
    plt.title('Convergence of Monte Carlo VaR')
    plt.savefig(f'reports/montecarlo_var_convergence_Q2.png')  # Save the plot as an image file
    plt.show()

    compare_var_models(portfolio_value, list(weights.values()), daily_returns)