import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import chi2

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

if __name__ == "__main__":
    alpha = 0.99

    # Calculate rolling historical VaR
    window_size = 252  # 1 year of trading days

    portfolio_value = 1000000  # 1 million $

    weights = {
        'AAPL': 0.25,
        'MSFT': 0.20,
        'JPM':  0.20,
        'XOM':  0.20,
        'JNJ':  0.15
    }


    data = yf.download(list(weights.keys()), start="2019-01-01", end="2024-01-01", auto_adjust=False)
    daily_returns = np.log(data['Adj Close'].pct_change().dropna() + 1) # pct_change() Calculate daily returns and drop the first NaN value
   

    w = np.array(list(weights.values()))
    portfolio_returns = daily_returns.dot(w)                  # 1D : rendement journalier du portefeuille
    actual_pnl        = portfolio_returns * portfolio_value   # P&L réel du jour J ($)

    rolling_historical_var = rolling_var(daily_returns, w, alpha, portfolio_value, window_size)

    # VaR prévue J-1 : on décale d'un jour pour comparer avec la perte réelle de J
    var_forecast = rolling_historical_var.shift(1).dropna()
    pnl_aligned  = actual_pnl.loc[var_forecast.index]

    # Violation : la perte réelle dépasse la VaR prévue
    is_violation    = pnl_aligned < -var_forecast
    n_violations    = int(is_violation.sum())
    violation_rate  = n_violations / len(var_forecast)
    expected_rate   = 1 - alpha

    print(f"\nBacktesting VaR {alpha*100:.0f}% (fenêtre {window_size}j)")
    print(f"Période         : {var_forecast.index[0].date()} → {var_forecast.index[-1].date()}")
    print(f"Jours testés    : {len(var_forecast)}")
    print(f"Violations      : {n_violations}")
    print(f"Taux observé    : {violation_rate:.2%}  (attendu : {expected_rate:.2%})")

    # --- Graphique P&L réel vs VaR prévue avec violations ---
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(pnl_aligned.index, pnl_aligned.values,
            color='steelblue', linewidth=0.8, label='P&L réel (J)')
    ax.plot(var_forecast.index, -var_forecast.values,
            color='orange', linewidth=1.5, linestyle='--', label=f'−VaR prévue J−1 ({alpha*100:.0f}%)')

    violation_dates = pnl_aligned.index[is_violation]
    ax.scatter(violation_dates, pnl_aligned.loc[violation_dates],
               color='red', zorder=5, s=20, label=f'Violations ({n_violations})')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle=':')
    ax.set_title(f'Backtesting VaR historique {alpha*100:.0f}% — P&L réel vs VaR prévue (J−1)', fontsize=13)
    ax.set_xlabel('Date')
    ax.set_ylabel('P&L ($)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'reports/backtest_var_{int(alpha*100)}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Test de Kupiec : LR = -2 * (L0 - L1), distribué en chi2(1) sous H0
    n = len(var_forecast)
    x = n_violations
    p  = expected_rate
    ph = violation_rate
    L0 = x * np.log(p)       + (n - x) * np.log(1 - p)
    L1 = x * np.log(ph)      + (n - x) * np.log(1 - ph)
    LR = -2 * (L0 - L1)

    p_value = 1 - chi2.cdf(LR, df=1)
    print(f"Test de Kupiec : LR={LR:.2f}, p-value={p_value:.4f}")