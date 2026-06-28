# Value at Risk (VaR) — Market Risk Analysis

Python toolkit implementing several VaR methodologies on a multi-asset equity portfolio, with backtesting and historical stress testing.

---

## Portfolio

| Ticker | Company               | Weight |
|--------|-----------------------|--------|
| AAPL   | Apple Inc.            | 25%    |
| MSFT   | Microsoft Corp.       | 20%    |
| JPM    | JPMorgan Chase        | 20%    |
| XOM    | ExxonMobil            | 20%    |
| JNJ    | Johnson & Johnson     | 15%    |

- **Portfolio value:** $1,000,000  
- **Data period:** 2019-01-01 → 2024-01-01 (via Yahoo Finance)  
- **Returns:** Log-returns on adjusted closing prices

---

## Implemented Methods

### VaR Models (`src/var_models.py`)

| Method | Description |
|--------|-------------|
| **Historical VaR** | Empirical percentile of past portfolio P&L |
| **Rolling VaR** | Historical VaR over a rolling window (50, 100, 252 days) |
| **CVaR / Expected Shortfall** | Average loss beyond the VaR threshold |
| **Parametric VaR** | Normal distribution assumption with portfolio mean and covariance |
| **Monte Carlo VaR** | Multivariate normal simulation (10,000 draws by default) |

All methods support confidence levels α = 95% and α = 99%.

### Backtesting (`src/backtest.py`)

- Compares actual daily P&L against the previous day's rolling VaR forecast  
- Reports the number of violations and observed rate vs. expected rate  
- **Kupiec test (likelihood ratio)** to statistically validate model coverage

### Stress Testing (`src/stress_test.py`)

Three historical crisis scenarios simulated on the $1M portfolio:

| Scenario | Assumptions |
|----------|-------------|
| **Lehman Crisis 2008** | −5%/day for 10 days, inter-day correlation +0.30 |
| **COVID-19 2020** | One-shot shock of −30%, then volatility ×3 for 20 days |
| **Rate Hike 2022** | Tech −25%, Value −10%, Energy +15% |

---

## Project Structure

```
VaR/
├── src/
│   ├── data_loader.py      # Yahoo Finance download, log-returns, EDA charts
│   ├── var_models.py       # All VaR/CVaR methods + comparison chart
│   ├── backtest.py         # Rolling VaR backtesting + Kupiec test
│   ├── stress_test.py      # Simulation of the three crisis scenarios
│   ├── plot_skewness.py    # Visual illustration of skewness (3 distributions)
│   └── plot_kurtosis.py    # Visual illustration of kurtosis / fat tails
├── reports/                # Auto-generated charts (PNG)
├── requirements.txt
└── Compte_Rendu_VaR.pdf    # Project report
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`, `yfinance`, `jupyterlab`

---

## Usage

Each module can be run independently:

```bash
# 1. Exploratory analysis (prices, returns, QQ-plots, Jarque-Bera test)
python src/data_loader.py

# 2. Compute all VaR models and generate comparison charts
python src/var_models.py

# 3. Rolling VaR backtesting (99%) with Kupiec test
python src/backtest.py

# 4. Stress test scenario simulation
python src/stress_test.py

# 5. Educational illustrations (skewness and kurtosis)
python src/plot_skewness.py
python src/plot_kurtosis.py
```

All charts are automatically saved to `reports/`.
