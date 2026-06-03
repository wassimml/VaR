import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import pandas as pd

REPORTS_DIR = Path(__file__).parent.parent / "reports"


class DataLoader:
    def __init__(self, start_date: str, end_date: str):
        self.start_date = start_date
        self.end_date = end_date

    def load(self, ticker: str):
        try:
            return yf.download(ticker, start=self.start_date, end=self.end_date, auto_adjust=False)
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            return None

    @staticmethod
    def log_returns(data):
        try:
            return np.log(data['Adj Close'].squeeze().pct_change().dropna() + 1)
        except Exception as e:
            print(f"Error calculating daily returns: {e}")
            return None

    @staticmethod
    def _plot_correlation_matrix(matrix, title: str, path: Path):
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
        fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        ax.set_title(title)
        ax.set_xlabel('Ticker')
        ax.set_ylabel('Ticker')
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45)
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        for (i, j), val in np.ndenumerate(matrix.values):
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=10, color='black' if abs(val) < 0.7 else 'white')
        plt.tight_layout()
        plt.savefig(path)
        plt.show()


if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "JPM", "XOM", "JNJ"]
    loader = DataLoader(start_date="2019-01-01", end_date="2024-01-01")

    # 1) Load data
    tickers_data = {t: loader.load(t) for t in tickers}
    print("\n\nData loaded\n\n")
    for ticker, data in tickers_data.items():
        print(data.head(), "\n")

    # 1b) Price plots
    for ticker, data in tickers_data.items():
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Adj Close'], label='Adjusted Close Price')
        plt.title(f"Adjusted Close Price for {ticker} (2019-2024)")
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid()
        plt.savefig(REPORTS_DIR / f"{ticker}_adjusted_close_price_Q1.png")
        plt.show()

    # 1c) Correlation matrix of adjusted close prices
    adj_close = pd.DataFrame({t: d['Adj Close'].squeeze() for t, d in tickers_data.items()})
    DataLoader._plot_correlation_matrix(
        adj_close.corr(),
        'Correlation Matrix of Adjusted Close Prices',
        REPORTS_DIR / 'correlation_matrix_Q1.png',
    )

    # 2) Daily log-returns
    daily_returns_dict = {t: loader.log_returns(d) for t, d in tickers_data.items()}
    print("\n\nDaily returns calculated\n\n")
    for ticker, returns in daily_returns_dict.items():
        print(f"Daily returns for {ticker}:\n{returns.head()}\n")

    # 2b) Correlation matrix of daily returns
    DataLoader._plot_correlation_matrix(
        pd.DataFrame(daily_returns_dict).corr(),
        'Correlation Matrix of Daily Returns',
        REPORTS_DIR / 'returns_correlation_matrix_Q1.png',
    )

    # 3) Descriptive statistics
    print("\n\nDescriptive statistics for daily returns\n\n")
    for ticker, returns in daily_returns_dict.items():
        print(f"Descriptive statistics for {ticker}:\n{returns.describe()}")
        print(f"Skewness for {ticker}: \n{returns.skew()}")
        print(f"Kurtosis for {ticker}: \n{returns.kurtosis()}\n")

    # 4) Histograms with normal distribution overlay
    for ticker, returns in daily_returns_dict.items():
        returns = returns.squeeze()
        mean, std = float(returns.mean()), float(returns.std())
        x = np.linspace(float(returns.min()), float(returns.max()), 100)
        p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, density=True, alpha=0.6, color='g')
        plt.plot(x, p, 'k', linewidth=2)
        stats_text = (
            f"Mean:      {mean:.4f}\n"
            f"Std:       {std:.4f}\n"
            f"Min:       {float(returns.min()):.4f}\n"
            f"Max:       {float(returns.max()):.4f}\n"
            f"Skew:      {float(returns.skew()):.4f}\n"
            f"Kurtosis:  {float(returns.kurtosis()):.4f}"
        )
        plt.gca().text(0.98, 0.97, stats_text, transform=plt.gca().transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
        plt.legend(['Normal Distribution', 'Daily Returns'])
        plt.title(f"Histogram of Daily Returns for {ticker} with Normal Distribution Overlay")
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.grid()
        plt.savefig(REPORTS_DIR / f"{ticker}_daily_returns_histogram_Q1.png")
        plt.show()

    # QQ plots
    for ticker, returns in daily_returns_dict.items():
        returns = returns.squeeze()
        plt.figure(figsize=(6, 6))
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title(f"QQ Plot of Daily Returns for {ticker}")
        plt.grid()
        plt.savefig(REPORTS_DIR / f"{ticker}_daily_returns_qqplot_Q1.png")
        plt.show()

    # 5) Jarque-Bera test
    print("\n\nJarque-Bera test for normality\n\n")
    for ticker, returns in daily_returns_dict.items():
        returns = returns.squeeze()
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        print(f"Jarque-Bera test for {ticker}:")
        print(f"JB Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}\n")
