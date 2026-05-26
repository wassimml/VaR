import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd 

def load_data(ticker, start_date, end_date):
    """
    Load historical stock data from Yahoo Finance.

    Parameters:
    ticker (str): The stock ticker symbol.
    start_date (str): The start date for the data in 'YYYY-MM-DD' format.
    end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
    pd.DataFrame: A DataFrame containing the historical stock data.
    """

    try:
        data = yf.download(ticker, start=start_date, end=end_date,auto_adjust=False)
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None
    

def log_daily_returns(data):
    """
    Calculate and log daily returns from the stock data.

    Parameters:
    data (pd.DataFrame): A DataFrame containing the historical stock data.

    Returns:
    pd.Series: A Series containing the daily returns.
    """
    try:
        daily_returns = np.log(data['Adj Close'].squeeze().pct_change().dropna() + 1)
        return daily_returns
    except Exception as e:
        print(f"Error calculating daily returns: {e}")
        return None

if __name__ == "__main__":
    # 1 ) Load data for multiple tickers

    tickers_list = {"AAPL" : None, "MSFT": None, "JPM": None, "XOM": None, "JNJ": None}
    start_date = "2019-01-01"
    end_date = "2024-01-01"

    for ticker in tickers_list:
        stock_data = load_data(ticker, start_date, end_date)
        tickers_list[ticker] = stock_data
    
    print("\n\nData loaded\n\n")
    for ticker, data in tickers_list.items():
        print(data.head(),"\n")

    # 1) bis) Price plot for each ticker
    for ticker, data in tickers_list.items():
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Adj Close'], label='Adjusted Close Price')
        plt.title(f"Adjusted Close Price for {ticker} (2019-2024)")
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid()
        plt.savefig(f"reports/{ticker}_adjusted_close_price_Q1.png")
        plt.show()

    # 1) ter) Correlation matrix of adjusted close prices
    adj_close_prices = pd.DataFrame({ticker: data['Adj Close'].squeeze() for ticker, data in tickers_list.items()})
    correlation_matrix = adj_close_prices.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label='Correlation Coefficient')
    ax.set_title('Correlation Matrix of Adjusted Close Prices')
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Ticker')
    ax.set_xticks(range(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns, rotation=45)
    ax.set_yticks(range(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)
    for i in range(len(correlation_matrix.index)):
        for j in range(len(correlation_matrix.columns)):
            val = correlation_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=10, color='black' if abs(val) < 0.7 else 'white')
    plt.tight_layout()
    plt.savefig(f"reports/correlation_matrix_Q1.png")
    plt.show()

    # 2) Calculate and log daily returns for each ticker

    daily_returns_dict = {}
    for ticker, data in tickers_list.items():
        daily_returns = log_daily_returns(data)
        daily_returns_dict[ticker] = daily_returns 
    
    print("\n\nDaily returns calculated\n\n")

    for ticker, returns in daily_returns_dict.items():
        print(f"Daily returns for {ticker}:\n{returns.head()}\n")

    # 2) bis) Correlation matrix of daily returns
    daily_returns_df = pd.DataFrame(daily_returns_dict)
    returns_correlation_matrix = daily_returns_df.corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(returns_correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label='Correlation Coefficient')
    ax.set_title('Correlation Matrix of Daily Returns')
    ax.set_xlabel('Ticker')
    ax.set_ylabel('Ticker')
    ax.set_xticks(range(len(returns_correlation_matrix.columns)))
    ax.set_xticklabels(returns_correlation_matrix.columns, rotation=45)
    ax.set_yticks(range(len(returns_correlation_matrix.index)))
    ax.set_yticklabels(returns_correlation_matrix.index)
    for i in range(len(returns_correlation_matrix.index)):
        for j in range(len(returns_correlation_matrix.columns)):
            val = returns_correlation_matrix.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha='center', va='center',
                    fontsize=10, color='black' if abs(val) < 0.7 else 'white')
    plt.tight_layout()
    plt.savefig(f"reports/returns_correlation_matrix_Q1.png")
    plt.show()

    # 3) Descriptive statistics for daily returns

    print("\n\nDescriptive statistics for daily returns\n\n")
    for ticker, returns in daily_returns_dict.items():
        print(f"Descriptive statistics for {ticker}:\n{returns.describe()}")
        print(f"Skewness for {ticker}: \n{returns.skew()}")
        print(f"Kurtosis for {ticker}: \n{returns.kurtosis()}\n")

    # 4) Histogram of daily returns for each ticker, with normal distribution overlay

    for ticker, returns in daily_returns_dict.items():
        returns = returns.squeeze()
        plt.figure(figsize=(10, 6))
        plt.hist(returns, bins=50, density=True, alpha=0.6, color='g')

        # Overlay normal distribution
        mean = float(returns.mean())
        std = float(returns.std())
        x = np.linspace(float(returns.min()), float(returns.max()), 100)
        p = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        plt.plot(x, p, 'k', linewidth=2)

        skewness = float(returns.skew())
        kurt     = float(returns.kurtosis())
        stats_text = (
            f"Mean:      {mean:.4f}\n"
            f"Std:       {std:.4f}\n"
            f"Min:       {float(returns.min()):.4f}\n"
            f"Max:       {float(returns.max()):.4f}\n"
            f"Skew:      {skewness:.4f}\n"
            f"Kurtosis:  {kurt:.4f}"
        )
        plt.gca().text(0.98, 0.97, stats_text,transform=plt.gca().transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )
        plt.legend(['Normal Distribution', 'Daily Returns'])
        
        title = f"Histogram of Daily Returns for {ticker} with Normal Distribution Overlay"
        plt.title(title)
        plt.xlabel('Daily Return')
        plt.ylabel('Density')
        plt.grid()
        plt.savefig(f"reports/{ticker}_daily_returns_histogram_Q1.png")
        plt.show()
    
    # QQ plot for each ticker

    for ticker, returns in daily_returns_dict.items():
        returns = returns.squeeze()
        plt.figure(figsize=(6, 6))
        stats.probplot(returns, dist="norm", plot=plt)
        plt.title(f"QQ Plot of Daily Returns for {ticker}")
        plt.grid()
        plt.savefig(f"reports/{ticker}_daily_returns_qqplot_Q1.png")
        plt.show()
    

    # 5) Jarque-Bera test for normality
    print("\n\nJarque-Bera test for normality\n\n")
    for ticker, returns in daily_returns_dict.items():
        returns = returns.squeeze()
        jb_stat, jb_pvalue = stats.jarque_bera(returns)
        print(f"Jarque-Bera test for {ticker}:")
        print(f"JB Statistic: {jb_stat:.4f}, p-value: {jb_pvalue:.4f}\n")

