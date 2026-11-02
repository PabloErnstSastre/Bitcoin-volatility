import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from arch import arch_model


def fetch_price(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download historical price data from Yahoo Finance.
    Returns a DataFrame with Date as index and a 'Close' column.
    """
    data = yf.download(symbol, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No data fetched for {symbol} between {start} and {end}")
    df = data[['Close']].copy()
    df.index = pd.to_datetime(df.index)
    return df


def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Calculate daily log returns from closing prices."""
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    returns.name = 'LogReturn'
    return returns


def annualized_volatility(returns: pd.Series) -> float:
    """Annualize the standard deviation of daily returns.
    Uses 365 trading days for Bitcoin (crypto trades every day).
    """
    daily_std = returns.std()
    return daily_std * np.sqrt(365)


def fit_garch(returns: pd.Series, p: int = 1, q: int = 1):
    """Fit a GARCH(p,q) model to the return series.
    Returns the fitted model.
    """
    am = arch_model(returns * 100, p=p, q=q, mean='Zero', vol='Garch', dist='Normal')
    res = am.fit(update_freq=5, disp='off')
    return res


def forecast_volatility(res, horizon: int = 30):
    """Forecast future volatility for given horizon (days)."""
    forecasts = res.forecast(horizon=horizon)
    # The forecasted variance for each day ahead
    variance = forecasts.variance.iloc[-1]
    # Convert variance to volatility (standard deviation) and annualize
    daily_vol = np.sqrt(variance) / np.sqrt(365)  # back to daily sigma
    annual_vol = daily_vol * np.sqrt(365)
    return annual_vol


def plot_results(df: pd.DataFrame, returns: pd.Series, annual_vol: float, forecast_vol: pd.Series, symbol: str, output_path: str):
    """Create a multi-panel plot with price, returns, historical volatility and forecasted volatility."""
    plt.style.use('seaborn-darkgrid')
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    # Price
    axes[0].plot(df.index, df['Close'], label='Close Price')
    axes[0].set_title(f'{symbol} Closing Price')
    axes[0].legend()

    # Returns
    axes[1].plot(returns.index, returns, label='Log Returns')
    axes[1].set_ylabel('Log Return')
    axes[1].legend()

    # Historical annualized volatility (rolling window)
    rolling_vol = returns.rolling(window=30).std() * np.sqrt(365)
    axes[2].plot(rolling_vol.index, rolling_vol, label='30‑day Rolling Volatility')
    axes[2].axhline(annual_vol, color='red', linestyle='--', label='Overall Annualized Vol')
    axes[2].set_ylabel('Volatility')
    axes[2].legend()

    # Forecasted volatility
    forecast_dates = pd.date_range(start=returns.index[-1] + pd.Timedelta(days=1), periods=len(forecast_vol))
    axes[3].plot(forecast_dates, forecast_vol, label='Forecasted Annualized Volatility')
    axes[3].set_ylabel('Volatility')
    axes[3].set_xlabel('Date')
    axes[3].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Bitcoin Volatility Analysis and Forecast')
    parser.add_argument('--symbol', default='BTC-USD', help='Ticker symbol (default: BTC-USD)')
    parser.add_argument('--start', default='2018-01-01', help='Start date YYYY-MM-DD')
    parser.add_argument('--end', default=datetime.today().strftime('%Y-%m-%d'), help='End date YYYY-MM-DD')
    parser.add_argument('--horizon', type=int, default=30, help='Forecast horizon in days')
    parser.add_argument('--output', default='results/volatility_plot.png', help='Path to save the plot')
    args = parser.parse_args()

    print('Fetching price data...')
    price_df = fetch_price(args.symbol, args.start, args.end)

    print('Computing returns...')
    returns = compute_returns(price_df)

    print('Calculating historical annualized volatility...')
    hist_vol = annualized_volatility(returns)
    print(f'Historical Annualized Volatility: {hist_vol:.2%}')

    print('Fitting GARCH model...')
    garch_res = fit_garch(returns)
    print(garch_res.summary())

    print(f'Forecasting volatility for next {args.horizon} days...')
    forecast_vol = forecast_volatility(garch_res, horizon=args.horizon)

    print('Generating plot...')
    plot_results(price_df, returns, hist_vol, forecast_vol, args.symbol, args.output)
    print(f'Plot saved to {args.output}')

if __name__ == '__main__':
    main()
