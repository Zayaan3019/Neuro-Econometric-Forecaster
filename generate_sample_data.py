"""
Generate sample S&P 500 data for testing when Yahoo Finance is unavailable.
This creates realistic OHLCV data based on historical patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sp500_data(start_date='2015-01-01', end_date='2024-12-31', initial_price=2000):
    """
    Generate realistic S&P 500 OHLCV data.
    
    Simulates trend, volatility, and realistic price movements.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business days (trading days)
    date_range = pd.date_range(start=start, end=end, freq='B')
    n_days = len(date_range)
    
    print(f"Generating {n_days} days of sample data from {start_date} to {end_date}...")
    
    # Initialize random seed for reproducibility
    np.random.seed(42)
    
    # Generate realistic returns (S&P 500 historical: ~10% annual return, ~15% annual volatility)
    daily_return_mean = 0.10 / 252  # 10% annual / 252 trading days
    daily_return_std = 0.15 / np.sqrt(252)  # 15% annual volatility
    
    # Add regime changes (bull/bear markets)
    returns = []
    for i in range(n_days):
        # Add some regime switching
        if i < n_days * 0.3:  # First 30%: bull market
            mean_adj = daily_return_mean * 1.5
            vol_adj = daily_return_std * 0.8
        elif i < n_days * 0.35:  # Brief correction
            mean_adj = daily_return_mean * -2
            vol_adj = daily_return_std * 2
        elif i < n_days * 0.7:  # Recovery and growth
            mean_adj = daily_return_mean * 1.2
            vol_adj = daily_return_std
        elif i < n_days * 0.75:  # COVID crash (2020)
            mean_adj = daily_return_mean * -3
            vol_adj = daily_return_std * 3
        else:  # Post-COVID recovery
            mean_adj = daily_return_mean * 1.8
            vol_adj = daily_return_std * 1.2
        
        daily_return = np.random.normal(mean_adj, vol_adj)
        returns.append(daily_return)
    
    # Generate close prices from returns
    returns = np.array(returns)
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    data = []
    for i, (date, close) in enumerate(zip(date_range, close_prices)):
        # Intraday volatility (typically 1-2%)
        intraday_vol = np.abs(np.random.normal(0, 0.01))
        
        # Open price (slight gap from previous close)
        if i == 0:
            open_price = close * (1 + np.random.normal(0, 0.002))
        else:
            open_price = close_prices[i-1] * (1 + np.random.normal(0, 0.003))
        
        # High and Low based on close and open
        high = max(open_price, close) * (1 + intraday_vol)
        low = min(open_price, close) * (1 - intraday_vol)
        
        # Ensure High >= Low and contains Open and Close
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Volume (realistic S&P 500 volume in billions)
        base_volume = 3.5e9  # ~3.5 billion shares
        volume_multiplier = np.random.lognormal(0, 0.3)  # Log-normal distribution
        # Higher volume on volatile days
        volatility_factor = 1 + abs(returns[i]) * 50
        volume = int(base_volume * volume_multiplier * volatility_factor)
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    
    print(f"\nSample data statistics:")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"  Trading days: {len(df)}")
    print(f"  Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
    print(f"  Final price: ${df['Close'].iloc[-1]:.2f}")
    print(f"  Total return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.1f}%")
    print(f"  Annual return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) ** (252/len(df)) - 1) * 100:.1f}%")
    
    return df


if __name__ == "__main__":
    # Generate S&P 500 data
    df = generate_sp500_data(
        start_date='2015-01-01',
        end_date='2024-12-31',
        initial_price=2058.20  # S&P 500 price on 2015-01-01
    )
    
    # Save to CSV
    output_path = 'data/GSPC_ohlcv.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Sample data saved to: {output_path}")
    print("\nYou can now run train_model.py with this local data!")
