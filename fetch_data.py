import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Define the symbol (using GC=F for gold futures as XAUUSD equivalent)
symbol = 'GC=F'

# Define date ranges
# 2000-2010
start_2000 = '2000-01-01'
end_2010 = '2010-12-31'

# 2010-2020 for hourly data (yfinance limits hourly data to ~2 years)
start_2010 = '2010-01-01'
end_2020 = '2020-12-31'

# Fetch daily data
print("Fetching daily data for 2000-2010...")
daily_2000_2010 = yf.download(symbol, start=start_2000, end=end_2010, interval='1d')

print("Fetching daily data for 2010-2020...")
daily_2010_2020 = yf.download(symbol, start=start_2010, end=end_2020, interval='1d')

# Combine daily data
daily_data = pd.concat([daily_2000_2010, daily_2010_2020])

# Flatten MultiIndex columns
daily_data.columns = daily_data.columns.droplevel(1)

print("Columns after flattening:", daily_data.columns)
print("Head:")
print(daily_data.head())

# Fetch hourly data for 2010-2020 (try longer range for forex)
print("Fetching hourly data for 2010-2020...")
hourly_data = yf.download(symbol, start=start_2010, end=end_2020, interval='1h')

# Fetch intraday data (limited to recent periods due to yfinance restrictions)
print("Fetching 30-minute data (last 60 days)...")
end_date = datetime.now()
start_date_30m = end_date - timedelta(days=59)  # 60 days max for 30m
m30_data = yf.download(symbol, start=start_date_30m.strftime('%Y-%m-%d'),
                       end=end_date.strftime('%Y-%m-%d'), interval='30m')

print("Fetching 15-minute data (last 60 days)...")
start_date_15m = end_date - timedelta(days=59)  # 60 days max for 15m
m15_data = yf.download(symbol, start=start_date_15m.strftime('%Y-%m-%d'),
                       end=end_date.strftime('%Y-%m-%d'), interval='15m')

print("Fetching 1-minute data (last 7 days)...")
start_date_1m = end_date - timedelta(days=6)  # 7 days max for 1m
m1_data = yf.download(symbol, start=start_date_1m.strftime('%Y-%m-%d'),
                      end=end_date.strftime('%Y-%m-%d'), interval='1m')

# Save to CSV files
daily_data.to_csv('daily_data.csv')
hourly_data.to_csv('hourly_data.csv')
m30_data.to_csv('30m_data.csv')
m15_data.to_csv('15m_data.csv')
m1_data.to_csv('1m_data.csv')

print("Data fetching completed.")
print(f"Daily data shape: {daily_data.shape}")
print(f"Hourly data shape: {hourly_data.shape}")
print(f"30m data shape: {m30_data.shape}")
print(f"15m data shape: {m15_data.shape}")
print(f"1m data shape: {m1_data.shape}")