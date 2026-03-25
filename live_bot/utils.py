import pandas as pd
import numpy as np

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataframe"""
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std

    return df

def detect_fvg(df):
    """Detect Fair Value Gaps"""
    fvgs = []
    # We only need to detect FVG for the recent data points to populate features
    # But for a dataframe, we iterate.
    for i in range(1, len(df) - 1):
        current_low = df['Low'].iloc[i]
        prev_high = df['High'].iloc[i-1]
        next_high = df['High'].iloc[i+1]

        if current_low > prev_high and current_low > next_high:
            gap_size = current_low - max(prev_high, next_high)
            fvgs.append({
                'index': i,
                'size': gap_size,
                'type': 'bullish'
            })
    return fvgs

def prepare_features(df):
    """Prepare features for the given dataframe logic"""
    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Detect FVGs
    fvgs = detect_fvg(df)
    df['FVG_Size'] = 0.0
    df['FVG_Type'] = 0  # 0 for no FVG, 1 for bullish

    for fvg in fvgs:
        # Use idx to access the correct row if index is standard RangeIndex
        # Or mapped index if not. 
        # The loop in detect_fvg used integer position `i`.
        # So we should use iloc to set values or ensure index is aligned.
        # Safest is to use iloc for setting if we are sure of the position.
        if fvg['index'] < len(df):
            df.iloc[fvg['index'], df.columns.get_loc('FVG_Size')] = fvg['size']
            df.iloc[fvg['index'], df.columns.get_loc('FVG_Type')] = 1

    # Simple Order Block detection (simplified as per training script)
    df['OB_Type'] = 0  # 0 for no OB, 1 for bullish OB

    # Lag features
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag3'] = df['Close'].shift(3)

    # We do NOT dropna here because we might need the last row even if some previous are NaN (though for training we drop).
    # Ideally for the latest candle, we need previous data to be present.
    # If we fetch 100 candles, the last one should have all features valid.

    # Select features in specific order
    feature_cols = [
        'Close', 'High', 'Low', 'Open', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_upper', 'BB_middle', 'BB_lower', 'FVG_Size', 'FVG_Type',
        'OB_Type', 'Close_lag1', 'Close_lag2', 'Close_lag3'
    ]

    # Return the dataframe with only feature columns
    return df[feature_cols]
