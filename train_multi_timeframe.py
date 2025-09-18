#!/usr/bin/env python3
"""
Train XGBoost models for multiple timeframes (1m, 15m, 30m)
Uses the existing model architecture and feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
from datetime import datetime

def calculate_technical_indicators(df):
    """Calculate technical indicators for the dataframe"""
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

def prepare_features(df, timeframe):
    """Prepare features for the given timeframe"""
    print(f"Processing {timeframe} data with {len(df)} rows...")

    # Calculate technical indicators
    df = calculate_technical_indicators(df)

    # Detect FVGs
    fvgs = detect_fvg(df)
    df['FVG_Size'] = 0.0
    df['FVG_Type'] = 0  # 0 for no FVG, 1 for bullish

    for fvg in fvgs:
        df.at[df.index[fvg['index']], 'FVG_Size'] = fvg['size']
        df.at[df.index[fvg['index']], 'FVG_Type'] = 1

    # Simple Order Block detection (simplified)
    df['OB_Type'] = 0  # 0 for no OB, 1 for bullish OB

    # Lag features
    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag2'] = df['Close'].shift(2)
    df['Close_lag3'] = df['Close'].shift(3)

    # Drop NaN values
    df = df.dropna()

    # Create target (5-period ahead prediction)
    prediction_horizon = 5
    df['Target'] = (df['Close'].shift(-prediction_horizon) > df['Close']).astype(int)

    # Drop the last rows where target is NaN
    df = df.dropna()

    # Select features
    feature_cols = [
        'Close', 'High', 'Low', 'Open', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_upper', 'BB_middle', 'BB_lower', 'FVG_Size', 'FVG_Type',
        'OB_Type', 'Close_lag1', 'Close_lag2', 'Close_lag3'
    ]

    X = df[feature_cols]
    y = df['Target']

    print(f"Prepared {len(X)} samples with {len(feature_cols)} features")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    return X, y

def train_model(X, y, timeframe):
    """Train XGBoost model for given timeframe"""
    print(f"\n=== Training {timeframe} Model ===")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate class weights
    scale_pos_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])
    print(f"Scale positive weight: {scale_pos_weight:.3f}")

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'scale_pos_weight': [scale_pos_weight]
    }

    # Train model
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(".4f")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 5 Features:")
    print(feature_importance.head())

    return best_model, accuracy, feature_importance

def main():
    """Main training function for multiple timeframes"""
    timeframes = ['1m', '15m', '30m']
    results = {}

    for timeframe in timeframes:
        data_file = f"{timeframe}_data.csv"

        if not os.path.exists(data_file):
            print(f"❌ Data file {data_file} not found, skipping {timeframe}")
            continue

        # Load data
        try:
            # Read CSV: skip first 3 rows (headers and ticker info), set column names manually
            df = pd.read_csv(data_file, skiprows=3, index_col=0, parse_dates=True, header=None)
            df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
            df.index.name = 'Datetime'

            print(f"\n📊 Loaded {timeframe} data: {df.shape}")

        except Exception as e:
            print(f"❌ Error loading {data_file}: {e}")
            continue

        # Prepare features
        try:
            X, y = prepare_features(df, timeframe)
        except Exception as e:
            print(f"❌ Error preparing features for {timeframe}: {e}")
            continue

        # Train model
        try:
            model, accuracy, feature_importance = train_model(X, y, timeframe)

            # Save model
            model_filename = f"trading_model_{timeframe}.pkl"
            joblib.dump(model, model_filename)
            print(f"✅ Model saved as {model_filename}")

            # Save feature importance
            feature_importance.to_csv(f"feature_importance_{timeframe}.csv", index=False)

            results[timeframe] = {
                'accuracy': accuracy,
                'model_file': model_filename,
                'samples': len(X)
            }

        except Exception as e:
            print(f"❌ Error training model for {timeframe}: {e}")
            continue

    # Summary
    print("\n" + "="*50)
    print("MULTI-TIMEFRAME TRAINING SUMMARY")
    print("="*50)

    for timeframe, result in results.items():
        print(f"{timeframe}:")
        print(".4f")
        print(f"  Samples: {result['samples']}")
        print(f"  Model: {result['model_file']}")
        print()

    print("Next steps:")
    print("1. Run backtests for each timeframe model")
    print("2. Compare performance across timeframes")
    print("3. Select best performing model for deployment")

if __name__ == "__main__":
    main()