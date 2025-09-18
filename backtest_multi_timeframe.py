#!/usr/bin/env python3
"""
Backtest multi-timeframe models using Backtrader
Tests the trained XGBoost models on different timeframes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import backtrader as bt
import joblib
import os
from sklearn.metrics import classification_report

class SMCStrategy(bt.Strategy):
    """Backtrader strategy using trained XGBoost model"""

    params = (
        ('model', None),
        ('timeframe', '15m'),
        ('stake_size', 1000),  # Fixed stake size in USD
    )

    def __init__(self):
        # Load the trained model
        self.model = self.params.model
        self.timeframe = self.params.timeframe

        # Technical indicators
        self.sma_20 = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.sma_50 = bt.indicators.SimpleMovingAverage(self.data.close, period=50)
        self.ema_12 = bt.indicators.ExponentialMovingAverage(self.data.close, period=12)
        self.ema_26 = bt.indicators.ExponentialMovingAverage(self.data.close, period=26)
        self.rsi = bt.indicators.RSI(self.data.close, period=14)

        # MACD
        self.macd = bt.indicators.MACD(self.data.close)
        self.macd_signal = bt.indicators.MACD(self.data.close, period_me1=9, period_me2=26, period_signal=9).signal

        # Bollinger Bands
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=20)

        # Lag features (use direct indexing)
        # self.close_lag1 = bt.indicators.Lag(self.data.close, period=1)
        # self.close_lag2 = bt.indicators.Lag(self.data.close, period=2)
        # self.close_lag3 = bt.indicators.Lag(self.data.close, period=3)

        # Track positions
        self.order = None
        self.prediction = None

    def next(self):
        if self.order:
            return

        # Prepare features for prediction
        features = self.prepare_features()

        if features is not None:
            # Make prediction
            self.prediction = self.model.predict_proba(features)[0][1]  # Probability of upward movement

            # Trading logic
            if not self.position:  # No position
                if self.prediction > 0.6:  # Strong bullish signal
                    stake = self.params.stake_size / self.data.close[0]
                    self.order = self.buy(size=stake)
                elif self.prediction < 0.4:  # Strong bearish signal
                    stake = self.params.stake_size / self.data.close[0]
                    self.order = self.sell(size=stake)
            else:  # Have position
                if self.position.size > 0 and self.prediction < 0.4:  # Close long if bearish
                    self.order = self.close()
                elif self.position.size < 0 and self.prediction > 0.6:  # Close short if bullish
                    self.order = self.close()

    def prepare_features(self):
        """Prepare features for model prediction"""
        try:
            features_dict = {
                'Close': float(self.data.close[0]),
                'High': float(self.data.high[0]),
                'Low': float(self.data.low[0]),
                'Open': float(self.data.open[0]),
                'SMA_20': float(self.sma_20[0]),
                'SMA_50': float(self.sma_50[0]),
                'EMA_12': float(self.ema_12[0]),
                'EMA_26': float(self.ema_26[0]),
                'RSI': float(self.rsi[0]),
                'MACD': float(self.macd.macd[0]),
                'MACD_signal': float(self.macd_signal[0]),
                'MACD_hist': float(self.macd.macd[0] - self.macd_signal[0]),
                'BB_upper': float(self.bbands.top[0]),
                'BB_middle': float(self.bbands.mid[0]),
                'BB_lower': float(self.bbands.bot[0]),
                'FVG_Size': 0.0,  # placeholder
                'FVG_Type': 0,    # placeholder
                'OB_Type': 0,     # placeholder
                'Close_lag1': float(self.data.close[-1] if len(self.data) > 1 else self.data.close[0]),
                'Close_lag2': float(self.data.close[-2] if len(self.data) > 2 else self.data.close[0]),
                'Close_lag3': float(self.data.close[-3] if len(self.data) > 3 else self.data.close[0]),
            }

            # Handle NaN values
            for key, value in features_dict.items():
                if np.isnan(value):
                    features_dict[key] = 0.0

            # Convert to DataFrame for model prediction
            features_df = pd.DataFrame([features_dict])
            return features_df

        except (IndexError, TypeError) as e:
            print(f"Error preparing features: {e}")
            return None

def load_data(timeframe):
    """Load data for backtesting"""
    data_file = f"{timeframe}_data.csv"

    if not os.path.exists(data_file):
        print(f"❌ Data file {data_file} not found")
        return None

    try:
        # Load data with proper parsing
        df = pd.read_csv(data_file, skiprows=3, index_col=0, parse_dates=True, header=None)
        df.columns = ['Close', 'High', 'Low', 'Open', 'Volume']
        df.index.name = 'Datetime'

        # Convert to backtrader format
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df['OpenInterest'] = 0  # Required by backtrader

        return df

    except Exception as e:
        print(f"❌ Error loading {data_file}: {e}")
        return None

def run_backtest(timeframe, model):
    """Run backtest for given timeframe and model"""
    print(f"\n=== Backtesting {timeframe} Model ===")

    # Load data
    df = load_data(timeframe)
    if df is None:
        return None

    # Create backtrader engine
    cerebro = bt.Cerebro()

    # Add strategy
    cerebro.addstrategy(SMCStrategy, model=model, timeframe=timeframe)

    # Add data feed
    data = bt.feeds.PandasData(dataname=df, openinterest=None)
    cerebro.adddata(data)

    # Set broker parameters
    cerebro.broker.setcash(10000.0)  # Starting capital
    cerebro.broker.setcommission(commission=0.001)  # 0.1% commission

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')

    # Run backtest
    print("Running backtest...")
    results = cerebro.run()
    strat = results[0]

    # Extract results
    final_value = cerebro.broker.getvalue()
    initial_value = 10000.0
    total_return = (final_value - initial_value) / initial_value * 100

    # Get analyzer results
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()

    # Print results
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    print(f"Total Trades: {trades.total.total if 'total' in trades and 'total' in trades.total else 0}")
    print(f"Winning Trades: {trades.won.total if 'won' in trades and 'total' in trades.won else 0}")
    print(f"Losing Trades: {trades.lost.total if 'lost' in trades and 'total' in trades.lost else 0}")

    if 'won' in trades and 'total' in trades.won and trades.won.total > 0:
        win_rate = trades.won.total / trades.total.total * 100
        print(".1f")

    return {
        'timeframe': timeframe,
        'initial_value': initial_value,
        'final_value': final_value,
        'total_return': total_return,
        'sharpe_ratio': sharpe.get('sharperatio', 0) if sharpe and 'sharperatio' in sharpe else 0,
        'max_drawdown': drawdown.max.drawdown if drawdown and hasattr(drawdown, 'max') else 0,
        'total_trades': trades.total.total if 'total' in trades and 'total' in trades.total else 0,
        'winning_trades': trades.won.total if 'won' in trades and 'total' in trades.won else 0,
        'losing_trades': trades.lost.total if 'lost' in trades and 'total' in trades.lost else 0,
    }

def main():
    """Main backtesting function"""
    timeframes = ['15m', '30m']  # Skip 1m for now due to data issues
    results = []

    for timeframe in timeframes:
        model_file = f"trading_model_{timeframe}.pkl"

        if not os.path.exists(model_file):
            print(f"❌ Model file {model_file} not found, skipping {timeframe}")
            continue

        try:
            # Load model
            model = joblib.load(model_file)
            print(f"✅ Loaded model: {model_file}")

            # Run backtest
            result = run_backtest(timeframe, model)
            if result:
                results.append(result)

        except Exception as e:
            print(f"❌ Error backtesting {timeframe}: {e}")
            continue

    # Summary
    print("\n" + "="*60)
    print("MULTI-TIMEFRAME BACKTEST SUMMARY")
    print("="*60)

    if results:
        for result in results:
            print(f"\n{result['timeframe'].upper()} Timeframe:")
            print(".2f")
            print(".2f")
            print(".2f")
            print(f"  Sharpe Ratio: {result['sharpe_ratio'] if result['sharpe_ratio'] is not None else 'N/A'}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"  Total Trades: {result['total_trades']}")
            print(f"  Winning Trades: {result['winning_trades']}")
            print(f"  Losing Trades: {result['losing_trades']}")

            if result['total_trades'] > 0:
                win_rate = result['winning_trades'] / result['total_trades'] * 100
                print(".1f")

        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv('backtest_multi_timeframe_results.csv', index=False)
        print("\n✅ Results saved to backtest_multi_timeframe_results.csv")
    else:
        print("❌ No successful backtests completed")

    print("\nNext steps:")
    print("1. Compare performance across timeframes")
    print("2. Optimize trading parameters (stake size, thresholds)")
    print("3. Add risk management features")
    print("4. Test on out-of-sample data")

if __name__ == "__main__":
    main()