import time
import MetaTrader5 as mt5
import pandas as pd
import joblib
import os
import sys
from datetime import datetime
import config
from utils import prepare_features

def initialize_mt5():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    authorized = mt5.login(config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER)
    if not authorized:
        print("failed to connect at account #{}, error code: {}".format(config.MT5_LOGIN, mt5.last_error()))
        mt5.shutdown()
        return False
    
    print(f"Connected to {config.MT5_SERVER} with account {config.MT5_LOGIN}")
    return True

def get_timeframe_constant(tf_str):
    if tf_str == "1m":
        return mt5.TIMEFRAME_M1
    elif tf_str == "15m":
        return mt5.TIMEFRAME_M15
    elif tf_str == "30m":
        return mt5.TIMEFRAME_M30
    elif tf_str == "1h":
        return mt5.TIMEFRAME_H1
    elif tf_str == "1d":
        return mt5.TIMEFRAME_D1
    else:
        return mt5.TIMEFRAME_M15

def get_data(symbol, timeframe, n_candles=100):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n_candles)
    if rates is None:
        print(f"Failed to get rates for {symbol}")
        return None
    
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Rename columns to match what utils/model expects
    # MT5 returns: time, open, high, low, close, tick_volume, spread, real_volume
    # Model expects: Close, High, Low, Open (and others calculated from these)
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'tick_volume': 'Volume' 
    }, inplace=True)
    
    return df

def execute_trade(action, symbol, lot, sl_points, tp_points, deviations):
    # Action: 'buy' or 'sell'
    point = mt5.symbol_info(symbol).point
    price = mt5.symbol_info_tick(symbol).ask if action == 'buy' else mt5.symbol_info_tick(symbol).bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == 'buy' else mt5.ORDER_TYPE_SELL,
        "price": price,
        "deviation": deviations,
        "magic": config.MAGIC_NUMBER,
        "comment": "python script open",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    # Calculate SL/TP
    if action == 'buy':
        request["sl"] = price - sl_points * point
        request["tp"] = price + tp_points * point
    else:
        request["sl"] = price + sl_points * point
        request["tp"] = price - tp_points * point
        
    result = mt5.order_send(request)
    print(f"Order send result: {result}")
    return result

def close_position(position):
    tick = mt5.symbol_info_tick(position.symbol)
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY,
        "price": tick.bid if position.type == mt5.ORDER_TYPE_BUY else tick.ask,
        "deviation": config.DEVIATION,
        "magic": config.MAGIC_NUMBER,
        "comment": "python script close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    print(f"Close position result: {result}")
    return result

def main():
    if not initialize_mt5():
        return

    # Load Model
    model_path = os.path.join(os.path.dirname(__file__), config.MODEL_PATH)
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
    
    try:
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    tf_constant = get_timeframe_constant(config.TIMEFRAME_STR)
    
    print("Bot started taking action...")
    
    while True:
        # 1. Get Data
        df = get_data(config.SYMBOL, tf_constant, n_candles=100)
        if df is None:
            time.sleep(10)
            continue
            
        # 2. Prepare Features
        try:
            features_df = prepare_features(df)
            # Take the last complete row. 
            # Note: The *current* candle might be forming. 
            # Usually we trade on 'close' of the previous candle.
            # If we take 'iloc[-1]', it's the current forming candle.
            # If we take 'iloc[-2]', it's the last closed candle.
            # Strategy choice: usually close of bar.
            # Let's check timestamp.
            
            last_features = features_df.iloc[[-1]] # Take as dataframe to keep shape
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            time.sleep(10)
            continue
            
        # 3. Predict
        try:
            # USE LAST CLOSED CANDLE (iloc[-2]) FOR STABILITY
            # iloc[-1] is the current forming candle, which causes signals to flicker.
            current_time = df['time'].iloc[-1]
            last_features = features_df.iloc[[-2]] 
            
            prediction_prob = model.predict_proba(last_features)[0][1] # Probability of Class 1 (Up)
            
            # Log only periodically or on new bar
            if 'last_log_time' not in locals() or current_time != locals().get('last_bar_time'):
                 print(f"[{datetime.now().strftime('%H:%M:%S')}] New Bar: {current_time} | Stable P(UP): {prediction_prob:.4f}")
                 last_bar_time = current_time
            
        except Exception as e:
            print(f"Error predicting: {e}")
            time.sleep(10)
            continue
            
        # 4. Trading Logic
        # Check current positions
        positions = mt5.positions_get(symbol=config.SYMBOL)
        current_position = None
        if positions:
            for pos in positions:
                if pos.magic == config.MAGIC_NUMBER:
                    current_position = pos
                    break
        
        # STRICTER THRESHOLDS (Higher Accuracy)
        BUY_THRESHOLD = 0.82
        SELL_THRESHOLD = 0.18
        
        if current_position is None:
            if prediction_prob > BUY_THRESHOLD:
                print(f">>> Strong BUY Signal detected ({prediction_prob:.4f}) on closed candle. Executing trade...")
                res = execute_trade('buy', config.SYMBOL, config.VOLUME, config.STOP_LOSS_POINTS, config.TAKE_PROFIT_POINTS, config.DEVIATION)
                if res.retcode != mt5.TRADE_RETCODE_DONE:
                     print(f"Trade failed! Error: {res.comment}")
                else:
                     print("Trade executed successfully.")
                     time.sleep(60) # Wait a bit
            elif prediction_prob < SELL_THRESHOLD:
                print(f">>> Strong SELL Signal detected ({prediction_prob:.4f}) on closed candle. Executing trade...")
                res = execute_trade('sell', config.SYMBOL, config.VOLUME, config.STOP_LOSS_POINTS, config.TAKE_PROFIT_POINTS, config.DEVIATION)
                if res.retcode != mt5.TRADE_RETCODE_DONE:
                     print(f"Trade failed! Error: {res.comment}")
                else:
                     print("Trade executed successfully.")
                     time.sleep(60)
            # Else: Do nothing, just wait.
        else:
            # We have a position. Only close if signal REVERSES significantly.
            # Don't close just because probability dropped slightly below threshold.
            
            # Using slightly looser exit thresholds to avoid "seconds" trade, 
            # but strong enough to save verify reversal.
            EXIT_BUY_THRESHOLD = 0.35  # Close Buy only if prob drops below 0.35 (Bearish lean)
            EXIT_SELL_THRESHOLD = 0.65 # Close Sell only if prob rises above 0.65 (Bullish lean)
            
            if current_position.type == mt5.ORDER_TYPE_BUY:
                if prediction_prob < EXIT_BUY_THRESHOLD:
                    print(f"Signal Reversed to Bearish ({prediction_prob:.4f}). Closing Buy.")
                    close_position(current_position)
            elif current_position.type == mt5.ORDER_TYPE_SELL:
                if prediction_prob > EXIT_SELL_THRESHOLD:
                    print(f"Signal Reversed to Bullish ({prediction_prob:.4f}). Closing Sell.")
                    close_position(current_position)
        
        # Sleep to avoid spamming.
        time.sleep(5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopping bot...")
        mt5.shutdown()
