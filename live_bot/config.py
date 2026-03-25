# Exness / MT5 Configuration

# Your Exness Account credentials
MT5_LOGIN = 262086474  # Replace with your account number
MT5_PASSWORD = "Apz-22-2414"
MT5_SERVER = "Exness-MT5Trial16"  # Replace with your server name (e.g., Exness-MT5Real, Exness-Trial, etc.)

# Trading Settings
SYMBOL = "XAUUSDm"  # Gold vs USD (using 'm' suffix for Standard account)
TIMEFRAME_STR = "15m"  # "1m", "15m", "30m" matching the model
VOLUME = 0.01  # Lot size for trades
DEVIATION = 20  # Deviation in points
MAGIC_NUMBER = 234000
STOP_LOSS_POINTS = 500  # 500 points (50 pips) - Adjust as needed or use logic
TAKE_PROFIT_POINTS = 1000 # 1000 points (100 pips)

# Model Path (relative to main.py or absolute)
MODEL_PATH = "../trading_model_15m.pkl"
