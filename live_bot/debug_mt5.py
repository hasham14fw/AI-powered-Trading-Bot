import MetaTrader5 as mt5
import pandas as pd
import config

def check_symbols():
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return

    # Check connection
    authorized = mt5.login(config.MT5_LOGIN, password=config.MT5_PASSWORD, server=config.MT5_SERVER)
    if not authorized:
        print("failed to connect at account #{}, error code: {}".format(config.MT5_LOGIN, mt5.last_error()))
        mt5.shutdown()
        return

    print(f"Connected to {config.MT5_SERVER}")

    # 1. Search for XAUUSD variations
    print("\nSearching for XAUUSD symbols...")
    symbols = mt5.symbols_get()
    found_any = False
    for s in symbols:
        if "XAU" in s.name or "GOLD" in s.name:
            print(f"Found symbol: {s.name}, Path: {s.path}, Visible: {s.visible}")
            found_any = True
            
            # Identify if this is the likely candidate
            if "XAUUSD" in s.name:
                # Try to select it
                if not s.visible:
                    print(f"Attempting to select {s.name}...")
                    if mt5.symbol_select(s.name, True):
                        print(f"Successfully selected {s.name}")
                    else:
                        print(f"Failed to select {s.name}")
    
    if not found_any:
        print("No XAU or GOLD symbols found!")

    # 2. Test fetching data for config symbol
    print(f"\nTesting data fetch for configured symbol: '{config.SYMBOL}'")
    rates = mt5.copy_rates_from_pos(config.SYMBOL, mt5.TIMEFRAME_M15, 0, 10)
    if rates is None:
        print(f"❌ Failed to get rates for {config.SYMBOL}. Error code: {mt5.last_error()}")
    else:
        print(f"✅ Successfully fetched {len(rates)} candles for {config.SYMBOL}")
        print(rates[:1])

    mt5.shutdown()

if __name__ == "__main__":
    check_symbols()
