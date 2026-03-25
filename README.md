# AI-Powered XAUUSD Trading Bot Suite

A complete machine learning solution for institutional-grade Gold (XAUUSD) trading using **Smart Money Concepts (SMC)**. This repository includes everything from data fetching and model training to real-time automated execution on MetaTrader 5.

---

## Repository Structure
*   **/live_bot**: Contains the production-ready script for real-time trading on Exness/MT5.
*   **trading_model_*.pkl**: Pre-trained machine learning models (1m, 15m, 30m timeframes).
*   **train_multi_timeframe.py**: Script to train the AI models using historical data.
*   **fetch_data.py**: Tool for pulling historical OHLC data from MT5.
*   **Datasets**: Contains CSV files for training and backtesting signals.
*   **Whitepapers**: Technical documentation and research papers explaining the SMC AI logic.

---

## AI & SMC Strategy
This project applies **Machine Learning (XGBoost/RandomForest)** to institutional trading patterns:
*   **Fair Value Gaps (FVG)**: Identifies imbalances where price is likely to return.
*   **Order Blocks (OB)**: Locates areas of high institutional activity.
*   **Technical Fusion**: Combines SMC with RSI, MACD, and Bollinger Bands for high-confidence filters (>80% accuracy threshold).

---

## Quick Start (Live Bot)
To start trading immediately:
1.  Navigate to the `live_bot` directory.
2.  Install requirements: `pip install -r requirements.txt`.
3.  Configure `config.py` with your MT5 credentials.
4.  Run `python main.py`.

*For detailed instructions on the Live Bot, see [live_bot/README.md](live_bot/README.md).*

---

## Backtesting & Training
You can retrain the models or test strategies using the provided scripts:
```bash
# Train models across multiple timeframes (1m, 15m, 30m)
python train_multi_timeframe.py
```

---

## Documentation
The project includes several whitepapers and technical guides (available in `.md`, `.docx`, and `.tex` formats) that dive deep into the mathematical and strategic foundations of the bot.

---

## Disclaimer
Trading Forex and Gold involves significant risk. This project is for **educational and research purposes**. Use it at your own risk. Past performance does not guarantee future results.

---

## Author
**Hasham** - [GitHub Profile](https://github.com/hasham14fw)