# 🤖 AI-Powered XAUUSD Trading Bot (SMC-v2)

An advanced automated trading bot for **XAUUSD (Gold)** on MetaTrader 5 (MT5), powered by Machine Learning and **Smart Money Concepts (SMC)**.

---

## 🌟 Introduction
This project is an AI-driven trading solution designed specifically for Gold (XAUUSD) traders using the Exness broker. It leverages pre-trained models to analyze market imbalances and execute trades with high precision.

Unlike traditional bots that rely on static indicators, this bot integrates **Smart Money Concepts (SMC)** features like Fair Value Gaps (FVG) and Order Blocks (OB) into its machine-learning decision process to identify institutional order flow.

## 🚀 Key Features
*   **AI-Model Integration**: Uses a pre-trained XGBoost/Scikit-learn model (`.pkl`) trained on historical XAUUSD data.
*   **SMC Logic**: Built-in detection for **Fair Value Gaps (FVG)** and **Order Blocks (OB)** to align with institutional trading patterns.
*   **MetaTrader 5 Integration**: Deep integration with MT5 via the Python API for seamless data fetching and order execution.
*   **Automated Risk Management**: Configurable Stop Loss (SL) and Take Profit (TP).
*   **Multi-Timeframe Support**: Primarily optimized for the **15M** timeframe but supports other intervals via configuration.
*   **Probability-Based Entry**: Trades are only executed when the model reaches a high-confidence threshold (e.g., >80% for Buy).

## 🛠️ Tech Stack
*   **Language**: Python 3.8+
*   **Platform**: MetaTrader 5 (MT5)
*   **ML Libraries**: Scikit-learn, XGBoost, Joblib, Pandas, NumPy
*   **Broker**: Exness (Optimized)

---

## 📖 Setup & Installation

### 1. Prerequisites
*   Install **MetaTrader 5** on your system.
*   Log in to your **Exness** MT5 trading account (Demo account recommended for testing).
*   Ensure **Python 3.8+** is installed.

### 2. Clone & Install Dependencies
```bash
# Clone the repository
git clone https://github.com/hasham14fw/AI-powered-Trading-Bot.git

# Navigate to the folder
cd AI-powered-Trading-Bot

# Install required python packages
pip install -r requirements.txt
```

### 3. Configuration
Open `config.py` and update your account details:
*   `MT5_LOGIN`: Your MT5 Account Number.
*   `MT5_PASSWORD`: Your Trading Password.
*   `MT5_SERVER`: Your server name (e.g., `Exness-MT5Real`).
*   Adjust `VOLUME`, `STOP_LOSS_POINTS`, and `TAKE_PROFIT_POINTS` as per your risk preference.

---

## 🏃 Manual & Usage

### Running the Bot
1.  Make sure your MetaTrader 5 terminal is open.
2.  Enable **"Algo Trading"** in MT5.
3.  Run the bot:
    ```bash
    python main.py
    ```

### How it Works
1.  **Connection**: The bot connects to your MT5 account.
2.  **Market Analysis**: It fetches the last 100 candles for Gold.
3.  **Feature Extraction**: It calculates technical indicators and SMC patterns (FVG/OB).
4.  **Prediction**: The AI model evaluates the current market state.
5.  **Trade Execution**: 
    *   If **Probability > 82%** → Executes **Strong BUY**.
    *   If **Probability < 18%** → Executes **Strong SELL**.
6.  **Looping**: The bot re-evaluates the market every 5-10 seconds to detect new bars or reversed signals.

---

## ⚠️ Risk Warning
**Financial markets involve significant risk.** This software is provided for **educational purposes only**. Past performance is not indicative of future results. **NEVER** trade money you cannot afford to lose. Always test on a **Demo Account** first.

---

## 👨‍💻 Author
**Hasham** - [GitHub](https://github.com/hasham14fw)
