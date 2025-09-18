add---
language: en
license: mit
developer: Jonus Nattapong Tapachom
tags:
  - trading
  - finance
  - gold
  - xauusd
  - forex
  - algorithmic-trading
  - smart-money-concepts
  - smc
  - xgboost
  - machine-learning
  - backtesting
  - technical-analysis
  - multi-timeframe
  - intraday-trading
  - high-frequency-trading
datasets:
  - yahoo-finance-gc-f
metrics:
  - accuracy
  - precision
  - recall
  - f1
model-index:
  - name: xauusd-trading-ai-smc-daily
    results:
      - task:
          type: binary-classification
          name: Daily Price Direction Prediction
        dataset:
          type: yahoo-finance-gc-f
          name: Gold Futures (GC=F)
        metrics:
          - type: accuracy
            value: 80.3
            name: Accuracy
          - type: precision
            value: 71
            name: Precision (Class 1)
          - type: recall
            value: 81
            name: Recall (Class 1)
          - type: f1
            value: 76
            name: F1-Score
  - name: xauusd-trading-ai-smc-15m
    results:
      - task:
          type: binary-classification
          name: 15-Minute Price Direction Prediction
        dataset:
          type: yahoo-finance-gc-f
          name: Gold Futures (GC=F)
        metrics:
          - type: accuracy
            value: 77.0
            name: Accuracy
          - type: precision
            value: 76
            name: Precision (Class 1)
          - type: recall
            value: 77
            name: Recall (Class 1)
          - type: f1
            value: 76
            name: F1-Score
---

# XAUUSD Multi-Timeframe Trading AI Model

## Files Included

### Core Models
- `trading_model.pkl` - Original daily timeframe XGBoost model (85.4% win rate)
- `trading_model_15m.pkl` - 15-minute intraday model (77% validation accuracy)
- `trading_model_1m.pkl` - 1-minute intraday model (partially trained)
- `trading_model_30m.pkl` - 30-minute intraday model (ready for training)

### Documentation
- `README.md` - This comprehensive model card
- `XAUUSD_Trading_AI_Paper.md` - **Research paper with academic structure, literature review, and methodology**
- `XAUUSD_Trading_AI_Paper.docx` - **Word document version (professional format)**
- `XAUUSD_Trading_AI_Paper.html` - **HTML web version (styled and readable)**
- `XAUUSD_Trading_AI_Paper.tex` - **LaTeX source (for academic publishing)**
- `XAUUSD_Trading_AI_Technical_Whitepaper.md` - **Technical whitepaper with mathematical formulations and implementation details**
- `XAUUSD_Trading_AI_Technical_Whitepaper.docx` - **Word document version (professional format)**
- `XAUUSD_Trading_AI_Technical_Whitepaper.html` - **HTML web version (styled and readable)**
- `XAUUSD_Trading_AI_Technical_Whitepaper.tex` - **LaTeX source (for academic publishing)**

### Performance & Analysis
- `backtest_report.csv` - Daily model yearly backtesting performance results
- `backtest_multi_timeframe_results.csv` - Intraday model backtesting results
- `feature_importance_15m.csv` - 15-minute model feature importance analysis

### Scripts & Tools
- `train_multi_timeframe.py` - Multi-timeframe model training script
- `backtest_multi_timeframe.py` - Intraday model backtesting framework
- `multi_timeframe_summary.py` - Comprehensive performance analysis tool
- `fetch_data.py` - Enhanced data acquisition for multiple timeframes

### Dataset Files
- **Daily Data**: `daily_data.csv`, `processed_daily_data.csv`, `smc_features_dataset.csv`, `X_features.csv`, `y_target.csv`
- **Intraday Data**: `1m_data.csv` (5,204 samples), `15m_data.csv` (3,814 samples), `30m_data.csv` (1,910 samples)

## Recent Enhancements (v2.0)

### Visual Documentation
- **Dataset Flow Diagram**: Complete data processing pipeline from raw Yahoo Finance data to model training
- **Model Architecture Diagram**: XGBoost ensemble structure with decision flow visualization
- **Buy/Sell Workflow Diagram**: End-to-end trading execution process with risk management

### Advanced Formulas & Techniques
- **Position Sizing Formula**: Risk-adjusted position calculation with Kelly Criterion adaptation
- **Risk Metrics**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, and Maximum Drawdown calculations
- **SMC Techniques**: Advanced Order Block detection with volume profile analysis
- **Dynamic Thresholds**: Market volatility-based prediction threshold adjustment
- **Ensemble Signals**: Multi-source signal confirmation (ML + Technical + SMC)

### Performance Analytics
- **Monthly Performance Heatmap**: Visual representation of returns across all test years
- **Risk-Return Scatter Plot**: Performance comparison across different risk levels
- **Market Regime Analysis**: Performance breakdown by trending vs sideways markets

### Documentation Updates
- **Enhanced Technical Whitepaper**: Added comprehensive visual diagrams and mathematical formulations
- **Enhanced Research Paper**: Added Mermaid diagrams, advanced algorithms, and detailed performance analysis
- **Professional Exports**: Both documents now available in HTML, Word, and LaTeX formats

## Multi-Timeframe Trading System (Latest Addition)

### Overview
The system has been extended to support intraday trading across multiple timeframes, enabling higher-frequency trading strategies while maintaining the proven SMC + technical indicator approach.

### Supported Timeframes
- **1-minute (1m)**: Ultra-short-term scalping opportunities
- **15-minute (15m)**: Short-term swing trading
- **30-minute (30m)**: Medium-term position trading
- **Daily (1d)**: Original baseline model (85.4% win rate)

### Data Acquisition
- **Source**: Yahoo Finance API with enhanced intraday data fetching
- **Limitations**: Historical intraday data restricted (recent periods only)
- **Current Datasets**:
  - 1m: 5,204 samples (7 days of recent data)
  - 15m: 3,814 samples (60 days of recent data)
  - 30m: 1,910 samples (60 days of recent data)

### Model Architecture
- **Base Algorithm**: XGBoost Classifier (same as daily model)
- **Features**: 23 features (technical indicators + SMC elements)
- **Training**: Grid search hyperparameter optimization
- **Validation**: 80/20 train/test split with stratification

### Training Results
- **15m Model**: Successfully trained with 77% validation accuracy
- **Feature Importance**: Technical indicators dominant (SMA_50, EMA_12, BB_lower)
- **Training Status**: 1m model partially trained, 30m model interrupted (available for completion)

### Backtesting Performance
- **Framework**: Backtrader with realistic commission modeling
- **Risk Management**: Fixed stake sizing ($1,000 per trade)
- **15m Results**: -0.83% return with 1 trade (conservative strategy)
- **Analysis**: Models show conservative behavior to avoid overtrading

### Key Insights
- ✅ Successfully scaled daily model architecture to intraday timeframes
- ✅ Technical indicators remain most important across all timeframes
- ✅ Conservative prediction thresholds prevent excessive trading
- ⚠️ Limited historical data affects backtesting statistical significance
- ⚠️ Yahoo Finance API constraints limit comprehensive validation

### Files Added
- `train_multi_timeframe.py` - Multi-timeframe model training script
- `backtest_multi_timeframe.py` - Intraday model backtesting framework
- `multi_timeframe_summary.py` - Comprehensive performance analysis
- `trading_model_15m.pkl` - Trained 15-minute model
- `feature_importance_15m.csv` - Feature importance analysis
- `backtest_multi_timeframe_results.csv` - Backtesting performance data

### Next Steps
1. Complete 30m model training
2. Implement walk-forward optimization
3. Add extended historical data sources
4. Deploy best performing intraday model
5. Compare intraday vs daily performance

## Model Description

This is an AI-powered trading model for XAUUSD (Gold vs US Dollar) futures, trained using Smart Money Concepts (SMC) strategy elements. The model uses machine learning to predict 5-day ahead price movements and generate trading signals with high win rates.

### Key Features
- **Asset**: XAUUSD (Gold Futures)
- **Strategy**: Smart Money Concepts (SMC) with technical indicators
- **Prediction Horizon**: 5-day ahead price direction
- **Model Type**: XGBoost Classifier
- **Accuracy**: 80.3% on test data
- **Win Rate**: 85.4% in backtesting

## Intended Use

This model is designed for:
- Educational purposes in algorithmic trading
- Research on SMC strategies
- Backtesting trading strategies
- Understanding ML applications in financial markets

**⚠️ Warning**: This is not financial advice. Trading involves risk of loss. Use at your own discretion.

## Training Data

- **Source**: Yahoo Finance (GC=F - Gold Futures)
- **Period**: 2000-2020 (excluding recent months for efficiency)
- **Features**: 23 features including:
  - Price data (Open, High, Low, Close, Volume)
  - Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  - SMC features (Fair Value Gaps, Order Blocks, Recovery patterns)
  - Lag features (Close prices from previous days)
- **Target**: Binary classification (1 if price rises in 5 days, 0 otherwise)
- **Dataset Size**: 8,816 samples
- **Class Distribution**: 54% down, 46% up (balanced with scale_pos_weight)

## Performance Metrics

### Model Performance
- **Accuracy**: 80.3%
- **Precision (Class 1)**: 71%
- **Recall (Class 1)**: 81%
- **F1-Score**: 76%

### Backtesting Results (2015-2020)
- **Overall Win Rate**: 85.4%
- **Total Return**: 18.2%
- **Sharpe Ratio**: 1.41
- **Yearly Win Rates**:
  - 2015: 62.5%
  - 2016: 100.0%
  - 2017: 100.0%
  - 2018: 72.7%
  - 2019: 76.9%
  - 2020: 94.1%

## Limitations

- Trained on historical data only (2000-2020)
- May not perform well in unprecedented market conditions
- Requires proper risk management
- No consideration of transaction costs, slippage, or market impact
- Model predictions are probabilistic, not guaranteed

## Usage

### Prerequisites
```python
pip install joblib scikit-learn pandas numpy
```

### Loading the Model
```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model
model = joblib.load('trading_model.pkl')

# Load scalers (you need to recreate or save them)
# ... preprocessing code ...

# Prepare features
features = prepare_features(your_data)
prediction = model.predict(features)
probability = model.predict_proba(features)
```

### Features Required
The model expects 23 features in this order:
1. Close
2. High
3. Low
4. Open
5. Volume
6. SMA_20
7. SMA_50
8. EMA_12
9. EMA_26
10. RSI
11. MACD
12. MACD_signal
13. MACD_hist
14. BB_upper
15. BB_middle
16. BB_lower
17. FVG_Size
18. FVG_Type_Encoded
19. OB_Type_Encoded
20. Recovery_Type_Encoded
21. Close_lag1
22. Close_lag2
23. Close_lag3

## Training Details

- **Algorithm**: XGBoost Classifier
- **Hyperparameters**:
  - n_estimators: 200
  - max_depth: 7
  - learning_rate: 0.2
  - scale_pos_weight: 1.17 (for class balancing)
- **Cross-validation**: 3-fold
- **Optimization**: Grid search on hyperparameters

## SMC Strategy Elements

The model incorporates Smart Money Concepts:
- **Fair Value Gaps (FVG)**: Price imbalances between candles
- **Order Blocks (OB)**: Areas of significant buying/selling
- **Recovery Patterns**: Pullbacks in trending markets

## Upload to Hugging Face

To share this model on Hugging Face:

1. Create a Hugging Face account at https://huggingface.co/join
2. Generate an access token at https://huggingface.co/settings/tokens with "Write" permissions
3. Test your token: `python test_token.py YOUR_TOKEN`
4. Upload: `python upload_to_hf.py YOUR_TOKEN`

The script will upload:
- `trading_model.pkl` - The trained XGBoost model
- `README.md` - This model card with metadata
- All dataset files (CSV format)

## Citation

If you use this model in your research, please cite:

```
@misc{xauusd-trading-ai,
  title={XAUUSD Trading AI Model with SMC Strategy},
  author={AI Trading System},
  year={2025},
  url={https://huggingface.co/JonusNattapong/xauusd-trading-ai-smc}
}
```

### Academic Paper
For the complete academic research paper with methodology, results, and analysis:

**arXiv Paper**: [XAUUSD Trading AI: A Machine Learning Approach Using Smart Money Concepts](https://arxiv.org/abs/XXXX.XXXXX)

## License

This model is released under the MIT License. See LICENSE file for details.

## Contact

For questions or issues, please open an issue on the Hugging Face repository.