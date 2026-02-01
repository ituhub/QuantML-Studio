# 🚀 QuantML Studio v4.0

> **Enterprise AI/ML Platform** - A comprehensive AutoML and deep learning platform combining traditional machine learning, neural networks, model explainability, drift detection, and risk-aware metrics.

![Version](https://img.shields.io/badge/version-4.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10+-green)
![License](https://img.shields.io/badge/license-MIT-purple)

## ✨ Features

### 🤖 AutoML Engine
- **15+ ML Algorithms**: Linear models, tree-based, boosting (XGBoost, LightGBM, CatBoost), SVM, KNN
- **Automatic Hyperparameter Tuning**: Random search, grid search, Bayesian optimization
- **Cross-Validation**: K-Fold, Stratified K-Fold, Time Series Split
- **Auto Task Detection**: Regression vs Classification

### 🧠 Deep Learning
- **5 Neural Network Architectures**:
  - 🤖 **Transformer**: Attention-based architecture for tabular data
  - 🔄 **CNN-LSTM**: Hybrid convolutional + recurrent network
  - 📊 **TCN**: Temporal Convolutional Network with dilated convolutions
  - 🧠 **MLP**: Multi-Layer Perceptron with residual connections
  - 🔁 **LSTM-GRU Ensemble**: Combined recurrent architecture

### 🔍 Model Explainability
- **SHAP Values**: Feature attribution analysis
- **Permutation Importance**: Model-agnostic importance
- **Feature Importance**: Tree-based importance extraction
- **Explanation Reports**: Human-readable interpretation

### 🚨 Drift Detection
- **KS Test**: Kolmogorov-Smirnov statistical test
- **PSI**: Population Stability Index
- **Feature-Level Analysis**: Per-feature drift metrics

### 📊 Risk Metrics
- **Sharpe Ratio**: Risk-adjusted returns
- **Sortino Ratio**: Downside risk measure
- **VaR**: Value at Risk (95%, 99%)
- **CVaR**: Conditional Value at Risk
- **Maximum Drawdown**: Peak-to-trough decline
- **Calmar Ratio**: Return vs drawdown ratio

### 🎭 Ensemble Methods
- **Weighted Ensemble**: CV-performance based weighting
- **Model Stacking**: Meta-learning approach
- **Voting**: Soft and hard voting

## 📋 Subscription Tiers

| Feature | Free | Starter ($29/mo) | Professional ($99/mo) | Enterprise |
|---------|------|------------------|----------------------|------------|
| Max Rows | 1,000 | 10,000 | 100,000 | Unlimited |
| Max Models | 5 | 15 | 30 | 50 |
| Time Limit | 5 min | 30 min | 2 hours | 8 hours |
| Neural Networks | ❌ | ❌ | ✅ | ✅ |
| Ensemble Models | ❌ | ✅ | ✅ | ✅ |
| Explainability | ❌ | ✅ | ✅ | ✅ |
| Drift Detection | ❌ | ❌ | ✅ | ✅ |
| Risk Metrics | ❌ | ❌ | ✅ | ✅ |
| Batch Predictions | ❌ | ✅ | ✅ | ✅ |
| API Access | ❌ | ❌ | ✅ | ✅ |
| Model Export | ❌ | ✅ | ✅ | ✅ |

## 🚀 Quick Start

### Local Installation

```bash
# Clone or download the project
cd quantml_studio

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Docker Installation

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access the application
# Streamlit UI: http://localhost:8501
# REST API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## 🔑 Demo Keys

| Key | Tier | Features |
|-----|------|----------|
| `STARTER-2024` | Starter | Ensemble, batch predictions, explainability |
| `PRO-2024` | Professional | + Neural networks, drift detection, risk metrics, API |
| `ENTERPRISE-2024` | Enterprise | All features, unlimited data |

## 📁 Project Structure

```
quantml_studio/
├── app.py                 # Main Streamlit application
├── api.py                 # FastAPI REST API
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── README.md             # Documentation
└── core/                 # Core ML engine
    ├── __init__.py
    ├── automl_engine.py  # AutoML training engine
    └── neural_networks.py # Deep learning models
```

## 🔧 Configuration

### Environment Variables

```bash
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
PYTHONPATH=/app
```

### AutoML Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| max_models | Maximum models to train | 8 |
| cv_folds | Cross-validation folds | 5 |
| test_size | Test set proportion | 0.2 |
| time_limit | Training time limit (min) | 5 |
| early_stopping | Early stopping rounds | 10 |

## 📊 Supported Algorithms

### Traditional ML
- **Linear**: Linear Regression, Ridge, Lasso, ElasticNet, Logistic Regression
- **Tree-based**: Decision Tree, Random Forest, Extra Trees
- **Boosting**: Gradient Boosting, XGBoost, LightGBM, CatBoost
- **Other**: SVR/SVC, KNN, Naive Bayes

### Neural Networks
- **Transformer**: Multi-head attention with positional encoding
- **CNN-LSTM**: Convolutional feature extraction + LSTM sequence modeling
- **TCN**: Dilated causal convolutions for time series
- **MLP**: Deep feedforward with residual connections
- **LSTM-GRU**: Ensemble of bidirectional recurrent networks

## 🔌 REST API

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API root |
| GET | `/health` | Health check |
| GET | `/models` | List models |
| GET | `/models/{id}` | Get model info |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| POST | `/train` | Start training |
| GET | `/usage` | Usage statistics |

### API Keys

```bash
# Demo API key
curl -H "x-api-key: demo-api-key-12345" http://localhost:8000/health

# Professional API key
curl -H "x-api-key: pro-api-key-2024" http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"feature_1": 1.5, "feature_2": 2.3}}'
```

## 📝 Usage Example

### Python API

```python
from core.automl_engine import AutoMLEngine, AutoMLConfig, auto_train
import pandas as pd

# Load data
df = pd.read_csv("your_data.csv")

# Quick training
engine = auto_train(
    df, 
    target_column='target',
    task_type='regression',
    time_limit=5
)

# Get results
print(engine.get_leaderboard())
print(f"Best model: {engine.best_model_name}")

# Make predictions
predictions = engine.predict(X_new)

# Get explanations
explanations = engine.get_explanations()
print(engine.get_explanation_report())

# Check for drift
drift_detected, drift_score, feature_drift = engine.detect_drift(X_new)
```

### Streamlit App

1. **Upload Data**: CSV file with features and target column
2. **Configure Training**: Set models, time limit, and options
3. **Train Models**: Click "Start Training" button
4. **View Results**: Check leaderboard and visualizations
5. **Explainability**: Analyze feature importance and SHAP values
6. **Make Predictions**: Single or batch predictions

## 🐛 Troubleshooting

### Common Issues

1. **"AutoML engine not available"**
   - Ensure `core/` directory exists with all files
   - Check Python path: `export PYTHONPATH=/path/to/project`

2. **"PyTorch not available"**
   - Install PyTorch: `pip install torch`
   - For GPU: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

3. **Neural networks not showing**
   - Requires Professional tier or above
   - Use key: `PRO-2024` or `ENTERPRISE-2024`

4. **Training timeout**
   - Reduce `max_models` or `cv_folds`
   - Increase `time_limit` if your tier allows

5. **Memory issues**
   - Reduce data size
   - Use fewer models
   - Consider sampling large datasets

## 🆕 What's New in v4.0

- ✅ Unified platform combining AutoML and trading system
- ✅ 5 neural network architectures (Transformer, CNN-LSTM, TCN, MLP, LSTM-GRU)
- ✅ Advanced feature engineering (100+ features)
- ✅ Model explainability (SHAP, permutation, feature importance)
- ✅ Data drift detection (KS test, PSI)
- ✅ Risk metrics (Sharpe, Sortino, VaR, CVaR, Max Drawdown)
- ✅ Weighted ensemble predictions
- ✅ Beautiful, modern UI with distinctive design
- ✅ REST API with tiered access
- ✅ Docker support

## 📄 License

MIT License - Free for personal and commercial use.

## 🤝 Support

- 📧 Email: support@quantml-studio.com
- 📖 Docs: https://docs.quantml-studio.com
- 🐛 Issues: https://github.com/quantml-studio/issues

---

Built with ❤️ using Python, Streamlit, FastAPI, and PyTorch
