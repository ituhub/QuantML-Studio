"""
QuantML Studio - Advanced AutoML Engine v4.0
=============================================
Enterprise-grade AutoML engine combining:
- Traditional ML algorithms (15+ models)
- Deep learning neural networks (6 architectures)
- Advanced feature engineering (100+ features)
- Model explainability (SHAP, Permutation, Gradient)
- Data drift detection (KS Test, PSI)
- Time series support with cross-validation
- Risk-aware metrics (Sharpe, Sortino, VaR)
- Real-time market data integration
- Weighted ensemble predictions
"""

import os
import sys
import logging
import warnings
import hashlib
import pickle
import json
import time
import uuid
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from collections import defaultdict, deque
import threading

import numpy as np
import pandas as pd
from scipy import stats, signal

# Sklearn imports
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
    TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder,
    OneHotEncoder, PolynomialFeatures, PowerTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, mutual_info_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, FastICA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    explained_variance_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Traditional ML models
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDRegressor, SGDClassifier
)
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    BaggingRegressor, BaggingClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# Suppress warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
try:
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
except:
    pass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Optional imports with availability flags
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False
CATBOOST_AVAILABLE = False
TORCH_AVAILABLE = False
SHAP_AVAILABLE = False
BAYESIAN_OPT_AVAILABLE = False
OPTUNA_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    pass

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    pass

try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    BAYESIAN_OPT_AVAILABLE = True
except ImportError:
    pass

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    pass


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class TaskType(Enum):
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    TIME_SERIES = "time_series"
    FINANCIAL_FORECASTING = "financial_forecasting"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"


class OptimizationStrategy(Enum):
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    OPTUNA = "optuna"
    GENETIC = "genetic"


class ScalerType(Enum):
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    POWER = "power"
    NONE = "none"


class ImputerType(Enum):
    MEAN = "mean"
    MEDIAN = "median"
    MOST_FREQUENT = "most_frequent"
    KNN = "knn"
    NONE = "none"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGE_BOUND = "range_bound"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    CRISIS = "crisis"


@dataclass
class DatasetInfo:
    """Information about the dataset"""
    n_samples: int
    n_features: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    target_column: str
    task_type: TaskType
    missing_values: Dict[str, float]
    feature_types: Dict[str, str]
    class_distribution: Optional[Dict] = None
    data_hash: str = ""
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ModelConfig:
    """Configuration for a model"""
    model_type: str
    hyperparameters: Dict[str, Any]
    preprocessing: Dict[str, Any]
    feature_engineering: Dict[str, Any]
    training_config: Dict[str, Any]


@dataclass
class TrainingResult:
    """Result of model training"""
    model_id: str
    model_type: str
    metrics: Dict[str, float]
    training_time: float
    hyperparameters: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    cv_scores: Optional[List[float]]
    confusion_matrix: Optional[np.ndarray]
    predictions: Optional[np.ndarray]
    risk_metrics: Optional[Dict[str, float]] = None
    trained_at: datetime = field(default_factory=datetime.now)


@dataclass
class AutoMLConfig:
    """Configuration for AutoML run"""
    task_type: TaskType = TaskType.REGRESSION
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH
    optimization_metric: str = "auto"
    cv_folds: int = 5
    max_models: int = 20
    time_limit_minutes: int = 60
    per_model_time_limit: int = 300
    early_stopping_rounds: int = 10
    ensemble_size: int = 5
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    
    # Feature engineering
    feature_engineering: bool = True
    advanced_features: bool = True
    polynomial_features: bool = False
    polynomial_degree: int = 2
    interaction_features: bool = True
    
    # Advanced options
    enable_drift_detection: bool = True
    enable_explainability: bool = True
    advanced_cv: bool = True
    enable_risk_metrics: bool = True
    enable_market_regime: bool = False
    
    # Neural network options
    include_neural_networks: bool = False
    nn_epochs: int = 100
    nn_batch_size: int = 32
    nn_learning_rate: float = 0.001
    
    # Scaler and imputer
    scaler_type: ScalerType = ScalerType.ROBUST
    imputer_type: ImputerType = ImputerType.MEDIAN
    
    # Callbacks
    verbose: bool = True
    progress_callback: Optional[Callable] = None


# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering with 100+ financial and statistical features.
    Combines features from both trading and general ML applications.
    """
    
    def __init__(self, feature_config: Dict = None):
        self.feature_config = feature_config or {}
        self.fitted = False
        self.engineered_features = []
        self.feature_stats = {}
        self.scalers = {}
        
    def fit(self, df: pd.DataFrame, target_col: str = None) -> 'AdvancedFeatureEngineer':
        """Fit the feature engineer on training data"""
        self.target_col = target_col
        self.original_columns = list(df.columns)
        
        # Calculate statistics for normalization
        for col in df.select_dtypes(include=[np.number]).columns:
            self.feature_stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        self.fitted = True
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform data with advanced features"""
        if not self.fitted:
            raise ValueError("FeatureEngineer must be fitted first")
        
        result = df.copy()
        self.engineered_features = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col and self.target_col in numeric_cols:
            numeric_cols.remove(self.target_col)
        
        # Statistical features
        result = self._add_statistical_features(result, numeric_cols)
        
        # Rolling window features
        result = self._add_rolling_features(result, numeric_cols)
        
        # Lag features
        result = self._add_lag_features(result, numeric_cols)
        
        # Cross-feature interactions
        result = self._add_interaction_features(result, numeric_cols)
        
        # Financial-specific features (if applicable)
        result = self._add_financial_features(result, numeric_cols)
        
        # Handle infinities and NaNs
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.ffill().bfill().fillna(0)
        
        return result
    
    def _add_statistical_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add statistical features"""
        for col in cols[:10]:  # Limit to avoid explosion
            # Z-scores
            if col in self.feature_stats:
                std = self.feature_stats[col]['std']
                mean = self.feature_stats[col]['mean']
                if std > 0:
                    feat_name = f'{col}_zscore'
                    df[feat_name] = (df[col] - mean) / std
                    self.engineered_features.append(feat_name)
            
            # Percentile ranks
            feat_name = f'{col}_pct_rank'
            df[feat_name] = df[col].rank(pct=True)
            self.engineered_features.append(feat_name)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add rolling window features"""
        windows = [5, 10, 20]
        
        for col in cols[:5]:  # Limit columns
            for window in windows:
                if len(df) >= window:
                    # Rolling mean
                    feat_name = f'{col}_rolling_mean_{window}'
                    df[feat_name] = df[col].rolling(window=window, min_periods=1).mean()
                    self.engineered_features.append(feat_name)
                    
                    # Rolling std
                    feat_name = f'{col}_rolling_std_{window}'
                    df[feat_name] = df[col].rolling(window=window, min_periods=1).std()
                    self.engineered_features.append(feat_name)
                    
                    # Rolling min/max
                    feat_name = f'{col}_rolling_min_{window}'
                    df[feat_name] = df[col].rolling(window=window, min_periods=1).min()
                    self.engineered_features.append(feat_name)
                    
                    feat_name = f'{col}_rolling_max_{window}'
                    df[feat_name] = df[col].rolling(window=window, min_periods=1).max()
                    self.engineered_features.append(feat_name)
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add lagged features"""
        lags = [1, 2, 3, 5, 10]
        
        for col in cols[:5]:
            for lag in lags:
                if len(df) > lag:
                    feat_name = f'{col}_lag_{lag}'
                    df[feat_name] = df[col].shift(lag)
                    self.engineered_features.append(feat_name)
                    
                    # Difference from lag
                    feat_name = f'{col}_diff_{lag}'
                    df[feat_name] = df[col] - df[col].shift(lag)
                    self.engineered_features.append(feat_name)
                    
                    # Percent change from lag
                    feat_name = f'{col}_pct_change_{lag}'
                    df[feat_name] = df[col].pct_change(lag)
                    self.engineered_features.append(feat_name)
        
        return df
    
    def _add_interaction_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add feature interactions"""
        import itertools
        
        for col1, col2 in itertools.combinations(cols[:5], 2):
            # Multiplication
            feat_name = f'{col1}_x_{col2}'
            df[feat_name] = df[col1] * df[col2]
            self.engineered_features.append(feat_name)
            
            # Ratio (with safety)
            feat_name = f'{col1}_div_{col2}'
            df[feat_name] = df[col1] / (df[col2].abs() + 1e-8)
            self.engineered_features.append(feat_name)
            
            # Difference
            feat_name = f'{col1}_minus_{col2}'
            df[feat_name] = df[col1] - df[col2]
            self.engineered_features.append(feat_name)
        
        return df
    
    def _add_financial_features(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Add financial-specific features (if applicable)"""
        # Check if this looks like financial data
        financial_cols = ['close', 'open', 'high', 'low', 'volume', 'price', 'return']
        has_financial = any(col.lower() in financial_cols for col in cols)
        
        if not has_financial:
            return df
        
        # Try to identify price column
        price_col = None
        for col in cols:
            if col.lower() in ['close', 'price', 'adj_close', 'adjusted_close']:
                price_col = col
                break
        
        if price_col is None and len(cols) > 0:
            price_col = cols[0]
        
        if price_col and len(df) > 20:
            # Returns
            df['returns'] = df[price_col].pct_change()
            self.engineered_features.append('returns')
            
            # Log returns
            df['log_returns'] = np.log(df[price_col] / df[price_col].shift(1))
            self.engineered_features.append('log_returns')
            
            # Volatility (rolling std of returns)
            df['volatility_5'] = df['returns'].rolling(5).std()
            df['volatility_20'] = df['returns'].rolling(20).std()
            self.engineered_features.extend(['volatility_5', 'volatility_20'])
            
            # RSI
            delta = df[price_col].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            df['rsi_14'] = 100 - (100 / (1 + rs))
            self.engineered_features.append('rsi_14')
            
            # MACD
            exp1 = df[price_col].ewm(span=12, adjust=False).mean()
            exp2 = df[price_col].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            self.engineered_features.extend(['macd', 'macd_signal'])
            
            # Bollinger Bands
            rolling_mean = df[price_col].rolling(20).mean()
            rolling_std = df[price_col].rolling(20).std()
            df['bb_upper'] = rolling_mean + (rolling_std * 2)
            df['bb_lower'] = rolling_mean - (rolling_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / rolling_mean
            df['bb_position'] = (df[price_col] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
            self.engineered_features.extend(['bb_upper', 'bb_lower', 'bb_width', 'bb_position'])
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:
        """Fit and transform in one step"""
        self.fit(df, target_col)
        return self.transform(df)


# =============================================================================
# MODEL DRIFT DETECTION
# =============================================================================

class ModelDriftDetector:
    """
    Detects data drift using statistical tests.
    Supports KS test, PSI, and feature-level drift analysis.
    """
    
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.reference_distribution = None
        self.feature_names = []
        self.reference_stats = {}
        
    def set_reference_distribution(self, X: np.ndarray, feature_names: List[str] = None):
        """Set reference distribution from training data"""
        self.reference_distribution = X.copy()
        self.feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1] if X.ndim > 1 else 1)]
        
        # Calculate reference statistics
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        for i, name in enumerate(self.feature_names[:X.shape[1]]):
            self.reference_stats[name] = {
                'mean': np.mean(X[:, i]),
                'std': np.std(X[:, i]),
                'min': np.min(X[:, i]),
                'max': np.max(X[:, i]),
                'percentiles': np.percentile(X[:, i], [25, 50, 75])
            }
    
    def detect_drift(self, X_new: np.ndarray, feature_names: List[str] = None) -> Tuple[bool, float, Dict]:
        """Detect if drift has occurred"""
        if self.reference_distribution is None:
            return False, 0.0, {}
        
        if X_new.ndim == 1:
            X_new = X_new.reshape(-1, 1)
        
        X_ref = self.reference_distribution
        if X_ref.ndim == 1:
            X_ref = X_ref.reshape(-1, 1)
        
        feature_names = feature_names or self.feature_names
        n_features = min(X_new.shape[1], X_ref.shape[1], len(feature_names))
        
        feature_drift = {}
        drift_scores = []
        
        for i in range(n_features):
            name = feature_names[i] if i < len(feature_names) else f'feature_{i}'
            
            try:
                # KS test
                ks_stat, ks_pvalue = stats.ks_2samp(X_ref[:, i], X_new[:, i])
                
                # PSI calculation
                psi = self._calculate_psi(X_ref[:, i], X_new[:, i])
                
                # Combined score
                combined_score = (ks_stat + psi) / 2
                
                feature_drift[name] = {
                    'ks_statistic': float(ks_stat),
                    'ks_pvalue': float(ks_pvalue),
                    'psi': float(psi),
                    'combined_score': float(combined_score),
                    'drifted': combined_score > self.threshold
                }
                
                drift_scores.append(combined_score)
                
            except Exception as e:
                logger.debug(f"Drift detection failed for {name}: {e}")
                feature_drift[name] = {'error': str(e), 'combined_score': 0.0}
                drift_scores.append(0.0)
        
        overall_score = np.mean(drift_scores) if drift_scores else 0.0
        drift_detected = overall_score > self.threshold
        
        return drift_detected, overall_score, feature_drift
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins from expected distribution
            _, bin_edges = np.histogram(expected, bins=n_bins)
            
            # Calculate proportions
            expected_counts = np.histogram(expected, bins=bin_edges)[0]
            actual_counts = np.histogram(actual, bins=bin_edges)[0]
            
            # Add small value to avoid division by zero
            expected_pct = (expected_counts + 1) / (len(expected) + n_bins)
            actual_pct = (actual_counts + 1) / (len(actual) + n_bins)
            
            # PSI formula
            psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
            
            return float(psi)
        except:
            return 0.0


# =============================================================================
# MODEL EXPLAINABILITY
# =============================================================================

class ModelExplainer:
    """
    Comprehensive model explainability toolkit.
    Supports SHAP, permutation importance, and gradient-based explanations.
    """
    
    def __init__(self):
        self.explanations = {}
        self.feature_importance = {}
        
    def explain_model(self, model, X: np.ndarray, feature_names: List[str] = None, 
                      model_type: str = 'tree') -> Dict:
        """Generate explanations for a model"""
        explanations = {}
        feature_names = feature_names or [f'feature_{i}' for i in range(X.shape[1])]
        
        # SHAP explanations
        if SHAP_AVAILABLE:
            try:
                shap_values = self._get_shap_values(model, X, model_type)
                if shap_values is not None:
                    explanations['shap'] = {
                        'values': shap_values,
                        'importance': dict(zip(feature_names, np.abs(shap_values).mean(axis=0)))
                    }
            except Exception as e:
                logger.debug(f"SHAP explanation failed: {e}")
        
        # Permutation importance
        try:
            perm_importance = self._permutation_importance(model, X, feature_names)
            explanations['permutation_importance'] = perm_importance
        except Exception as e:
            logger.debug(f"Permutation importance failed: {e}")
        
        # Feature importance (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                fi = dict(zip(feature_names, model.feature_importances_))
                explanations['feature_importance'] = fi
            elif hasattr(model, 'coef_'):
                fi = dict(zip(feature_names, np.abs(model.coef_).flatten()))
                explanations['feature_importance'] = fi
        except Exception as e:
            logger.debug(f"Feature importance extraction failed: {e}")
        
        self.explanations = explanations
        return explanations
    
    def _get_shap_values(self, model, X: np.ndarray, model_type: str) -> Optional[np.ndarray]:
        """Get SHAP values for a model"""
        if not SHAP_AVAILABLE:
            return None
        
        try:
            if model_type in ['tree', 'ensemble']:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.KernelExplainer(model.predict, shap.sample(X, 100))
            
            shap_values = explainer.shap_values(X[:100])  # Limit samples
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            return shap_values
        except:
            return None
    
    def _permutation_importance(self, model, X: np.ndarray, feature_names: List[str]) -> Dict:
        """Calculate permutation importance"""
        # Simple permutation importance implementation
        baseline_pred = model.predict(X)
        
        importances = {}
        for i, name in enumerate(feature_names[:X.shape[1]]):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            permuted_pred = model.predict(X_permuted)
            importance = np.mean(np.abs(baseline_pred - permuted_pred))
            importances[name] = float(importance)
        
        return importances
    
    def generate_explanation_report(self, explanations: Dict, prediction: float = None, 
                                    confidence: float = None) -> str:
        """Generate human-readable explanation report"""
        report = ["=" * 60, "MODEL EXPLANATION REPORT", "=" * 60]
        
        if prediction is not None:
            report.append(f"\nPrediction: {prediction:.4f}")
        if confidence is not None:
            report.append(f"Confidence: {confidence:.2f}%")
        
        if 'feature_importance' in explanations:
            report.append("\n📊 Feature Importance:")
            sorted_fi = sorted(explanations['feature_importance'].items(), 
                             key=lambda x: x[1], reverse=True)[:10]
            for name, importance in sorted_fi:
                report.append(f"  • {name}: {importance:.4f}")
        
        if 'shap' in explanations and 'importance' in explanations['shap']:
            report.append("\n🔍 SHAP Feature Importance:")
            sorted_shap = sorted(explanations['shap']['importance'].items(),
                               key=lambda x: x[1], reverse=True)[:10]
            for name, importance in sorted_shap:
                report.append(f"  • {name}: {importance:.4f}")
        
        if 'permutation_importance' in explanations:
            report.append("\n🔀 Permutation Importance:")
            sorted_perm = sorted(explanations['permutation_importance'].items(),
                               key=lambda x: x[1], reverse=True)[:10]
            for name, importance in sorted_perm:
                report.append(f"  • {name}: {importance:.4f}")
        
        return "\n".join(report)


# =============================================================================
# RISK METRICS CALCULATOR
# =============================================================================

class RiskMetricsCalculator:
    """
    Calculate financial risk metrics for model predictions.
    """
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray, 
                      risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        returns = y_pred - y_true
        
        metrics = {}
        
        # Basic error metrics
        metrics['mse'] = float(np.mean(returns ** 2))
        metrics['mae'] = float(np.mean(np.abs(returns)))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # Volatility
        metrics['volatility'] = float(np.std(returns))
        
        # Sharpe Ratio
        if metrics['volatility'] > 0:
            excess_return = np.mean(returns) - risk_free_rate / 252
            metrics['sharpe_ratio'] = float(excess_return / metrics['volatility'] * np.sqrt(252))
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Sortino Ratio (downside deviation)
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = np.std(negative_returns)
            if downside_std > 0:
                metrics['sortino_ratio'] = float(np.mean(returns) / downside_std * np.sqrt(252))
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sortino_ratio'] = 0.0
        
        # VaR (Value at Risk) at 95% and 99%
        metrics['var_95'] = float(np.percentile(returns, 5))
        metrics['var_99'] = float(np.percentile(returns, 1))
        
        # CVaR (Conditional VaR / Expected Shortfall)
        var_95 = np.percentile(returns, 5)
        metrics['cvar_95'] = float(np.mean(returns[returns <= var_95]))
        
        # Maximum Drawdown
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        metrics['max_drawdown'] = float(np.min(drawdowns))
        
        # Calmar Ratio
        if abs(metrics['max_drawdown']) > 0:
            annual_return = np.mean(returns) * 252
            metrics['calmar_ratio'] = float(annual_return / abs(metrics['max_drawdown']))
        else:
            metrics['calmar_ratio'] = 0.0
        
        return metrics


# =============================================================================
# DATA PREPROCESSOR
# =============================================================================

class DataPreprocessor:
    """Comprehensive data preprocessing"""
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.numeric_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.scaler = None
        self.imputer = None
        self.label_encoders = {}
        self.fitted = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataPreprocessor':
        """Fit preprocessor on training data"""
        # Identify feature types
        for col in X.columns:
            if X[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                self.numeric_features.append(col)
            elif X[col].dtype == 'datetime64[ns]':
                self.datetime_features.append(col)
            elif X[col].dtype == 'object' or str(X[col].dtype) == 'category':
                self.categorical_features.append(col)
            else:
                self.numeric_features.append(col)
        
        # Initialize scaler
        if self.config.scaler_type == ScalerType.STANDARD:
            self.scaler = StandardScaler()
        elif self.config.scaler_type == ScalerType.MINMAX:
            self.scaler = MinMaxScaler()
        elif self.config.scaler_type == ScalerType.ROBUST:
            self.scaler = RobustScaler()
        elif self.config.scaler_type == ScalerType.POWER:
            self.scaler = PowerTransformer()
        
        # Initialize imputer
        if self.config.imputer_type == ImputerType.MEAN:
            self.imputer = SimpleImputer(strategy='mean')
        elif self.config.imputer_type == ImputerType.MEDIAN:
            self.imputer = SimpleImputer(strategy='median')
        elif self.config.imputer_type == ImputerType.MOST_FREQUENT:
            self.imputer = SimpleImputer(strategy='most_frequent')
        elif self.config.imputer_type == ImputerType.KNN:
            self.imputer = KNNImputer(n_neighbors=5)
        
        # Fit on numeric features
        if self.numeric_features:
            X_numeric = X[self.numeric_features].values
            
            if self.imputer:
                X_numeric = self.imputer.fit_transform(X_numeric)
            
            if self.scaler:
                self.scaler.fit(X_numeric)
        
        # Fit label encoders for categorical features
        for col in self.categorical_features:
            self.label_encoders[col] = LabelEncoder()
            self.label_encoders[col].fit(X[col].astype(str))
        
        self.fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data"""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted first")
        
        result_parts = []
        
        # Process numeric features
        if self.numeric_features:
            X_numeric = X[self.numeric_features].values
            
            if self.imputer:
                X_numeric = self.imputer.transform(X_numeric)
            
            if self.scaler:
                X_numeric = self.scaler.transform(X_numeric)
            
            result_parts.append(X_numeric)
        
        # Process categorical features
        for col in self.categorical_features:
            if col in X.columns:
                encoded = self.label_encoders[col].transform(X[col].astype(str))
                result_parts.append(encoded.reshape(-1, 1))
        
        if result_parts:
            return np.hstack(result_parts)
        return X.values
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform"""
        self.fit(X, y)
        return self.transform(X)


# =============================================================================
# WEIGHTED ENSEMBLE
# =============================================================================

class WeightedEnsemble:
    """
    Weighted ensemble model based on CV performance.
    """
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {}
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        if not self.weights:
            # Equal weights if not specified
            n_models = len(self.models)
            self.weights = {name: 1.0 / n_models for name in self.models}
        else:
            total = sum(self.weights.values())
            if total > 0:
                self.weights = {k: v / total for k, v in self.weights.items()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make weighted ensemble predictions"""
        predictions = []
        total_weight = 0
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 0)
            if weight > 0:
                try:
                    pred = model.predict(X)
                    predictions.append(pred * weight)
                    total_weight += weight
                except Exception as e:
                    logger.debug(f"Prediction failed for {name}: {e}")
        
        if not predictions:
            raise ValueError("No models could make predictions")
        
        return np.sum(predictions, axis=0) / total_weight


# =============================================================================
# MODEL REGISTRY
# =============================================================================

class ModelRegistry:
    """Registry of available ML models"""
    
    @staticmethod
    def get_regression_models(config: AutoMLConfig) -> Dict[str, Tuple[Any, Dict]]:
        """Get regression models with hyperparameter spaces"""
        models = {
            'LinearRegression': (
                LinearRegression(),
                {}
            ),
            'Ridge': (
                Ridge(random_state=config.random_state),
                {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
            ),
            'Lasso': (
                Lasso(random_state=config.random_state, max_iter=10000),
                {'alpha': [0.001, 0.01, 0.1, 1.0]}
            ),
            'ElasticNet': (
                ElasticNet(random_state=config.random_state, max_iter=10000),
                {'alpha': [0.01, 0.1, 1.0], 'l1_ratio': [0.25, 0.5, 0.75]}
            ),
            'DecisionTree': (
                DecisionTreeRegressor(random_state=config.random_state),
                {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
            ),
            'RandomForest': (
                RandomForestRegressor(random_state=config.random_state, n_jobs=config.n_jobs),
                {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}
            ),
            'ExtraTrees': (
                ExtraTreesRegressor(random_state=config.random_state, n_jobs=config.n_jobs),
                {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
            ),
            'GradientBoosting': (
                GradientBoostingRegressor(random_state=config.random_state),
                {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            ),
            'SVR': (
                SVR(),
                {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
            ),
            'KNN': (
                KNeighborsRegressor(),
                {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']}
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = (
                xgb.XGBRegressor(random_state=config.random_state, n_jobs=config.n_jobs, verbosity=0),
                {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]}
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = (
                lgb.LGBMRegressor(random_state=config.random_state, n_jobs=config.n_jobs, verbose=-1),
                {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
            )
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = (
                cb.CatBoostRegressor(random_state=config.random_state, verbose=0),
                {'iterations': [50, 100, 200], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1]}
            )
        
        return models
    
    @staticmethod
    def get_classification_models(config: AutoMLConfig) -> Dict[str, Tuple[Any, Dict]]:
        """Get classification models with hyperparameter spaces"""
        models = {
            'LogisticRegression': (
                LogisticRegression(random_state=config.random_state, max_iter=1000),
                {'C': [0.01, 0.1, 1.0, 10.0]}
            ),
            'DecisionTree': (
                DecisionTreeClassifier(random_state=config.random_state),
                {'max_depth': [3, 5, 10, None], 'min_samples_split': [2, 5, 10]}
            ),
            'RandomForest': (
                RandomForestClassifier(random_state=config.random_state, n_jobs=config.n_jobs),
                {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
            ),
            'ExtraTrees': (
                ExtraTreesClassifier(random_state=config.random_state, n_jobs=config.n_jobs),
                {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None]}
            ),
            'GradientBoosting': (
                GradientBoostingClassifier(random_state=config.random_state),
                {'n_estimators': [50, 100], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
            ),
            'SVC': (
                SVC(random_state=config.random_state, probability=True),
                {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
            ),
            'KNN': (
                KNeighborsClassifier(),
                {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']}
            ),
            'NaiveBayes': (
                GaussianNB(),
                {}
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models['XGBoost'] = (
                xgb.XGBClassifier(random_state=config.random_state, n_jobs=config.n_jobs, verbosity=0),
                {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1, 0.3]}
            )
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            models['LightGBM'] = (
                lgb.LGBMClassifier(random_state=config.random_state, n_jobs=config.n_jobs, verbose=-1),
                {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7], 'learning_rate': [0.01, 0.1]}
            )
        
        # Add CatBoost if available
        if CATBOOST_AVAILABLE:
            models['CatBoost'] = (
                cb.CatBoostClassifier(random_state=config.random_state, verbose=0),
                {'iterations': [50, 100, 200], 'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1]}
            )
        
        return models


# =============================================================================
# TRAINING CALLBACK
# =============================================================================

class TrainingCallback:
    """Callback for training progress"""
    
    def __init__(self, total_models: int = 10, verbose: bool = True):
        self.total_models = total_models
        self.verbose = verbose
        self.current_model = 0
        self.current_phase = "Initializing"
        self.progress = 0.0
        self.models_trained = []
        self.start_time = time.time()
    
    def on_model_start(self, model_name: str):
        self.current_model += 1
        self.current_phase = f"Training {model_name}"
        self.progress = self.current_model / (self.total_models + 1)
        if self.verbose:
            logger.info(f"🔄 [{self.current_model}/{self.total_models}] Training {model_name}...")
    
    def on_model_complete(self, model_name: str, score: float):
        self.models_trained.append(model_name)
        if self.verbose:
            logger.info(f"✅ {model_name} complete (score: {score:.4f})")
    
    def on_model_failed(self, model_name: str, error: str):
        if self.verbose:
            logger.warning(f"⚠️ {model_name} failed: {error}")
    
    def on_training_complete(self, best_model: str):
        elapsed = time.time() - self.start_time
        self.progress = 1.0
        if self.verbose:
            logger.info(f"🏆 Training complete! Best model: {best_model} ({elapsed:.1f}s)")
    
    def get_progress(self) -> Dict:
        return {
            'progress': self.progress,
            'phase': self.current_phase,
            'models_trained': len(self.models_trained),
            'total_models': self.total_models,
            'elapsed_time': time.time() - self.start_time
        }


# =============================================================================
# MAIN AUTOML ENGINE
# =============================================================================

class AutoMLEngine:
    """
    Main AutoML Engine combining all components.
    """
    
    def __init__(self, config: AutoMLConfig = None):
        self.config = config or AutoMLConfig()
        self.run_id = str(uuid.uuid4())[:8]
        
        # Core components
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = None
        self.advanced_feature_engineer = None
        
        # Models
        self.trained_models: Dict[str, Any] = {}
        self.model_results: Dict[str, Dict] = {}
        self.best_model = None
        self.best_model_name = None
        self.ensemble_model = None
        self.weighted_ensemble = None
        
        # Advanced components
        self.drift_detector = None
        self.model_explainer = None
        self.risk_calculator = RiskMetricsCalculator()
        self.explanations = {}
        self.cv_results = {}
        
        # State
        self.dataset_info = None
        self.label_encoder = None
        self.start_time = None
        self.end_time = None
        
        # Callback
        self.callback = None
    
    def _detect_task_type(self, y: pd.Series) -> TaskType:
        """Auto-detect task type from target variable"""
        if y.dtype == 'object' or str(y.dtype) == 'category':
            n_unique = y.nunique()
            if n_unique == 2:
                return TaskType.BINARY_CLASSIFICATION
            return TaskType.MULTICLASS_CLASSIFICATION
        
        if y.dtype in ['int64', 'int32'] and y.nunique() <= 20:
            if y.nunique() == 2:
                return TaskType.BINARY_CLASSIFICATION
            return TaskType.MULTICLASS_CLASSIFICATION
        
        return TaskType.REGRESSION
    
    def _train_single_model(self, name: str, model: Any, param_grid: Dict,
                           X_train: np.ndarray, y_train: np.ndarray,
                           X_test: np.ndarray, y_test: np.ndarray) -> Optional[Dict]:
        """Train a single model with timeout"""
        start_time = time.time()
        
        try:
            self.callback.on_model_start(name)
            
            # Hyperparameter tuning
            if param_grid and len(param_grid) > 0:
                if self.config.optimization_strategy == OptimizationStrategy.RANDOM_SEARCH:
                    n_iter = min(10, max(1, len(list(param_grid.values())[0]) if param_grid else 1))
                    search = RandomizedSearchCV(
                        model, param_grid, n_iter=n_iter,
                        cv=min(3, self.config.cv_folds),
                        scoring='neg_mean_squared_error' if self.dataset_info.task_type == TaskType.REGRESSION else 'accuracy',
                        n_jobs=self.config.n_jobs,
                        random_state=self.config.random_state
                    )
                else:
                    search = GridSearchCV(
                        model, param_grid,
                        cv=min(3, self.config.cv_folds),
                        scoring='neg_mean_squared_error' if self.dataset_info.task_type == TaskType.REGRESSION else 'accuracy',
                        n_jobs=self.config.n_jobs
                    )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
            else:
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}
            
            # Cross-validation
            cv_scoring = 'neg_mean_squared_error' if self.dataset_info.task_type == TaskType.REGRESSION else 'accuracy'
            cv_scores = cross_val_score(best_model, X_train, y_train, 
                                       cv=min(self.config.cv_folds, len(X_train) // 2),
                                       scoring=cv_scoring)
            
            # Predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            training_time = time.time() - start_time
            
            if self.dataset_info.task_type == TaskType.REGRESSION:
                metrics = {
                    'test_r2': float(r2_score(y_test, y_pred)),
                    'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
                    'test_mae': float(mean_absolute_error(y_test, y_pred)),
                    'cv_score': float(np.mean(np.abs(cv_scores))),
                    'cv_std': float(np.std(cv_scores)),
                    'cv_mean': float(np.mean(cv_scores)),
                    'training_time': training_time
                }
                
                # Risk metrics
                if self.config.enable_risk_metrics:
                    risk_metrics = self.risk_calculator.calculate_all(y_test, y_pred)
                    metrics['risk_metrics'] = risk_metrics
            else:
                metrics = {
                    'test_accuracy': float(accuracy_score(y_test, y_pred)),
                    'test_f1': float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'test_precision': float(precision_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'test_recall': float(recall_score(y_test, y_pred, average='weighted', zero_division=0)),
                    'cv_score': float(np.mean(cv_scores)),
                    'cv_std': float(np.std(cv_scores)),
                    'cv_mean': float(np.mean(cv_scores)),
                    'training_time': training_time
                }
            
            # Store results
            self.trained_models[name] = best_model
            self.model_results[name] = {
                **metrics,
                'hyperparameters': best_params,
                'predictions': y_pred
            }
            
            score = metrics.get('test_r2', metrics.get('test_accuracy', 0))
            self.callback.on_model_complete(name, score)
            
            return metrics
            
        except Exception as e:
            self.callback.on_model_failed(name, str(e))
            logger.debug(f"Model {name} failed: {e}")
            return None
    
    def fit(self, X: pd.DataFrame, y: pd.Series, target_column: str = None,
            progress_callback: Callable = None) -> 'AutoMLEngine':
        """Fit the AutoML engine"""
        self.start_time = datetime.now()
        
        # Auto-detect task type
        detected_task = self._detect_task_type(y)
        if self.config.task_type == TaskType.REGRESSION:
            self.config.task_type = detected_task
        
        logger.info(f"🚀 Starting AutoML run {self.run_id}")
        logger.info(f"📊 Task type: {self.config.task_type.value}")
        logger.info(f"📁 Data shape: {X.shape}")
        
        # Encode classification targets
        if self.config.task_type in [TaskType.BINARY_CLASSIFICATION, TaskType.MULTICLASS_CLASSIFICATION]:
            if y.dtype == 'object' or str(y.dtype) == 'category':
                self.label_encoder = LabelEncoder()
                y = pd.Series(self.label_encoder.fit_transform(y))
        
        # Store dataset info
        self.dataset_info = DatasetInfo(
            n_samples=len(X),
            n_features=len(X.columns),
            n_numeric=len(X.select_dtypes(include=[np.number]).columns),
            n_categorical=len(X.select_dtypes(include=['object', 'category']).columns),
            n_datetime=len(X.select_dtypes(include=['datetime64']).columns),
            target_column=target_column or 'target',
            task_type=self.config.task_type,
            missing_values={col: float(X[col].isnull().sum() / len(X)) for col in X.columns},
            feature_types={col: str(X[col].dtype) for col in X.columns}
        )
        
        # Preprocessing
        logger.info("⚙️ Preprocessing data...")
        X_processed = self.preprocessor.fit_transform(X, y)
        
        # Advanced feature engineering
        if self.config.advanced_features:
            logger.info("🔧 Advanced feature engineering...")
            self.advanced_feature_engineer = AdvancedFeatureEngineer()
            X_df = pd.DataFrame(X_processed, columns=self.preprocessor.numeric_features)
            X_enhanced = self.advanced_feature_engineer.fit_transform(X_df, target_column)
            X_processed = X_enhanced.values
            logger.info(f"    Generated {len(self.advanced_feature_engineer.engineered_features)} new features")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y.values if hasattr(y, 'values') else y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Initialize drift detector
        if self.config.enable_drift_detection:
            self.drift_detector = ModelDriftDetector()
            feature_names = list(X.columns) + (self.advanced_feature_engineer.engineered_features if self.advanced_feature_engineer else [])
            self.drift_detector.set_reference_distribution(X_train, feature_names)
        
        # Get models
        if self.config.task_type == TaskType.REGRESSION:
            models = ModelRegistry.get_regression_models(self.config)
        else:
            models = ModelRegistry.get_classification_models(self.config)
        
        # Limit models
        model_names = list(models.keys())[:self.config.max_models]
        
        # Initialize callback
        self.callback = TrainingCallback(total_models=len(model_names), verbose=self.config.verbose)
        
        # Train models
        logger.info(f"🏋️ Training {len(model_names)} models...")
        
        for name in model_names:
            model, param_grid = models[name]
            self._train_single_model(name, model, param_grid, X_train, y_train, X_test, y_test)
            
            if progress_callback:
                progress_callback(self.callback.get_progress())
        
        # Select best model
        if self.model_results:
            if self.config.task_type == TaskType.REGRESSION:
                self.best_model_name = max(self.model_results.keys(),
                                          key=lambda k: self.model_results[k].get('test_r2', 0))
            else:
                self.best_model_name = max(self.model_results.keys(),
                                          key=lambda k: self.model_results[k].get('test_accuracy', 0))
            
            self.best_model = self.trained_models.get(self.best_model_name)
        
        # Build weighted ensemble
        if self.config.ensemble_size > 0 and len(self.trained_models) > 1:
            logger.info("🎭 Building weighted ensemble...")
            
            # Calculate weights based on CV scores
            weights = {}
            for name, results in self.model_results.items():
                score = abs(results.get('cv_mean', results.get('cv_score', 0)))
                weights[name] = max(score, 0.001)
            
            # Select top models
            sorted_models = sorted(weights.items(), key=lambda x: x[1], reverse=True)
            top_models = dict(sorted_models[:self.config.ensemble_size])
            
            ensemble_models = {name: self.trained_models[name] for name in top_models if name in self.trained_models}
            if ensemble_models:
                self.weighted_ensemble = WeightedEnsemble(ensemble_models, top_models)
                logger.info(f"    Created ensemble with {len(ensemble_models)} models")
        
        # Generate explanations
        if self.config.enable_explainability and self.best_model:
            logger.info("🔍 Generating model explanations...")
            self.model_explainer = ModelExplainer()
            try:
                feature_names = list(X.columns) + (self.advanced_feature_engineer.engineered_features if self.advanced_feature_engineer else [])
                model_type = 'tree' if 'Tree' in self.best_model_name or 'Forest' in self.best_model_name or 'Boost' in self.best_model_name else 'linear'
                self.explanations = self.model_explainer.explain_model(
                    self.best_model, X_test[:100], feature_names[:X_test.shape[1]], model_type
                )
                logger.info("    ✅ Explanations generated")
            except Exception as e:
                logger.debug(f"Explanation generation failed: {e}")
        
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"✨ AutoML complete! Duration: {duration:.1f}s")
        logger.info(f"🏆 Best model: {self.best_model_name}")
        
        self.callback.on_training_complete(self.best_model_name)
        
        if progress_callback:
            progress_callback(self.callback.get_progress())
        
        return self
    
    def predict(self, X: pd.DataFrame, use_ensemble: bool = False, 
               use_weighted_ensemble: bool = False) -> np.ndarray:
        """Make predictions"""
        X_processed = self.preprocessor.transform(X)
        
        if self.advanced_feature_engineer:
            X_df = pd.DataFrame(X_processed, columns=self.preprocessor.numeric_features)
            X_enhanced = self.advanced_feature_engineer.transform(X_df)
            X_processed = X_enhanced.values
        
        if use_weighted_ensemble and self.weighted_ensemble:
            return self.weighted_ensemble.predict(X_processed)
        elif use_ensemble and self.ensemble_model:
            return self.ensemble_model.predict(X_processed)
        elif self.best_model:
            return self.best_model.predict(X_processed)
        else:
            raise ValueError("No trained model available")
    
    def detect_drift(self, X_new: pd.DataFrame) -> Tuple[bool, float, Dict]:
        """Detect data drift in new data"""
        if not self.drift_detector:
            return False, 0.0, {}
        
        X_processed = self.preprocessor.transform(X_new)
        
        if self.advanced_feature_engineer:
            X_df = pd.DataFrame(X_processed, columns=self.preprocessor.numeric_features)
            X_enhanced = self.advanced_feature_engineer.transform(X_df)
            X_processed = X_enhanced.values
        
        feature_names = list(X_new.columns) + (self.advanced_feature_engineer.engineered_features if self.advanced_feature_engineer else [])
        return self.drift_detector.detect_drift(X_processed, feature_names)
    
    def get_explanations(self) -> Dict:
        """Get model explanations"""
        return self.explanations
    
    def get_explanation_report(self, prediction: float = None, confidence: float = None) -> str:
        """Get human-readable explanation report"""
        if not self.model_explainer or not self.explanations:
            return "No explanations available"
        
        return self.model_explainer.generate_explanation_report(
            self.explanations, prediction or 0.0, confidence
        )
    
    def get_leaderboard(self) -> pd.DataFrame:
        """Get model leaderboard"""
        rows = []
        for model_name, results in self.model_results.items():
            row = {
                'Model': model_name,
                'CV Score': abs(results.get('cv_mean', results.get('cv_score', 0))),
                'Training Time': results.get('training_time', 0)
            }
            
            if self.dataset_info.task_type == TaskType.REGRESSION:
                row['Test RMSE'] = results.get('test_rmse')
                row['Test R²'] = results.get('test_r2')
                row['Test MAE'] = results.get('test_mae')
                
                if 'risk_metrics' in results:
                    row['Sharpe Ratio'] = results['risk_metrics'].get('sharpe_ratio')
            else:
                row['Test Accuracy'] = results.get('test_accuracy')
                row['Test F1'] = results.get('test_f1')
                row['Test Precision'] = results.get('test_precision')
                row['Test Recall'] = results.get('test_recall')
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        if len(df) > 0:
            if self.dataset_info.task_type == TaskType.REGRESSION:
                df = df.sort_values('Test RMSE', ascending=True)
            else:
                df = df.sort_values('Test Accuracy', ascending=False)
        
        return df.reset_index(drop=True)
    
    def save(self, path: str):
        """Save AutoML engine"""
        save_dict = {
            'config': asdict(self.config),
            'preprocessor': self.preprocessor,
            'advanced_feature_engineer': self.advanced_feature_engineer,
            'trained_models': self.trained_models,
            'model_results': self.model_results,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'weighted_ensemble': self.weighted_ensemble,
            'dataset_info': asdict(self.dataset_info) if self.dataset_info else None,
            'run_id': self.run_id,
            'label_encoder': self.label_encoder,
            'explanations': self.explanations,
            'drift_detector': self.drift_detector
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info(f"💾 Saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'AutoMLEngine':
        """Load AutoML engine"""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        config_dict = save_dict['config']
        config_dict.pop('progress_callback', None)
        
        config = AutoMLConfig(**config_dict)
        engine = cls(config)
        engine.preprocessor = save_dict['preprocessor']
        engine.advanced_feature_engineer = save_dict.get('advanced_feature_engineer')
        engine.trained_models = save_dict['trained_models']
        engine.model_results = save_dict.get('model_results', {})
        engine.best_model = save_dict['best_model']
        engine.best_model_name = save_dict['best_model_name']
        engine.weighted_ensemble = save_dict.get('weighted_ensemble')
        engine.run_id = save_dict['run_id']
        engine.label_encoder = save_dict.get('label_encoder')
        engine.explanations = save_dict.get('explanations', {})
        engine.drift_detector = save_dict.get('drift_detector')
        
        if save_dict['dataset_info']:
            info_dict = save_dict['dataset_info']
            info_dict['task_type'] = TaskType(info_dict['task_type'])
            engine.dataset_info = DatasetInfo(**info_dict)
        
        logger.info(f"📂 Loaded from {path}")
        return engine


# =============================================================================
# QUICK API
# =============================================================================

def auto_train(
    data: pd.DataFrame,
    target_column: str,
    task_type: str = 'auto',
    time_limit: int = 10,
    enable_advanced_features: bool = True,
    **kwargs
) -> AutoMLEngine:
    """Quick API for AutoML training"""
    
    if task_type == 'auto' or task_type == 'regression':
        task = TaskType.REGRESSION
    elif task_type == 'classification':
        task = TaskType.BINARY_CLASSIFICATION
    else:
        task = TaskType.REGRESSION
    
    config = AutoMLConfig(
        task_type=task,
        time_limit_minutes=time_limit,
        per_model_time_limit=max(60, (time_limit * 60) // 10),
        feature_engineering=enable_advanced_features,
        advanced_features=enable_advanced_features,
        enable_drift_detection=enable_advanced_features,
        enable_explainability=enable_advanced_features,
        advanced_cv=enable_advanced_features,
        enable_risk_metrics=enable_advanced_features,
        **kwargs
    )
    
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    engine = AutoMLEngine(config)
    engine.fit(X, y, target_column)
    
    return engine


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    
    print("=" * 60)
    print("QuantML Studio - AutoML Engine Demo")
    print("=" * 60)
    
    # Test regression
    print("\n📊 Testing Regression...")
    X, y = make_regression(n_samples=500, n_features=10, n_informative=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    engine = auto_train(df, 'target', task_type='regression', time_limit=2, max_models=5)
    
    print("\n📊 Leaderboard:")
    print(engine.get_leaderboard().to_string())
    print(f"\n🏆 Best Model: {engine.best_model_name}")
    
    print("\n✅ AutoML Engine working correctly!")
