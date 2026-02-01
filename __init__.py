"""
QuantML Studio - Core Module
============================
Enterprise-grade AutoML and ML Platform
"""

from .automl_engine import (
    AutoMLEngine,
    AutoMLConfig,
    DataPreprocessor,
    AdvancedFeatureEngineer,
    ModelRegistry,
    ModelDriftDetector,
    ModelExplainer,
    RiskMetricsCalculator,
    WeightedEnsemble,
    TaskType,
    OptimizationStrategy,
    ScalerType,
    ImputerType,
    DatasetInfo,
    TrainingResult,
    auto_train,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE,
    TORCH_AVAILABLE,
    SHAP_AVAILABLE
)

from .neural_networks import (
    NeuralNetworkFactory,
    NeuralNetworkTrainer,
    NNTrainingConfig,
    TORCH_AVAILABLE as NN_TORCH_AVAILABLE
)

__all__ = [
    # Engine
    'AutoMLEngine',
    'AutoMLConfig',
    'DataPreprocessor',
    'AdvancedFeatureEngineer',
    'ModelRegistry',
    'ModelDriftDetector',
    'ModelExplainer',
    'RiskMetricsCalculator',
    'WeightedEnsemble',
    
    # Enums
    'TaskType',
    'OptimizationStrategy',
    'ScalerType',
    'ImputerType',
    
    # Data classes
    'DatasetInfo',
    'TrainingResult',
    
    # Quick API
    'auto_train',
    
    # Neural Networks
    'NeuralNetworkFactory',
    'NeuralNetworkTrainer',
    'NNTrainingConfig',
    
    # Availability flags
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE',
    'CATBOOST_AVAILABLE',
    'TORCH_AVAILABLE',
    'SHAP_AVAILABLE',
    'NN_TORCH_AVAILABLE'
]

__version__ = '4.0.0'
__author__ = 'QuantML Studio Team'
