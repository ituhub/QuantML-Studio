"""
QuantML Studio - Neural Network Models v4.0
=============================================
Advanced deep learning models including:
- Advanced Transformer with multi-head attention
- CNN-LSTM hybrid with attention mechanism
- Temporal Convolutional Networks (TCN)
- N-BEATS for time series
- LSTM-GRU Ensemble
- MLP with residual connections
"""

import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch.optim import Adam, AdamW, SGD
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Neural network models disabled.")


if TORCH_AVAILABLE:
    # =============================================================================
    # UTILITY MODULES
    # =============================================================================

    class PositionalEncoding(nn.Module):
        """Positional encoding for transformer models"""

        def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            self.register_buffer('pe', pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x + self.pe[:, :x.size(1)]
            return self.dropout(x)


    # =============================================================================
    # TRANSFORMER MODEL
    # =============================================================================

    class TransformerModel(nn.Module):
        """Advanced Transformer model for tabular/time series data"""

        def __init__(
            self,
            n_features: int,
            d_model: int = 128,
            n_heads: int = 8,
            n_layers: int = 4,
            d_ff: int = 512,
            seq_len: int = 60,
            output_dim: int = 1,
            dropout: float = 0.1,
            task_type: str = 'regression'
        ):
            super().__init__()
            self.task_type = task_type
            self.seq_len = seq_len

            self.input_projection = nn.Linear(n_features, d_model)
            self.pos_encoding = PositionalEncoding(d_model, seq_len, dropout)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

            self.layer_norm = nn.LayerNorm(d_model)
            self.output = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )

            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)

            x = self.input_projection(x)
            x = self.pos_encoding(x)
            x = self.transformer(x)
            x = self.layer_norm(x)

            if x.size(1) > 1:
                x = x.mean(dim=1)
            else:
                x = x.squeeze(1)

            return self.output(x)


    # =============================================================================
    # CNN-LSTM MODEL
    # =============================================================================

    class CNNLSTMModel(nn.Module):
        """CNN-LSTM hybrid model with attention"""

        def __init__(
            self,
            n_features: int,
            seq_len: int = 60,
            cnn_filters: List[int] = [64, 128, 64],
            lstm_hidden: int = 100,
            lstm_layers: int = 2,
            output_dim: int = 1,
            dropout: float = 0.2,
            task_type: str = 'regression'
        ):
            super().__init__()
            self.task_type = task_type

            self.conv_layers = nn.ModuleList()
            self.bn_layers = nn.ModuleList()

            in_channels = n_features
            for out_channels in cnn_filters:
                self.conv_layers.append(
                    nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
                )
                self.bn_layers.append(nn.BatchNorm1d(out_channels))
                in_channels = out_channels

            self.cnn_dropout = nn.Dropout(dropout)

            self.lstm = nn.LSTM(
                cnn_filters[-1],
                lstm_hidden,
                num_layers=lstm_layers,
                batch_first=True,
                dropout=dropout if lstm_layers > 1 else 0,
                bidirectional=True
            )

            self.attention = nn.MultiheadAttention(
                lstm_hidden * 2,
                num_heads=4,
                batch_first=True,
                dropout=dropout
            )

            self.fc = nn.Sequential(
                nn.Linear(lstm_hidden * 2, lstm_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden, lstm_hidden // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_hidden // 2, output_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)

            x = x.transpose(1, 2)

            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = F.relu(bn(conv(x)))

            x = self.cnn_dropout(x)
            x = x.transpose(1, 2)

            lstm_out, _ = self.lstm(x)
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

            if attn_out.size(1) > 1:
                x = attn_out.mean(dim=1)
            else:
                x = attn_out.squeeze(1)

            return self.fc(x)


    # =============================================================================
    # TEMPORAL CONVOLUTIONAL NETWORK (TCN)
    # =============================================================================

    class TemporalConvNet(nn.Module):
        """Temporal Convolutional Network (TCN)"""

        def __init__(
            self,
            n_features: int,
            num_channels: List[int] = [64, 128, 256, 128],
            kernel_size: int = 3,
            output_dim: int = 1,
            dropout: float = 0.2,
            task_type: str = 'regression'
        ):
            super().__init__()
            self.task_type = task_type

            layers = []
            num_levels = len(num_channels)

            for i in range(num_levels):
                dilation = 2 ** i
                in_channels = n_features if i == 0 else num_channels[i - 1]
                out_channels = num_channels[i]
                
                padding = (kernel_size - 1) * dilation // 2
                
                layers.extend([
                    nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=padding, dilation=dilation),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ])

            self.tcn = nn.Sequential(*layers)
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Sequential(
                nn.Linear(num_channels[-1], num_channels[-1] // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(num_channels[-1] // 2, output_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)

            x = x.transpose(1, 2)
            x = self.tcn(x)
            x = self.global_pool(x).squeeze(-1)
            return self.fc(x)


    # =============================================================================
    # MLP WITH RESIDUAL CONNECTIONS
    # =============================================================================

    class ResidualBlock(nn.Module):
        """Residual block for MLP"""
        
        def __init__(self, dim: int, dropout: float = 0.1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 2, dim),
                nn.Dropout(dropout)
            )
            self.norm = nn.LayerNorm(dim)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(x + self.net(x))


    class MLP(nn.Module):
        """Multi-layer perceptron with residual connections"""

        def __init__(
            self,
            n_features: int,
            hidden_dims: List[int] = [256, 128, 64],
            output_dim: int = 1,
            dropout: float = 0.2,
            task_type: str = 'regression',
            use_residual: bool = True
        ):
            super().__init__()
            self.task_type = task_type
            self.use_residual = use_residual

            # Input layer
            self.input_layer = nn.Linear(n_features, hidden_dims[0])
            self.input_norm = nn.LayerNorm(hidden_dims[0])
            
            # Hidden layers
            self.hidden_layers = nn.ModuleList()
            for i in range(len(hidden_dims) - 1):
                if use_residual and hidden_dims[i] == hidden_dims[i + 1]:
                    self.hidden_layers.append(ResidualBlock(hidden_dims[i], dropout))
                else:
                    self.hidden_layers.append(nn.Sequential(
                        nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                        nn.LayerNorm(hidden_dims[i + 1]),
                        nn.GELU(),
                        nn.Dropout(dropout)
                    ))

            # Output layer
            self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
            
            self._init_weights()

        def _init_weights(self):
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 3:
                x = x.reshape(x.size(0), -1)
            
            x = F.gelu(self.input_norm(self.input_layer(x)))
            
            for layer in self.hidden_layers:
                x = layer(x)
            
            return self.output_layer(x)


    # =============================================================================
    # LSTM-GRU ENSEMBLE
    # =============================================================================

    class LSTMGRUEnsemble(nn.Module):
        """Ensemble of LSTM and GRU networks"""

        def __init__(
            self,
            n_features: int,
            hidden_size: int = 128,
            num_layers: int = 2,
            output_dim: int = 1,
            dropout: float = 0.2,
            task_type: str = 'regression'
        ):
            super().__init__()
            self.task_type = task_type

            # LSTM branch
            self.lstm = nn.LSTM(
                n_features, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            # GRU branch
            self.gru = nn.GRU(
                n_features, hidden_size, num_layers,
                batch_first=True, dropout=dropout if num_layers > 1 else 0,
                bidirectional=True
            )
            
            # Attention for combining
            self.attention = nn.MultiheadAttention(
                hidden_size * 2, num_heads=4, batch_first=True, dropout=dropout
            )
            
            # Output layer
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 4, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, output_dim)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)

            # LSTM path
            lstm_out, _ = self.lstm(x)
            
            # GRU path
            gru_out, _ = self.gru(x)
            
            # Apply attention to LSTM output
            lstm_attn, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Combine outputs
            if lstm_attn.size(1) > 1:
                lstm_pooled = lstm_attn.mean(dim=1)
                gru_pooled = gru_out.mean(dim=1)
            else:
                lstm_pooled = lstm_attn.squeeze(1)
                gru_pooled = gru_out.squeeze(1)
            
            combined = torch.cat([lstm_pooled, gru_pooled], dim=-1)
            
            return self.fc(combined)


    # =============================================================================
    # TRAINING CONFIG AND TRAINER
    # =============================================================================

    @dataclass
    class NNTrainingConfig:
        """Configuration for neural network training"""
        epochs: int = 100
        batch_size: int = 32
        learning_rate: float = 0.001
        weight_decay: float = 1e-5
        patience: int = 15
        min_delta: float = 1e-4
        scheduler: str = 'cosine'  # 'plateau', 'cosine', 'onecycle'
        optimizer: str = 'adamw'
        device: str = 'auto'
        verbose: bool = True
        gradient_clip: float = 1.0


    class NeuralNetworkTrainer:
        """Trainer for neural network models"""

        def __init__(self, model: nn.Module, config: NNTrainingConfig = None):
            self.config = config or NNTrainingConfig()
            self.model = model
            
            # Device setup
            if self.config.device == 'auto':
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            else:
                self.device = torch.device(self.config.device)
            
            self.model.to(self.device)
            
            # Loss function
            if hasattr(model, 'task_type') and model.task_type in ['classification', 'binary_classification']:
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.MSELoss()
            
            # Optimizer
            if self.config.optimizer == 'adamw':
                self.optimizer = AdamW(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            elif self.config.optimizer == 'adam':
                self.optimizer = Adam(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    weight_decay=self.config.weight_decay
                )
            else:
                self.optimizer = SGD(
                    model.parameters(),
                    lr=self.config.learning_rate,
                    momentum=0.9,
                    weight_decay=self.config.weight_decay
                )
            
            # Scheduler
            self.scheduler = None
            self.history = {'train_loss': [], 'val_loss': []}
            
            self._init_scheduler()

        def _init_scheduler(self):
            """Initialize learning rate scheduler"""
            if self.config.scheduler == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer, mode='min', factor=0.5, patience=5
                )
            elif self.config.scheduler == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=self.config.epochs
                )

        def fit(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None
        ) -> Dict[str, List[float]]:
            """Train the model"""
            X_train_t = torch.FloatTensor(X_train).to(self.device)
            
            # Handle targets based on loss function
            if isinstance(self.criterion, nn.CrossEntropyLoss):
                y_train_t = torch.LongTensor(y_train).to(self.device)
                if y_train_t.dim() > 1:
                    y_train_t = y_train_t.squeeze()
            else:
                y_train_t = torch.FloatTensor(y_train).to(self.device)
                if y_train_t.dim() == 1:
                    y_train_t = y_train_t.unsqueeze(1)

            train_dataset = TensorDataset(X_train_t, y_train_t)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            val_loader = None
            if X_val is not None and y_val is not None:
                X_val_t = torch.FloatTensor(X_val).to(self.device)
                
                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    y_val_t = torch.LongTensor(y_val).to(self.device)
                    if y_val_t.dim() > 1:
                        y_val_t = y_val_t.squeeze()
                else:
                    y_val_t = torch.FloatTensor(y_val).to(self.device)
                    if y_val_t.dim() == 1:
                        y_val_t = y_val_t.unsqueeze(1)
                        
                val_dataset = TensorDataset(X_val_t, y_val_t)
                val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size)

            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0

                for batch_X, batch_y in train_loader:
                    self.optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = self.criterion(output, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip
                    )
                    
                    self.optimizer.step()
                    train_loss += loss.item()

                train_loss /= len(train_loader)
                self.history['train_loss'].append(train_loss)

                # Validation phase
                val_loss = 0.0
                if val_loader:
                    self.model.eval()
                    with torch.no_grad():
                        for batch_X, batch_y in val_loader:
                            output = self.model(batch_X)
                            loss = self.criterion(output, batch_y)
                            val_loss += loss.item()
                    val_loss /= len(val_loader)
                    self.history['val_loss'].append(val_loss)

                    # Scheduler step
                    if self.scheduler:
                        if isinstance(self.scheduler, ReduceLROnPlateau):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()

                    # Early stopping
                    if val_loss < best_val_loss - self.config.min_delta:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1

                    if patience_counter >= self.config.patience:
                        if self.config.verbose:
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                        break

                # Logging
                if self.config.verbose and (epoch + 1) % 10 == 0:
                    msg = f"Epoch {epoch + 1}/{self.config.epochs} - Train Loss: {train_loss:.6f}"
                    if val_loader:
                        msg += f" - Val Loss: {val_loss:.6f}"
                    logger.info(msg)

            return self.history

        def predict(self, X: np.ndarray) -> np.ndarray:
            """Make predictions"""
            self.model.eval()
            X_t = torch.FloatTensor(X).to(self.device)

            with torch.no_grad():
                predictions = self.model(X_t)

            return predictions.cpu().numpy()


    # =============================================================================
    # MODEL FACTORY
    # =============================================================================

    class NeuralNetworkFactory:
        """Factory for creating neural network models"""

        @staticmethod
        def get_available_models() -> List[str]:
            """Get list of available models"""
            return ['transformer', 'cnn_lstm', 'tcn', 'mlp', 'lstm_gru']

        @staticmethod
        def create_model(
            model_type: str,
            n_features: int,
            output_dim: int = 1,
            task_type: str = 'regression',
            **kwargs
        ) -> nn.Module:
            """Create a neural network model"""

            model_map = {
                'transformer': TransformerModel,
                'cnn_lstm': CNNLSTMModel,
                'tcn': TemporalConvNet,
                'mlp': MLP,
                'lstm_gru': LSTMGRUEnsemble
            }

            if model_type not in model_map:
                raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_map.keys())}")

            model_class = model_map[model_type]

            # Build params based on model type
            if model_type == 'transformer':
                return model_class(
                    n_features=n_features,
                    output_dim=output_dim,
                    task_type=task_type,
                    d_model=kwargs.get('d_model', 128),
                    n_heads=kwargs.get('n_heads', 8),
                    n_layers=kwargs.get('n_layers', 4),
                    dropout=kwargs.get('dropout', 0.1)
                )
            elif model_type == 'cnn_lstm':
                return model_class(
                    n_features=n_features,
                    output_dim=output_dim,
                    task_type=task_type,
                    cnn_filters=kwargs.get('cnn_filters', [64, 128, 64]),
                    lstm_hidden=kwargs.get('lstm_hidden', 100),
                    dropout=kwargs.get('dropout', 0.2)
                )
            elif model_type == 'tcn':
                return model_class(
                    n_features=n_features,
                    output_dim=output_dim,
                    task_type=task_type,
                    num_channels=kwargs.get('num_channels', [64, 128, 256, 128]),
                    dropout=kwargs.get('dropout', 0.2)
                )
            elif model_type == 'lstm_gru':
                return model_class(
                    n_features=n_features,
                    output_dim=output_dim,
                    task_type=task_type,
                    hidden_size=kwargs.get('hidden_size', 128),
                    num_layers=kwargs.get('num_layers', 2),
                    dropout=kwargs.get('dropout', 0.2)
                )
            else:  # mlp
                return model_class(
                    n_features=n_features,
                    output_dim=output_dim,
                    task_type=task_type,
                    hidden_dims=kwargs.get('hidden_dims', [256, 128, 64]),
                    dropout=kwargs.get('dropout', 0.2)
                )

else:
    # Dummy classes when PyTorch is not available
    @dataclass
    class NNTrainingConfig:
        """Configuration for neural network training (dummy)"""
        epochs: int = 100
        batch_size: int = 32
        learning_rate: float = 0.001
        weight_decay: float = 1e-5
        patience: int = 15
        min_delta: float = 1e-4
        scheduler: str = 'cosine'
        optimizer: str = 'adamw'
        device: str = 'auto'
        verbose: bool = True
        gradient_clip: float = 1.0
    
    class NeuralNetworkFactory:
        @staticmethod
        def get_available_models():
            return []
        
        @staticmethod
        def create_model(*args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")

    class NeuralNetworkTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch not available. Install with: pip install torch")


__all__ = [
    'TORCH_AVAILABLE',
    'NeuralNetworkFactory',
    'NeuralNetworkTrainer',
    'NNTrainingConfig'
]
