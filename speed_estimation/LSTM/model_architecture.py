"""
Model Architecture for Dynamic Dilemma Zone Prediction

Implements LSTM and CNN sequence encoders with interpretable linear output layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .config import (
    MODEL_TYPE,
    LSTM_HIDDEN_SIZE,
    LSTM_NUM_LAYERS,
    LSTM_DROPOUT,
    CNN_NUM_FILTERS,
    CNN_KERNEL_SIZE,
    CNN_STRIDE,
    FEATURE_DIM,
    SEQUENCE_LENGTH
)


class LSTMEncoder(nn.Module):
    """
    LSTM sequence encoder for temporal feature extraction.
    """
    
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        bidirectional: bool = False
    ):
        super(LSTMEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension depends on bidirectional
        self.output_dim = hidden_size * 2 if bidirectional else hidden_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM encoder.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Hidden representation of shape (batch_size, output_dim)
        """
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state from all layers
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            forward_hidden = hidden[-2]  # Last forward layer
            backward_hidden = hidden[-1]  # Last backward layer
            output = torch.cat([forward_hidden, backward_hidden], dim=1)
        else:
            # Use last layer's hidden state
            output = hidden[-1]
        
        return output


class CNNEncoder(nn.Module):
    """
    Temporal CNN encoder (1D Convolution) for sequence feature extraction.
    """
    
    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        num_filters: int = CNN_NUM_FILTERS,
        kernel_size: int = CNN_KERNEL_SIZE,
        stride: int = CNN_STRIDE,
        num_layers: int = 2
    ):
        super(CNNEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        
        # Build CNN layers
        layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2  # Same padding
                )
            )
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(num_filters))
            in_channels = num_filters
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.output_dim = num_filters
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Hidden representation of shape (batch_size, output_dim)
        """
        # Convert to (batch_size, input_dim, sequence_length) for Conv1d
        x = x.transpose(1, 2)
        
        # Apply CNN layers
        x = self.conv_layers(x)
        
        # Global average pooling
        x = self.global_pool(x)
        
        # Flatten: (batch_size, num_filters, 1) -> (batch_size, num_filters)
        x = x.squeeze(-1)
        
        return x


class DilemmaZoneModel(nn.Module):
    """
    Main model for STOP/GO prediction with interpretable linear output layer.
    
    Architecture:
    - Sequence encoder (LSTM or CNN)
    - Linear output layer with sigmoid activation
    - P(stop) = sigmoid(wᵀz + b)
    """
    
    def __init__(
        self,
        model_type: str = MODEL_TYPE,
        input_dim: int = FEATURE_DIM,
        sequence_length: int = SEQUENCE_LENGTH,
        hidden_size: int = LSTM_HIDDEN_SIZE,
        num_layers: int = LSTM_NUM_LAYERS,
        dropout: float = LSTM_DROPOUT,
        cnn_num_filters: int = CNN_NUM_FILTERS,
        cnn_kernel_size: int = CNN_KERNEL_SIZE
    ):
        super(DilemmaZoneModel, self).__init__()
        
        self.model_type = model_type.lower()
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        # Choose encoder
        if self.model_type == "lstm":
            self.encoder = LSTMEncoder(
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
            encoder_output_dim = self.encoder.output_dim
        elif self.model_type == "cnn":
            self.encoder = CNNEncoder(
                input_dim=input_dim,
                num_filters=cnn_num_filters,
                kernel_size=cnn_kernel_size
            )
            encoder_output_dim = self.encoder.output_dim
        else:
            raise ValueError(f"Unknown model type: {model_type}. Choose 'lstm' or 'cnn'")
        
        # Interpretable linear output layer
        # P(stop) = sigmoid(wᵀz + b)
        self.output_layer = nn.Linear(encoder_output_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """
        Initialize model weights using Xavier uniform for linear layers.
        """
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            Probability tensor of shape (batch_size, 1) representing P(stop)
        """
        # Encode sequence
        z = self.encoder(x)  # (batch_size, encoder_output_dim)
        
        # Linear transformation
        logits = self.output_layer(z)  # (batch_size, 1)
        
        # Sigmoid activation for binary classification
        prob = torch.sigmoid(logits)  # P(stop)
        
        return prob
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Predict binary class (0=GO, 1=STOP) given input.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            threshold: Probability threshold for classification
            
        Returns:
            Binary predictions of shape (batch_size, 1)
        """
        with torch.no_grad():
            prob = self.forward(x)
            predictions = (prob >= threshold).long()
        return predictions
    
    def get_feature_importance_weights(self) -> torch.Tensor:
        """
        Get the weights of the linear output layer for interpretability.
        
        Returns:
            Weight tensor of shape (encoder_output_dim,)
        """
        return self.output_layer.weight.squeeze().detach()


def create_model(
    model_type: str = MODEL_TYPE,
    input_dim: int = FEATURE_DIM,
    sequence_length: int = SEQUENCE_LENGTH,
    **kwargs
) -> DilemmaZoneModel:
    """
    Factory function to create a model instance.
    
    Args:
        model_type: Type of encoder ('lstm' or 'cnn')
        input_dim: Dimension of input features
        sequence_length: Length of input sequences
        **kwargs: Additional arguments passed to model constructor
        
    Returns:
        DilemmaZoneModel instance
    """
    model = DilemmaZoneModel(
        model_type=model_type,
        input_dim=input_dim,
        sequence_length=sequence_length,
        **kwargs
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, input_shape: tuple = (1, SEQUENCE_LENGTH, FEATURE_DIM)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Shape of input tensor (batch_size, sequence_length, feature_dim)
    """
    print("=" * 80)
    print("Model Architecture Summary")
    print("=" * 80)
    print(f"Model Type: {model.model_type.upper()}")
    print(f"Input Shape: {input_shape}")
    print(f"Total Parameters: {count_parameters(model):,}")
    print("\nModel Structure:")
    print(model)
    print("\n" + "=" * 80)
    
    # Test forward pass
    try:
        dummy_input = torch.randn(input_shape)
        with torch.no_grad():
            output = model(dummy_input)
        print(f"Output Shape: {output.shape}")
        print(f"Output Range: [{output.min().item():.4f}, {output.max().item():.4f}]")
    except Exception as e:
        print(f"Error in forward pass: {e}")
    
    print("=" * 80)

