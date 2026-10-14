"""
PyTorch Neural Network Model for Differential-Drive Obstacle Avoidance

A compact Multi-Layer Perceptron (MLP) designed for low-latency ROS 2 inference.
Takes normalized LiDAR sector distances as input and predicts safe linear and
angular velocity commands.
"""

import os
from typing import Tuple, Union
import numpy as np
import torch
import torch.nn as nn


class ObstacleAvoidanceMLP(nn.Module):
    """
    Compact MLP for mapping normalized LiDAR sector inputs to (v, omega) control commands.
    """

    def __init__(
        self,
        input_size: int = 36,
        hidden_dim1: int = 64,
        hidden_dim2: int = 32,
        max_linear_vel: float = 0.40,
        max_angular_vel: float = 1.20,
        dropout_rate: float = 0.05
    ):
        """
        Args:
            input_size: Dimension of normalized LiDAR feature vector (default: 36 sectors).
            hidden_dim1: Number of neurons in first hidden layer.
            hidden_dim2: Number of neurons in second hidden layer.
            max_linear_vel: Maximum forward velocity in m/s.
            max_angular_vel: Maximum rotation rate in rad/s.
            dropout_rate: Light dropout to prevent overfitting on repetitive trajectories.
        """
        super().__init__()

        self.input_size = input_size
        self.max_linear_vel = float(max_linear_vel)
        self.max_angular_vel = float(max_angular_vel)

        # Feature extraction layers
        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU()
        )

        # Output head producing unconstrained logits [v_raw, w_raw]
        self.head = nn.Linear(hidden_dim2, 2)

        # Initialize weights with standard Xavier normal for smooth initial gradients
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (batch_size, input_size) with values in [0.0, 1.0].
        Returns:
            Tensor of shape (batch_size, 2) where:
                col 0: linear velocity v in [0.0, max_linear_vel] (via Sigmoid)
                col 1: angular velocity omega in [-max_angular_vel, max_angular_vel] (via Tanh)
        """
        features = self.backbone(x)
        raw_outputs = self.head(features)

        # Column 0: Linear speed (strictly forward >= 0 for obstacle avoidance)
        v_unscaled = torch.sigmoid(raw_outputs[:, 0:1])
        v = v_unscaled * self.max_linear_vel

        # Column 1: Angular steering (left/right symmetric)
        w_unscaled = torch.tanh(raw_outputs[:, 1:2])
        w = w_unscaled * self.max_angular_vel

        return torch.cat([v, w], dim=1)

    @torch.no_grad()
    def predict_single(
        self,
        features: Union[np.ndarray, torch.Tensor],
        device: str = 'cpu'
    ) -> Tuple[float, float]:
        """
        Runs real-time inference for a single LiDAR observation.
        Designed for minimal overhead inside a ROS 2 callback.

        Args:
            features: 1D array or tensor of length input_size.
            device: 'cpu' or 'cuda'.

        Returns:
            (linear_velocity, angular_velocity) as standard Python floats.
        """
        self.eval()

        if isinstance(features, np.ndarray):
            tensor = torch.from_numpy(features).float()
        else:
            tensor = features.float()

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)  # Shape: (1, input_size)

        tensor = tensor.to(device)
        output = self.forward(tensor).squeeze(0)

        linear_v = float(output[0].item())
        angular_w = float(output[1].item())

        return linear_v, angular_w

    def save_checkpoint(self, filepath: str):
        """Saves model weights along with architectural hyperparameters."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            'state_dict': self.state_dict(),
            'hyperparameters': {
                'input_size': self.input_size,
                'max_linear_vel': self.max_linear_vel,
                'max_angular_vel': self.max_angular_vel,
            }
        }
        torch.save(payload, filepath)

    @classmethod
    def load_from_checkpoint(
        cls,
        filepath: str,
        device: str = 'cpu'
    ) -> 'ObstacleAvoidanceMLP':
        """Instantiates and loads an ObstacleAvoidanceMLP from a saved checkpoint."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        params = checkpoint.get('hyperparameters', {})
        model = cls(
            input_size=params.get('input_size', 36),
            max_linear_vel=params.get('max_linear_vel', 0.40),
            max_angular_vel=params.get('max_angular_vel', 1.20)
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        model.eval()
        return model
