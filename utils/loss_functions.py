import torch
import torch.nn as nn

class LossFactory:
    @staticmethod
    def get_loss(loss_type: str, **kwargs):
        losses = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'huber': nn.HuberLoss(delta=kwargs.get('delta', 1.0)),
            'smooth_l1': nn.SmoothL1Loss(),
            'weighted_mse': WeightedMSELoss(weights=kwargs.get('weights'))
        }
        
        if loss_type not in losses:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        return losses[loss_type]

class WeightedMSELoss(nn.Module):
    """Weighted MSE loss that assigns different weights to different value ranges."""
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
    
    def forward(self, pred, target):
        if self.weights is None:
            return nn.functional.mse_loss(pred, target)
        else:
            # Dynamically adjust weights based on target values.
            weight_tensor = torch.ones_like(target)
            for threshold, weight in self.weights.items():
                mask = target > threshold
                weight_tensor[mask] = weight
            return torch.mean(weight_tensor * (pred - target) ** 2)
