import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import copy

class SWAG(nn.Module):
    """
    Stochastic Weight Averaging Gaussian (SWAG) for Bayesian Deep Learning.
    
    Captures first and second moments of SGD iterates to approximate 
    the posterior distribution over weights.
    """
    
    def __init__(self, base_model: nn.Module, max_num_models: int = 20, 
                 var_clamp: float = 1e-30):
        super().__init__()
        
        self.base_model = base_model
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        
        # Initialize SWAG parameters
        self.register_buffer('n_models', torch.zeros(1, dtype=torch.long))
        
        # First moment (mean)
        for name, param in self.base_model.named_parameters():
            self.register_buffer(f'{name}_mean', torch.zeros_like(param))
        
        # Second moment (for diagonal variance)
        for name, param in self.base_model.named_parameters():
            self.register_buffer(f'{name}_sq_mean', torch.zeros_like(param))
        
        # Deviation matrix for low-rank approximation
        self.param_names = [name for name, _ in self.base_model.named_parameters()]
        self.deviations = defaultdict(list)
        
    def update(self):
        """Update SWAG statistics with current model parameters."""
        n = float(self.n_models.item())
        
        for name, param in self.base_model.named_parameters():
            mean = getattr(self, f'{name}_mean')
            sq_mean = getattr(self, f'{name}_sq_mean')
            
            # Update first moment
            mean.mul_(n / (n + 1.0)).add_(param.data, alpha=1.0 / (n + 1.0))
            
            # Update second moment
            sq_mean.mul_(n / (n + 1.0)).add_(param.data ** 2, alpha=1.0 / (n + 1.0))
            
            # Store deviation for low-rank approximation
            if len(self.deviations[name]) >= self.max_num_models:
                self.deviations[name].pop(0)
            
            deviation = param.data - mean
            self.deviations[name].append(deviation.clone())
        
        self.n_models += 1
    
    def sample(self, scale: float = 1.0, diag_noise: bool = True) -> nn.Module:
        """
        Sample a model from the SWAG posterior.
        
        Args:
            scale: Scaling factor for sampling
            diag_noise: Whether to include diagonal noise
            
        Returns:
            Sampled model
        """
        if self.n_models == 0:
            raise RuntimeError("No models have been collected")
        
        # Create a copy of the base model
        sampled_model = copy.deepcopy(self.base_model)
        
        for name, param in sampled_model.named_parameters():
            mean = getattr(self, f'{name}_mean')
            sq_mean = getattr(self, f'{name}_sq_mean')
            
            # Start with mean
            param.data.copy_(mean)
            
            # Add diagonal noise
            if diag_noise:
                var = torch.clamp(sq_mean - mean ** 2, self.var_clamp)
                param.data.add_(torch.randn_like(param) * torch.sqrt(var) * scale)
            
            # Add low-rank noise
            if len(self.deviations[name]) > 1:
                # Stack deviations
                deviations = torch.stack(self.deviations[name])  # (K, ...)
                
                # Sample coefficients
                K = deviations.shape[0]
                z = torch.randn(K, device=param.device) * scale / np.sqrt(K - 1)
                
                # Add low-rank component
                low_rank_noise = torch.sum(z.view(-1, *([1] * (deviations.ndim - 1))) * deviations, dim=0)
                param.data.add_(low_rank_noise)
        
        return sampled_model
    
    def get_mean_model(self) -> nn.Module:
        """Get model with mean parameters."""
        mean_model = copy.deepcopy(self.base_model)
        
        for name, param in mean_model.named_parameters():
            mean = getattr(self, f'{name}_mean')
            param.data.copy_(mean)
        
        return mean_model
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            x: Input tensor
            n_samples: Number of samples for uncertainty estimation
            
        Returns:
            Mean prediction and uncertainty (std)
        """
        if self.n_models == 0:
            raise RuntimeError("No models have been collected")
        
        predictions = []
        
        for _ in range(n_samples):
            sampled_model = self.sample()
            sampled_model.eval()
            
            with torch.no_grad():
                pred = sampled_model(x)
                predictions.append(pred)
        
        # Stack predictions
        predictions = torch.stack(predictions)  # (n_samples, batch_size, ...)
        
        # Compute mean and std
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using mean model."""
        mean_model = self.get_mean_model()
        return mean_model(x)

class SWAGTrainer:
    """Trainer for SWAG models."""
    
    def __init__(self, model: nn.Module, max_num_models: int = 20):
        self.swag_model = SWAG(model, max_num_models)
        self.collection_started = False
        
    def start_collection(self):
        """Start collecting models for SWAG."""
        self.collection_started = True
        
    def update_swag(self):
        """Update SWAG with current model state."""
        if self.collection_started:
            self.swag_model.update()
    
    def get_swag_model(self) -> SWAG:
        """Get the SWAG model."""
        return self.swag_model
    
    def save_swag(self, path: str):
        """Save SWAG model."""
        torch.save({
            'swag_state_dict': self.swag_model.state_dict(),
            'n_models': self.swag_model.n_models.item(),
            'deviations': dict(self.swag_model.deviations)
        }, path)
    
    def load_swag(self, path: str, device: torch.device):
        """Load SWAG model."""
        checkpoint = torch.load(path, map_location=device)
        self.swag_model.load_state_dict(checkpoint['swag_state_dict'])
        self.swag_model.deviations = defaultdict(list, checkpoint['deviations'])
