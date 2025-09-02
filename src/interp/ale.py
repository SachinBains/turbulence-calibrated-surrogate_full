import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class ALEAnalyzer:
    """Accumulated Local Effects (ALE) for model interpretability."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def compute_ale_1d(self, X: torch.Tensor, feature_idx: int, 
                       n_bins: int = 20, target_func: Optional[Callable] = None) -> Dict:
        """
        Compute 1D ALE for a specific feature.
        
        Args:
            X: Input data (N, features)
            feature_idx: Index of feature to analyze
            n_bins: Number of bins for ALE computation
            target_func: Function to extract target from model output
            
        Returns:
            ALE results dictionary
        """
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        n_samples, n_features = X_np.shape
        
        # Get feature values and create bins
        feature_values = X_np[:, feature_idx]
        bin_edges = np.linspace(feature_values.min(), feature_values.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Initialize ALE values
        ale_values = np.zeros(n_bins)
        
        for i in range(n_bins):
            # Find samples in this bin
            in_bin = (feature_values >= bin_edges[i]) & (feature_values < bin_edges[i + 1])
            if i == n_bins - 1:  # Include right edge in last bin
                in_bin = (feature_values >= bin_edges[i]) & (feature_values <= bin_edges[i + 1])
            
            if np.sum(in_bin) == 0:
                continue
            
            X_bin = X_np[in_bin].copy()
            
            # Create two versions: one at left edge, one at right edge
            X_left = X_bin.copy()
            X_right = X_bin.copy()
            X_left[:, feature_idx] = bin_edges[i]
            X_right[:, feature_idx] = bin_edges[i + 1]
            
            # Get model predictions
            with torch.no_grad():
                X_left_tensor = torch.tensor(X_left, dtype=torch.float32).to(self.device)
                X_right_tensor = torch.tensor(X_right, dtype=torch.float32).to(self.device)
                
                pred_left = self.model(X_left_tensor)
                pred_right = self.model(X_right_tensor)
                
                if target_func:
                    pred_left = target_func(pred_left)
                    pred_right = target_func(pred_right)
                else:
                    pred_left = pred_left.mean(dim=tuple(range(1, pred_left.ndim)))
                    pred_right = pred_right.mean(dim=tuple(range(1, pred_right.ndim)))
                
                # Compute local effect
                local_effect = (pred_right - pred_left).cpu().numpy()
                ale_values[i] = np.mean(local_effect)
        
        # Accumulate effects
        ale_accumulated = np.cumsum(ale_values)
        
        # Center ALE (subtract mean)
        ale_centered = ale_accumulated - np.mean(ale_accumulated)
        
        return {
            'bin_centers': bin_centers,
            'ale_values': ale_centered,
            'bin_edges': bin_edges,
            'feature_idx': feature_idx
        }
    
    def compute_ale_2d(self, X: torch.Tensor, feature_indices: Tuple[int, int],
                       n_bins: Tuple[int, int] = (10, 10),
                       target_func: Optional[Callable] = None) -> Dict:
        """
        Compute 2D ALE for interaction between two features.
        
        Args:
            X: Input data
            feature_indices: Tuple of two feature indices
            n_bins: Number of bins for each feature
            target_func: Function to extract target from model output
            
        Returns:
            2D ALE results
        """
        X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
        feat1_idx, feat2_idx = feature_indices
        n_bins1, n_bins2 = n_bins
        
        # Get feature values and create bins
        feat1_values = X_np[:, feat1_idx]
        feat2_values = X_np[:, feat2_idx]
        
        bin_edges1 = np.linspace(feat1_values.min(), feat1_values.max(), n_bins1 + 1)
        bin_edges2 = np.linspace(feat2_values.min(), feat2_values.max(), n_bins2 + 1)
        
        bin_centers1 = (bin_edges1[:-1] + bin_edges1[1:]) / 2
        bin_centers2 = (bin_edges2[:-1] + bin_edges2[1:]) / 2
        
        # Initialize ALE grid
        ale_grid = np.zeros((n_bins1, n_bins2))
        
        for i in range(n_bins1):
            for j in range(n_bins2):
                # Find samples in this 2D bin
                in_bin1 = (feat1_values >= bin_edges1[i]) & (feat1_values < bin_edges1[i + 1])
                in_bin2 = (feat2_values >= bin_edges2[j]) & (feat2_values < bin_edges2[j + 1])
                
                if i == n_bins1 - 1:
                    in_bin1 = (feat1_values >= bin_edges1[i]) & (feat1_values <= bin_edges1[i + 1])
                if j == n_bins2 - 1:
                    in_bin2 = (feat2_values >= bin_edges2[j]) & (feat2_values <= bin_edges2[j + 1])
                
                in_bin = in_bin1 & in_bin2
                
                if np.sum(in_bin) == 0:
                    continue
                
                X_bin = X_np[in_bin].copy()
                
                # Create four corner versions
                X_corners = []
                for f1_edge in [bin_edges1[i], bin_edges1[i + 1]]:
                    for f2_edge in [bin_edges2[j], bin_edges2[j + 1]]:
                        X_corner = X_bin.copy()
                        X_corner[:, feat1_idx] = f1_edge
                        X_corner[:, feat2_idx] = f2_edge
                        X_corners.append(X_corner)
                
                # Get predictions for all corners
                corner_preds = []
                for X_corner in X_corners:
                    with torch.no_grad():
                        X_tensor = torch.tensor(X_corner, dtype=torch.float32).to(self.device)
                        pred = self.model(X_tensor)
                        
                        if target_func:
                            pred = target_func(pred)
                        else:
                            pred = pred.mean(dim=tuple(range(1, pred.ndim)))
                        
                        corner_preds.append(pred.cpu().numpy().mean())
                
                # Compute 2D local effect: f(x1_high, x2_high) - f(x1_high, x2_low) - f(x1_low, x2_high) + f(x1_low, x2_low)
                ale_grid[i, j] = corner_preds[3] - corner_preds[2] - corner_preds[1] + corner_preds[0]
        
        # Accumulate effects
        ale_accumulated = np.cumsum(np.cumsum(ale_grid, axis=0), axis=1)
        
        # Center ALE
        ale_centered = ale_accumulated - np.mean(ale_accumulated)
        
        return {
            'bin_centers1': bin_centers1,
            'bin_centers2': bin_centers2,
            'ale_grid': ale_centered,
            'feature_indices': feature_indices
        }

class TurbulenceALE:
    """ALE analysis specialized for turbulence models."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.ale_analyzer = ALEAnalyzer(model, device)
    
    def analyze_spatial_effects(self, X: torch.Tensor, n_regions: int = 8,
                               n_bins: int = 15) -> Dict:
        """
        Analyze ALE effects for different spatial regions.
        
        Args:
            X: Input velocity fields (N, C, H, W) or (N, C, D, H, W)
            n_regions: Number of spatial regions to analyze
            n_bins: Number of bins for ALE
            
        Returns:
            ALE results for spatial regions
        """
        # Flatten spatial dimensions and create regional features
        X_flat = self._create_regional_features(X, n_regions)
        
        results = {}
        
        for region_idx in range(n_regions):
            print(f"Computing ALE for region {region_idx + 1}/{n_regions}")
            
            ale_result = self.ale_analyzer.compute_ale_1d(
                X_flat, region_idx, n_bins=n_bins,
                target_func=lambda x: x.mean()
            )
            
            results[f'region_{region_idx}'] = ale_result
        
        return results
    
    def _create_regional_features(self, X: torch.Tensor, n_regions: int) -> torch.Tensor:
        """Create features representing different spatial regions."""
        if len(X.shape) == 4:  # (N, C, H, W)
            N, C, H, W = X.shape
            
            # Divide into grid regions
            regions_per_dim = int(np.sqrt(n_regions))
            h_step = H // regions_per_dim
            w_step = W // regions_per_dim
            
            regional_features = []
            
            for i in range(regions_per_dim):
                for j in range(regions_per_dim):
                    h_start = i * h_step
                    h_end = min((i + 1) * h_step, H)
                    w_start = j * w_step
                    w_end = min((j + 1) * w_step, W)
                    
                    # Average over this region
                    region_avg = X[:, :, h_start:h_end, w_start:w_end].mean(dim=(2, 3))
                    regional_features.append(region_avg.mean(dim=1))  # Average over channels
            
            return torch.stack(regional_features, dim=1)
        
        elif len(X.shape) == 5:  # (N, C, D, H, W)
            N, C, D, H, W = X.shape
            
            # Divide into 3D regions
            regions_per_dim = int(np.cbrt(n_regions))
            d_step = D // regions_per_dim
            h_step = H // regions_per_dim
            w_step = W // regions_per_dim
            
            regional_features = []
            
            for i in range(regions_per_dim):
                for j in range(regions_per_dim):
                    for k in range(regions_per_dim):
                        d_start = i * d_step
                        d_end = min((i + 1) * d_step, D)
                        h_start = j * h_step
                        h_end = min((j + 1) * h_step, H)
                        w_start = k * w_step
                        w_end = min((k + 1) * w_step, W)
                        
                        region_avg = X[:, :, d_start:d_end, h_start:h_end, w_start:w_end].mean(dim=(2, 3, 4))
                        regional_features.append(region_avg.mean(dim=1))
            
            return torch.stack(regional_features, dim=1)
        
        else:
            raise ValueError(f"Unsupported input shape: {X.shape}")
    
    def analyze_velocity_component_effects(self, X: torch.Tensor, 
                                         n_bins: int = 20) -> Dict:
        """Analyze ALE effects for different velocity components."""
        if len(X.shape) < 4 or X.shape[1] < 3:
            raise ValueError("Input must have at least 3 channels for u, v, w components")
        
        # Extract velocity components and compute statistics
        velocity_features = self._extract_velocity_features(X)
        
        results = {}
        feature_names = ['u_mean', 'v_mean', 'w_mean', 'u_std', 'v_std', 'w_std', 
                        'vel_magnitude', 'vorticity_approx']
        
        for i, feature_name in enumerate(feature_names):
            if i < velocity_features.shape[1]:
                print(f"Computing ALE for {feature_name}")
                
                ale_result = self.ale_analyzer.compute_ale_1d(
                    velocity_features, i, n_bins=n_bins,
                    target_func=lambda x: x.mean()
                )
                
                results[feature_name] = ale_result
        
        return results
    
    def _extract_velocity_features(self, X: torch.Tensor) -> torch.Tensor:
        """Extract velocity-based features for ALE analysis."""
        # Assume first 3 channels are u, v, w
        u = X[:, 0]
        v = X[:, 1] if X.shape[1] > 1 else torch.zeros_like(u)
        w = X[:, 2] if X.shape[1] > 2 else torch.zeros_like(u)
        
        features = []
        
        # Mean values
        features.append(u.mean(dim=tuple(range(1, u.ndim))))
        features.append(v.mean(dim=tuple(range(1, v.ndim))))
        features.append(w.mean(dim=tuple(range(1, w.ndim))))
        
        # Standard deviations
        features.append(u.std(dim=tuple(range(1, u.ndim))))
        features.append(v.std(dim=tuple(range(1, v.ndim))))
        features.append(w.std(dim=tuple(range(1, w.ndim))))
        
        # Velocity magnitude
        vel_mag = torch.sqrt(u**2 + v**2 + w**2)
        features.append(vel_mag.mean(dim=tuple(range(1, vel_mag.ndim))))
        
        # Approximate vorticity (simplified)
        if len(u.shape) >= 3:
            # Simple finite difference approximation
            du_dy = torch.diff(u, dim=-2, prepend=u[..., :1, :])
            dv_dx = torch.diff(v, dim=-1, prepend=v[..., :1])
            vorticity_z = dv_dx - du_dy
            features.append(vorticity_z.mean(dim=tuple(range(1, vorticity_z.ndim))))
        else:
            features.append(torch.zeros_like(features[0]))
        
        return torch.stack(features, dim=1)
    
    def plot_ale_results(self, ale_results: Dict, save_dir: Optional[str] = None):
        """Plot ALE analysis results."""
        n_features = len(ale_results)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i, (feature_name, ale_result) in enumerate(ale_results.items()):
            if i >= len(axes):
                break
            
            bin_centers = ale_result['bin_centers']
            ale_values = ale_result['ale_values']
            
            axes[i].plot(bin_centers, ale_values, 'b-', linewidth=2)
            axes[i].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('Feature Value')
            axes[i].set_ylabel('ALE Effect')
            axes[i].set_title(f'ALE: {feature_name}')
            axes[i].grid(True, alpha=0.3)
        
        # Remove empty subplots
        for i in range(len(ale_results), len(axes)):
            axes[i].remove()
        
        plt.tight_layout()
        
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f"{save_dir}/ale_analysis.png", dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def compute_ale_importance_ranking(self, ale_results: Dict) -> Dict:
        """Compute importance ranking based on ALE effect magnitudes."""
        importance_scores = {}
        
        for feature_name, ale_result in ale_results.items():
            ale_values = ale_result['ale_values']
            
            # Compute importance as range of ALE effects
            ale_range = np.max(ale_values) - np.min(ale_values)
            ale_std = np.std(ale_values)
            ale_mean_abs = np.mean(np.abs(ale_values))
            
            importance_scores[feature_name] = {
                'ale_range': float(ale_range),
                'ale_std': float(ale_std),
                'ale_mean_abs': float(ale_mean_abs)
            }
        
        # Sort by ALE range
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1]['ale_range'], reverse=True)
        
        return {
            'importance_scores': importance_scores,
            'ranking_by_range': [feat[0] for feat in sorted_features]
        }
