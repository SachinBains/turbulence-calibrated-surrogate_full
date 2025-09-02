import numpy as np
import torch
import torch.nn as nn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Dict
import warnings
warnings.filterwarnings('ignore')

class TurbulenceGPR:
    """Gaussian Process Regression for turbulence modeling with dimensionality reduction."""
    
    def __init__(self, n_components: int = 50, kernel_type: str = 'rbf', 
                 alpha: float = 1e-6, normalize_y: bool = True):
        self.n_components = n_components
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.normalize_y = normalize_y
        
        # Dimensionality reduction
        self.input_pca = PCA(n_components=n_components)
        self.output_pca = PCA(n_components=n_components)
        self.input_scaler = StandardScaler()
        self.output_scaler = StandardScaler()
        
        # GP models (one per output component)
        self.gp_models = []
        self.is_fitted = False
        
    def _get_kernel(self):
        """Get kernel based on type."""
        if self.kernel_type == 'rbf':
            return C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2)) + WhiteKernel(1e-6, (1e-10, 1e-1))
        elif self.kernel_type == 'matern':
            return C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-1))
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
    
    def _flatten_spatial(self, X: np.ndarray) -> np.ndarray:
        """Flatten spatial dimensions: (N, C, H, W) -> (N, C*H*W)"""
        if X.ndim == 4:
            return X.reshape(X.shape[0], -1)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(f"Expected 4D or 2D input, got {X.ndim}D")
    
    def _unflatten_spatial(self, X_flat: np.ndarray, original_shape: Tuple) -> np.ndarray:
        """Unflatten to original spatial dimensions."""
        return X_flat.reshape(original_shape)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'TurbulenceGPR':
        """
        Fit GPR model.
        
        Args:
            X: Input features (N, C, H, W) or (N, features)
            y: Target values (N, C, H, W) or (N, targets)
        """
        print(f"Fitting GPR with input shape: {X.shape}, output shape: {y.shape}")
        
        # Store original shapes
        self.input_shape = X.shape
        self.output_shape = y.shape
        
        # Flatten spatial dimensions
        X_flat = self._flatten_spatial(X)
        y_flat = self._flatten_spatial(y)
        
        print(f"Flattened shapes: X={X_flat.shape}, y={y_flat.shape}")
        
        # Scale inputs
        X_scaled = self.input_scaler.fit_transform(X_flat)
        y_scaled = self.output_scaler.fit_transform(y_flat)
        
        # Apply PCA for dimensionality reduction
        X_pca = self.input_pca.fit_transform(X_scaled)
        y_pca = self.output_pca.fit_transform(y_scaled)
        
        print(f"PCA shapes: X={X_pca.shape}, y={y_pca.shape}")
        
        # Fit GP models for each output component
        self.gp_models = []
        kernel = self._get_kernel()
        
        for i in range(y_pca.shape[1]):
            print(f"  Fitting GP {i+1}/{y_pca.shape[1]}")
            
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=self.alpha,
                normalize_y=self.normalize_y,
                n_restarts_optimizer=2,
                random_state=42
            )
            
            gp.fit(X_pca, y_pca[:, i])
            self.gp_models.append(gp)
        
        self.is_fitted = True
        print("GPR fitting completed")
        return self
    
    def predict(self, X: np.ndarray, return_std: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions with uncertainty.
        
        Args:
            X: Input features
            return_std: Whether to return uncertainty estimates
            
        Returns:
            Predictions and optionally standard deviations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        # Flatten and scale inputs
        X_flat = self._flatten_spatial(X)
        X_scaled = self.input_scaler.transform(X_flat)
        X_pca = self.input_pca.transform(X_scaled)
        
        # Predict with each GP
        predictions = []
        uncertainties = []
        
        for gp in self.gp_models:
            if return_std:
                pred, std = gp.predict(X_pca, return_std=True)
                predictions.append(pred)
                uncertainties.append(std)
            else:
                pred = gp.predict(X_pca, return_std=False)
                predictions.append(pred)
        
        # Stack predictions
        y_pca_pred = np.column_stack(predictions)
        
        # Inverse transform
        y_scaled_pred = self.output_pca.inverse_transform(y_pca_pred)
        y_pred = self.output_scaler.inverse_transform(y_scaled_pred)
        
        # Unflatten to original shape
        y_pred = self._unflatten_spatial(y_pred, self.output_shape)
        
        if return_std:
            # Process uncertainties
            y_pca_std = np.column_stack(uncertainties)
            
            # Transform uncertainties (approximate)
            y_scaled_std = np.abs(self.output_pca.inverse_transform(y_pca_std))
            y_std = np.abs(y_scaled_std * self.output_scaler.scale_)
            
            y_std = self._unflatten_spatial(y_std, self.output_shape)
            return y_pred, y_std
        else:
            return y_pred, None
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred, _ = self.predict(X, return_std=False)
        
        # Flatten for scoring
        y_flat = self._flatten_spatial(y)
        y_pred_flat = self._flatten_spatial(y_pred)
        
        ss_res = np.sum((y_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_flat - np.mean(y_flat, axis=0)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def get_marginal_likelihood(self) -> float:
        """Get average log marginal likelihood across all GPs."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        log_likelihoods = [gp.log_marginal_likelihood() for gp in self.gp_models]
        return np.mean(log_likelihoods)

class SparseGPR:
    """Sparse GPR using inducing points for scalability."""
    
    def __init__(self, n_inducing: int = 100, n_components: int = 20):
        self.n_inducing = n_inducing
        self.n_components = n_components
        self.input_pca = PCA(n_components=n_components)
        self.output_pca = PCA(n_components=n_components)
        self.scaler = StandardScaler()
        
        # Inducing points and parameters
        self.inducing_points = None
        self.is_fitted = False
    
    def _select_inducing_points(self, X: np.ndarray) -> np.ndarray:
        """Select inducing points using k-means clustering."""
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=self.n_inducing, random_state=42)
        kmeans.fit(X)
        return kmeans.cluster_centers_
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SparseGPR':
        """Fit sparse GPR (simplified implementation)."""
        # This is a placeholder for sparse GP implementation
        # In practice, you'd use libraries like GPyTorch or GPflow
        print("Sparse GPR fitting not fully implemented - use TurbulenceGPR instead")
        return self
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with sparse GPR."""
        raise NotImplementedError("Use TurbulenceGPR for full implementation")
