import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations_with_replacement
from typing import List, Tuple, Optional, Dict

class SINDyLibrary:
    """Sparse Identification of Nonlinear Dynamics (SINDy) feature library."""
    
    def __init__(self, poly_degree: int = 3, include_trig: bool = True, 
                 include_exp: bool = False, custom_functions: Optional[List] = None):
        self.poly_degree = poly_degree
        self.include_trig = include_trig
        self.include_exp = include_exp
        self.custom_functions = custom_functions or []
        self.feature_names = []
        
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Build library of candidate functions."""
        n_samples, n_features = X.shape
        library_functions = []
        self.feature_names = []
        
        # Polynomial features
        poly = PolynomialFeatures(degree=self.poly_degree, include_bias=True)
        poly_features = poly.fit_transform(X)
        library_functions.append(poly_features)
        self.feature_names.extend(poly.get_feature_names_out([f'x{i}' for i in range(n_features)]))
        
        # Trigonometric functions
        if self.include_trig:
            trig_features = []
            trig_names = []
            for i in range(n_features):
                sin_feat = np.sin(X[:, i:i+1])
                cos_feat = np.cos(X[:, i:i+1])
                trig_features.extend([sin_feat, cos_feat])
                trig_names.extend([f'sin(x{i})', f'cos(x{i})'])
            
            if trig_features:
                library_functions.append(np.hstack(trig_features))
                self.feature_names.extend(trig_names)
        
        # Exponential functions
        if self.include_exp:
            exp_features = []
            exp_names = []
            for i in range(n_features):
                # Clamp to avoid overflow
                exp_feat = np.exp(np.clip(X[:, i:i+1], -10, 10))
                exp_features.append(exp_feat)
                exp_names.append(f'exp(x{i})')
            
            if exp_features:
                library_functions.append(np.hstack(exp_features))
                self.feature_names.extend(exp_names)
        
        # Custom functions
        for func_name, func in self.custom_functions:
            custom_feat = func(X)
            if custom_feat.ndim == 1:
                custom_feat = custom_feat.reshape(-1, 1)
            library_functions.append(custom_feat)
            if custom_feat.shape[1] == 1:
                self.feature_names.append(func_name)
            else:
                self.feature_names.extend([f'{func_name}_{i}' for i in range(custom_feat.shape[1])])
        
        return np.hstack(library_functions)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform new data using fitted library."""
        return self.fit_transform(X)  # Stateless for now

class SINDyRegressor:
    """SINDy sparse regression model."""
    
    def __init__(self, library: SINDyLibrary, alpha: float = 0.01, 
                 max_iter: int = 1000, normalize: bool = True,
                 threshold: float = 1e-6):
        self.library = library
        self.alpha = alpha
        self.max_iter = max_iter
        self.normalize = normalize
        self.threshold = threshold
        self.coef_ = None
        self.feature_names_ = None
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SINDyRegressor':
        """Fit SINDy model."""
        # Build library
        Theta = self.library.fit_transform(X)
        self.feature_names_ = self.library.feature_names.copy()
        
        # Fit sparse regression
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_outputs = y.shape[1]
        self.coef_ = np.zeros((Theta.shape[1], n_outputs))
        
        for i in range(n_outputs):
            # Use Lasso for sparsity
            lasso = Lasso(alpha=self.alpha, max_iter=self.max_iter, 
                         normalize=self.normalize, fit_intercept=False)
            lasso.fit(Theta, y[:, i])
            
            # Apply threshold for additional sparsity
            coef = lasso.coef_.copy()
            coef[np.abs(coef) < self.threshold] = 0
            self.coef_[:, i] = coef
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        Theta = self.library.transform(X)
        return Theta @ self.coef_
    
    def get_equations(self, feature_names: Optional[List[str]] = None) -> List[str]:
        """Get discovered equations in human-readable form."""
        if self.coef_ is None:
            raise ValueError("Model must be fitted first")
        
        if feature_names is None:
            feature_names = [f'y{i}' for i in range(self.coef_.shape[1])]
        
        equations = []
        for i in range(self.coef_.shape[1]):
            terms = []
            for j, coef in enumerate(self.coef_[:, i]):
                if abs(coef) > self.threshold:
                    if len(terms) == 0:
                        terms.append(f"{coef:.4f} * {self.feature_names_[j]}")
                    else:
                        sign = "+" if coef >= 0 else "-"
                        terms.append(f" {sign} {abs(coef):.4f} * {self.feature_names_[j]}")
            
            if terms:
                equation = f"d{feature_names[i]}/dt = " + "".join(terms)
            else:
                equation = f"d{feature_names[i]}/dt = 0"
            equations.append(equation)
        
        return equations

class TurbulenceSINDy:
    """SINDy specifically for turbulence modeling."""
    
    def __init__(self, alpha: float = 0.01):
        # Define turbulence-specific library functions
        custom_functions = [
            ("vorticity_mag", lambda X: np.sqrt(X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2).reshape(-1, 1)),
            ("kinetic_energy", lambda X: 0.5 * (X[:, 0]**2 + X[:, 1]**2 + X[:, 2]**2).reshape(-1, 1)),
            ("strain_rate", lambda X: np.sqrt(X[:, 0]**2 + X[:, 1]**2).reshape(-1, 1)),
        ]
        
        self.library = SINDyLibrary(
            poly_degree=2, 
            include_trig=False,  # Usually not needed for turbulence
            include_exp=False,
            custom_functions=custom_functions
        )
        
        self.model = SINDyRegressor(self.library, alpha=alpha)
    
    def fit(self, velocity_fields: np.ndarray, time_derivatives: np.ndarray) -> 'TurbulenceSINDy':
        """
        Fit SINDy to turbulence data.
        
        Args:
            velocity_fields: (n_samples, 3) array of [u, v, w] velocities
            time_derivatives: (n_samples, 3) array of [du/dt, dv/dt, dw/dt]
        """
        self.model.fit(velocity_fields, time_derivatives)
        return self
    
    def predict(self, velocity_fields: np.ndarray) -> np.ndarray:
        """Predict time derivatives."""
        return self.model.predict(velocity_fields)
    
    def get_turbulence_equations(self) -> List[str]:
        """Get discovered turbulence equations."""
        return self.model.get_equations(['u', 'v', 'w'])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        return 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y, axis=0))**2)
