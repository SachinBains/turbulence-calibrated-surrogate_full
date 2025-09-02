import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import shap
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

class TurbulenceSHAP:
    """SHAP analysis for turbulence models with uncertainty quantification."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
    def create_background_dataset(self, dataset_loader, n_background: int = 100) -> torch.Tensor:
        """Create background dataset for SHAP analysis."""
        background_samples = []
        
        for i, (x, _) in enumerate(dataset_loader):
            if i >= n_background:
                break
            background_samples.append(x[0])  # Remove batch dimension
        
        return torch.stack(background_samples)
    
    def model_wrapper(self, x: torch.Tensor) -> np.ndarray:
        """Wrapper function for SHAP that returns numpy arrays."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            
        return output.cpu().numpy()
    
    def uncertainty_wrapper(self, x: torch.Tensor, mc_samples: int = 30) -> np.ndarray:
        """Wrapper that returns uncertainty estimates."""
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.to(self.device)
        
        # Enable MC dropout if available
        if hasattr(self.model, 'enable_mc_dropout'):
            self.model.enable_mc_dropout()
        
        predictions = []
        for _ in range(mc_samples):
            with torch.no_grad():
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        uncertainty = np.std(predictions, axis=0)  # Standard deviation as uncertainty
        
        return uncertainty
    
    def analyze_prediction_drivers(self, background_data: torch.Tensor, 
                                 test_samples: torch.Tensor,
                                 max_evals: int = 1000) -> Dict:
        """
        Analyze what drives model predictions using SHAP.
        
        Args:
            background_data: Background dataset for SHAP
            test_samples: Test samples to explain
            max_evals: Maximum evaluations for SHAP
            
        Returns:
            SHAP analysis results
        """
        print("Creating SHAP explainer...")
        
        # Flatten spatial dimensions for SHAP
        background_flat = background_data.reshape(background_data.shape[0], -1)
        test_flat = test_samples.reshape(test_samples.shape[0], -1)
        
        # Create explainer
        explainer = shap.KernelExplainer(
            lambda x: self.model_wrapper(x.reshape(-1, *background_data.shape[1:])),
            background_flat[:50]  # Use subset for efficiency
        )
        
        print(f"Computing SHAP values for {len(test_flat)} samples...")
        shap_values = explainer.shap_values(test_flat[:10], nsamples=max_evals)
        
        # Reshape back to spatial dimensions
        original_shape = test_samples.shape
        if isinstance(shap_values, list):
            # Multi-output case
            shap_spatial = []
            for sv in shap_values:
                shap_spatial.append(sv.reshape(original_shape))
        else:
            # Single output case
            shap_spatial = shap_values.reshape(original_shape)
        
        return {
            'shap_values': shap_spatial,
            'test_samples': test_samples[:10],
            'background_data': background_data[:50]
        }
    
    def analyze_uncertainty_drivers(self, background_data: torch.Tensor,
                                  test_samples: torch.Tensor,
                                  max_evals: int = 500) -> Dict:
        """
        Analyze what drives model uncertainty using SHAP.
        
        Args:
            background_data: Background dataset
            test_samples: Test samples to explain
            max_evals: Maximum evaluations
            
        Returns:
            Uncertainty SHAP analysis results
        """
        print("Analyzing uncertainty drivers with SHAP...")
        
        # Flatten for SHAP
        background_flat = background_data.reshape(background_data.shape[0], -1)
        test_flat = test_samples.reshape(test_samples.shape[0], -1)
        
        # Create explainer for uncertainty
        explainer = shap.KernelExplainer(
            lambda x: self.uncertainty_wrapper(x.reshape(-1, *background_data.shape[1:])),
            background_flat[:30]  # Smaller background for uncertainty analysis
        )
        
        print(f"Computing uncertainty SHAP values...")
        uncertainty_shap = explainer.shap_values(test_flat[:5], nsamples=max_evals)
        
        # Reshape back
        original_shape = test_samples.shape
        if isinstance(uncertainty_shap, list):
            uncertainty_spatial = []
            for us in uncertainty_shap:
                uncertainty_spatial.append(us.reshape(original_shape))
        else:
            uncertainty_spatial = uncertainty_shap.reshape(original_shape)
        
        return {
            'uncertainty_shap': uncertainty_spatial,
            'test_samples': test_samples[:5],
            'background_data': background_data[:30]
        }
    
    def create_spatial_shap_maps(self, shap_results: Dict, 
                                save_dir: Optional[str] = None) -> None:
        """Create spatial SHAP importance maps."""
        shap_values = shap_results['shap_values']
        test_samples = shap_results['test_samples']
        
        if isinstance(shap_values, list):
            # Multi-output case
            for output_idx, sv in enumerate(shap_values):
                self._plot_spatial_shap(sv, test_samples, 
                                      f"output_{output_idx}", save_dir)
        else:
            # Single output case
            self._plot_spatial_shap(shap_values, test_samples, 
                                  "prediction", save_dir)
    
    def _plot_spatial_shap(self, shap_values: np.ndarray, 
                          test_samples: torch.Tensor,
                          title: str, save_dir: Optional[str] = None):
        """Plot spatial SHAP maps."""
        n_samples = min(3, len(shap_values))
        
        for sample_idx in range(n_samples):
            if len(shap_values.shape) == 4:  # (N, C, H, W)
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Original input (first channel)
                input_slice = test_samples[sample_idx, 0].cpu().numpy()
                shap_slice = shap_values[sample_idx, 0]
                
                # Show different slices if 3D
                if len(input_slice.shape) == 3:
                    mid_z = input_slice.shape[0] // 2
                    input_2d = input_slice[mid_z]
                    shap_2d = shap_slice[mid_z]
                else:
                    input_2d = input_slice
                    shap_2d = shap_slice
                
                # Original input
                axes[0, 0].imshow(input_2d, cmap='viridis')
                axes[0, 0].set_title('Original Input')
                axes[0, 0].axis('off')
                
                # SHAP values (positive)
                shap_pos = np.maximum(shap_2d, 0)
                axes[0, 1].imshow(shap_pos, cmap='Reds')
                axes[0, 1].set_title('Positive SHAP Values')
                axes[0, 1].axis('off')
                
                # SHAP values (negative)
                shap_neg = np.minimum(shap_2d, 0)
                axes[0, 2].imshow(np.abs(shap_neg), cmap='Blues')
                axes[0, 2].set_title('Negative SHAP Values')
                axes[0, 2].axis('off')
                
                # Combined SHAP
                axes[1, 0].imshow(input_2d, cmap='gray', alpha=0.7)
                axes[1, 0].imshow(shap_2d, cmap='RdBu_r', alpha=0.6)
                axes[1, 0].set_title('SHAP Overlay')
                axes[1, 0].axis('off')
                
                # SHAP magnitude
                axes[1, 1].imshow(np.abs(shap_2d), cmap='hot')
                axes[1, 1].set_title('SHAP Magnitude')
                axes[1, 1].axis('off')
                
                # Remove empty subplot
                axes[1, 2].remove()
                
                plt.suptitle(f'SHAP Analysis - {title} - Sample {sample_idx}')
                plt.tight_layout()
                
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f"{save_dir}/shap_{title}_sample_{sample_idx}.png", 
                              dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
    
    def compute_feature_importance(self, shap_values: np.ndarray) -> Dict:
        """Compute global feature importance from SHAP values."""
        if isinstance(shap_values, list):
            # Multi-output case
            importance = {}
            for i, sv in enumerate(shap_values):
                importance[f'output_{i}'] = {
                    'mean_abs_shap': float(np.mean(np.abs(sv))),
                    'std_abs_shap': float(np.std(np.abs(sv))),
                    'max_abs_shap': float(np.max(np.abs(sv)))
                }
        else:
            # Single output case
            importance = {
                'mean_abs_shap': float(np.mean(np.abs(shap_values))),
                'std_abs_shap': float(np.std(np.abs(shap_values))),
                'max_abs_shap': float(np.max(np.abs(shap_values)))
            }
        
        return importance
    
    def create_summary_plots(self, shap_results: Dict, 
                           save_dir: Optional[str] = None):
        """Create SHAP summary plots."""
        shap_values = shap_results['shap_values']
        test_samples = shap_results['test_samples']
        
        # Flatten for summary plot
        if isinstance(shap_values, list):
            for i, sv in enumerate(shap_values):
                sv_flat = sv.reshape(sv.shape[0], -1)
                test_flat = test_samples.reshape(test_samples.shape[0], -1)
                
                plt.figure(figsize=(10, 6))
                shap.summary_plot(sv_flat, test_flat, show=False, max_display=20)
                plt.title(f'SHAP Summary - Output {i}')
                
                if save_dir:
                    import os
                    os.makedirs(save_dir, exist_ok=True)
                    plt.savefig(f"{save_dir}/shap_summary_output_{i}.png", 
                              dpi=150, bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()
        else:
            sv_flat = shap_values.reshape(shap_values.shape[0], -1)
            test_flat = test_samples.reshape(test_samples.shape[0], -1)
            
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv_flat, test_flat, show=False, max_display=20)
            plt.title('SHAP Summary')
            
            if save_dir:
                import os
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(f"{save_dir}/shap_summary.png", 
                          dpi=150, bbox_inches='tight')
                plt.close()
            else:
                plt.show()

class DeepSHAP:
    """DeepSHAP implementation for neural networks."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
    
    def analyze(self, background_data: torch.Tensor, 
                test_samples: torch.Tensor) -> Dict:
        """Analyze using DeepSHAP."""
        print("Running DeepSHAP analysis...")
        
        # Create DeepExplainer
        explainer = shap.DeepExplainer(self.model, background_data.to(self.device))
        
        # Compute SHAP values
        shap_values = explainer.shap_values(test_samples.to(self.device))
        
        return {
            'shap_values': shap_values,
            'test_samples': test_samples,
            'background_data': background_data
        }
