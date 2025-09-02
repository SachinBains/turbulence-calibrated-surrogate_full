import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from SALib.sample import saltelli
from SALib.analyze import sobol
import warnings
warnings.filterwarnings('ignore')

class SobolAnalyzer:
    """Sobol sensitivity analysis for neural network models."""
    
    def __init__(self, model: nn.Module, input_bounds: Dict[str, Tuple[float, float]], 
                 device: torch.device = None):
        """
        Initialize Sobol analyzer.
        
        Args:
            model: PyTorch model to analyze
            input_bounds: Dictionary mapping parameter names to (min, max) bounds
            device: Device for computation
        """
        self.model = model
        self.input_bounds = input_bounds
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Create problem definition for SALib
        self.problem = {
            'num_vars': len(input_bounds),
            'names': list(input_bounds.keys()),
            'bounds': list(input_bounds.values())
        }
    
    def generate_samples(self, n_samples: int = 1024, calc_second_order: bool = True) -> np.ndarray:
        """Generate Saltelli samples for Sobol analysis."""
        return saltelli.sample(self.problem, n_samples, calc_second_order=calc_second_order)
    
    def evaluate_model(self, samples: np.ndarray, 
                      input_transform: Optional[Callable] = None,
                      output_transform: Optional[Callable] = None) -> np.ndarray:
        """
        Evaluate model on samples.
        
        Args:
            samples: Input samples from Saltelli sampling
            input_transform: Function to transform samples to model input format
            output_transform: Function to transform model output to scalar
            
        Returns:
            Model outputs for each sample
        """
        outputs = []
        
        with torch.no_grad():
            for sample in samples:
                # Transform sample to model input
                if input_transform:
                    model_input = input_transform(sample)
                else:
                    # Default: assume sample is flattened input
                    model_input = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
                
                model_input = model_input.to(self.device)
                
                # Forward pass
                output = self.model(model_input)
                
                # Transform output to scalar
                if output_transform:
                    scalar_output = output_transform(output)
                else:
                    # Default: take mean of output
                    scalar_output = output.mean().item()
                
                outputs.append(scalar_output)
        
        return np.array(outputs)
    
    def analyze(self, samples: np.ndarray, outputs: np.ndarray, 
                calc_second_order: bool = True) -> Dict:
        """Perform Sobol analysis."""
        return sobol.analyze(self.problem, outputs, calc_second_order=calc_second_order)
    
    def run_full_analysis(self, n_samples: int = 1024,
                         input_transform: Optional[Callable] = None,
                         output_transform: Optional[Callable] = None,
                         calc_second_order: bool = True) -> Dict:
        """Run complete Sobol sensitivity analysis."""
        print(f"Generating {n_samples} Saltelli samples...")
        samples = self.generate_samples(n_samples, calc_second_order)
        
        print(f"Evaluating model on {len(samples)} samples...")
        outputs = self.evaluate_model(samples, input_transform, output_transform)
        
        print("Computing Sobol indices...")
        results = self.analyze(samples, outputs, calc_second_order)
        
        return results

class TurbulenceSobolAnalyzer:
    """Specialized Sobol analyzer for turbulence models."""
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        self.model = model
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def analyze_velocity_sensitivity(self, sample_input: torch.Tensor, 
                                   n_samples: int = 512,
                                   perturbation_scale: float = 0.1) -> Dict:
        """
        Analyze sensitivity to velocity field perturbations.
        
        Args:
            sample_input: Base input tensor (1, C, H, W)
            n_samples: Number of samples for analysis
            perturbation_scale: Scale of perturbations
            
        Returns:
            Sensitivity analysis results
        """
        # Define spatial regions for sensitivity analysis
        _, C, H, W = sample_input.shape
        
        # Create problem for spatial regions
        n_regions = 8  # Divide space into 8 regions
        region_names = [f'region_{i}' for i in range(n_regions)]
        bounds = [(-perturbation_scale, perturbation_scale)] * n_regions
        
        problem = {
            'num_vars': n_regions,
            'names': region_names,
            'bounds': bounds
        }
        
        # Generate samples
        samples = saltelli.sample(problem, n_samples)
        
        # Evaluate model
        outputs = []
        base_output = self._get_base_output(sample_input)
        
        with torch.no_grad():
            for sample in samples:
                # Create perturbed input
                perturbed_input = self._create_perturbed_input(sample_input, sample, n_regions)
                
                # Forward pass
                output = self.model(perturbed_input)
                
                # Compute change in output
                output_change = torch.norm(output - base_output).item()
                outputs.append(output_change)
        
        # Analyze
        results = sobol.analyze(problem, np.array(outputs))
        
        return results
    
    def _get_base_output(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Get base model output."""
        with torch.no_grad():
            return self.model(input_tensor.to(self.device))
    
    def _create_perturbed_input(self, base_input: torch.Tensor, 
                               perturbations: np.ndarray, n_regions: int) -> torch.Tensor:
        """Create spatially perturbed input."""
        perturbed = base_input.clone()
        _, C, H, W = base_input.shape
        
        # Divide spatial dimensions into regions
        h_step = H // int(np.sqrt(n_regions))
        w_step = W // int(np.sqrt(n_regions))
        
        region_idx = 0
        for i in range(0, H, h_step):
            for j in range(0, W, w_step):
                if region_idx < n_regions:
                    h_end = min(i + h_step, H)
                    w_end = min(j + w_step, W)
                    
                    # Apply perturbation to this region
                    perturbation = perturbations[region_idx]
                    perturbed[:, :, i:h_end, j:w_end] += perturbation
                    
                    region_idx += 1
        
        return perturbed
    
    def analyze_parameter_sensitivity(self, sample_input: torch.Tensor,
                                    parameter_ranges: Dict[str, Tuple[float, float]],
                                    n_samples: int = 512) -> Dict:
        """
        Analyze sensitivity to model parameters.
        
        Args:
            sample_input: Input tensor for evaluation
            parameter_ranges: Ranges for parameter perturbation
            n_samples: Number of samples
            
        Returns:
            Parameter sensitivity results
        """
        # Get original parameters
        original_params = {}
        for name, param in self.model.named_parameters():
            if name in parameter_ranges:
                original_params[name] = param.data.clone()
        
        # Create problem
        problem = {
            'num_vars': len(parameter_ranges),
            'names': list(parameter_ranges.keys()),
            'bounds': list(parameter_ranges.values())
        }
        
        # Generate samples
        samples = saltelli.sample(problem, n_samples)
        
        # Get base output
        base_output = self._get_base_output(sample_input)
        
        # Evaluate with perturbed parameters
        outputs = []
        
        for sample in samples:
            # Perturb parameters
            for i, param_name in enumerate(parameter_ranges.keys()):
                if hasattr(self.model, param_name):
                    param = getattr(self.model, param_name)
                    if hasattr(param, 'data'):
                        param.data = original_params[param_name] + sample[i]
            
            # Forward pass
            with torch.no_grad():
                output = self.model(sample_input.to(self.device))
                output_change = torch.norm(output - base_output).item()
                outputs.append(output_change)
            
            # Restore original parameters
            for param_name in parameter_ranges.keys():
                if hasattr(self.model, param_name):
                    param = getattr(self.model, param_name)
                    if hasattr(param, 'data'):
                        param.data = original_params[param_name]
        
        # Analyze
        results = sobol.analyze(problem, np.array(outputs))
        
        return results

def plot_sobol_results(results: Dict, title: str = "Sobol Sensitivity Analysis",
                      save_path: Optional[str] = None):
    """Plot Sobol sensitivity analysis results."""
    import matplotlib.pyplot as plt
    
    # Extract indices
    names = results['names']
    s1 = results['S1']
    st = results['ST']
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # First-order indices
    ax1.barh(range(len(names)), s1)
    ax1.set_yticks(range(len(names)))
    ax1.set_yticklabels(names)
    ax1.set_xlabel('First-order Sensitivity Index (S1)')
    ax1.set_title('First-order Effects')
    ax1.grid(True, alpha=0.3)
    
    # Total indices
    ax2.barh(range(len(names)), st)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.set_xlabel('Total Sensitivity Index (ST)')
    ax2.set_title('Total Effects (including interactions)')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
