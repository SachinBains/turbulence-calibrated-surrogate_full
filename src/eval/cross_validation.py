import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Dict, List, Tuple, Optional, Callable, Any
import json
from pathlib import Path
import copy
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class TurbulenceCrossValidator:
    """Cross-validation framework for turbulence surrogate models."""
    
    def __init__(self, n_folds: int = 5, stratify: bool = True, 
                 random_state: int = 42):
        """
        Initialize cross-validator.
        
        Args:
            n_folds: Number of CV folds
            stratify: Whether to stratify based on turbulence intensity
            random_state: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.stratify = stratify
        self.random_state = random_state
        self.fold_results = []
        
    def create_turbulence_stratification(self, dataset) -> np.ndarray:
        """Create stratification labels based on turbulence intensity."""
        intensities = []
        
        for i in range(len(dataset)):
            sample, _ = dataset[i]
            # Compute turbulence intensity (kinetic energy)
            if isinstance(sample, torch.Tensor):
                intensity = torch.mean(sample**2).item()
            else:
                intensity = np.mean(sample**2)
            intensities.append(intensity)
        
        intensities = np.array(intensities)
        
        # Create stratification bins based on quartiles
        quartiles = np.percentile(intensities, [25, 50, 75])
        strat_labels = np.digitize(intensities, quartiles)
        
        return strat_labels
    
    def get_cv_splits(self, dataset) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate cross-validation splits."""
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        
        if self.stratify:
            strat_labels = self.create_turbulence_stratification(dataset)
            cv = StratifiedKFold(n_splits=self.n_folds, shuffle=True, 
                               random_state=self.random_state)
            splits = list(cv.split(indices, strat_labels))
        else:
            cv = KFold(n_splits=self.n_folds, shuffle=True, 
                      random_state=self.random_state)
            splits = list(cv.split(indices))
        
        return splits
    
    def evaluate_fold(self, model: nn.Module, train_loader: DataLoader, 
                     val_loader: DataLoader, device: torch.device,
                     metrics_fn: Callable, fold_idx: int) -> Dict[str, Any]:
        """Evaluate a single fold."""
        
        # Clone model for this fold
        fold_model = copy.deepcopy(model)
        fold_model.to(device)
        
        # Training setup
        optimizer = torch.optim.Adam(fold_model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        
        # Training loop (simplified)
        fold_model.train()
        train_losses = []
        
        for epoch in range(50):  # Reduced epochs for CV
            epoch_loss = 0
            n_batches = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                pred = fold_model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)
            
            # Early stopping check
            if len(train_losses) > 10:
                recent_losses = train_losses[-10:]
                if max(recent_losses) - min(recent_losses) < 1e-6:
                    break
        
        # Validation evaluation
        fold_model.eval()
        val_predictions = []
        val_targets = []
        val_losses = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = fold_model(batch_x)
                loss = criterion(pred, batch_y)
                
                val_predictions.append(pred.cpu().numpy())
                val_targets.append(batch_y.cpu().numpy())
                val_losses.append(loss.item())
        
        # Concatenate results
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        
        # Compute metrics
        fold_metrics = metrics_fn(val_predictions, val_targets)
        fold_metrics['val_loss'] = np.mean(val_losses)
        fold_metrics['train_loss_final'] = train_losses[-1]
        fold_metrics['n_epochs'] = len(train_losses)
        fold_metrics['fold_idx'] = fold_idx
        
        return fold_metrics
    
    def run_cross_validation(self, model: nn.Module, dataset, 
                           device: torch.device, metrics_fn: Callable,
                           batch_size: int = 8) -> Dict[str, Any]:
        """Run complete cross-validation."""
        
        print(f"Running {self.n_folds}-fold cross-validation...")
        
        # Get CV splits
        splits = self.get_cv_splits(dataset)
        
        fold_results = []
        
        for fold_idx, (train_indices, val_indices) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{self.n_folds}")
            
            # Create fold datasets
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
            
            train_loader = DataLoader(train_subset, batch_size=batch_size, 
                                    shuffle=True, num_workers=0)
            val_loader = DataLoader(val_subset, batch_size=batch_size, 
                                  shuffle=False, num_workers=0)
            
            # Evaluate fold
            fold_result = self.evaluate_fold(
                model, train_loader, val_loader, device, metrics_fn, fold_idx
            )
            
            fold_results.append(fold_result)
            
            # Print fold summary
            print(f"  Val Loss: {fold_result['val_loss']:.6f}")
            if 'mse' in fold_result:
                print(f"  Val MSE: {fold_result['mse']:.6f}")
        
        # Aggregate results
        cv_results = self.aggregate_results(fold_results)
        self.fold_results = fold_results
        
        return cv_results
    
    def aggregate_results(self, fold_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across folds."""
        
        # Collect all metric names
        all_metrics = set()
        for result in fold_results:
            all_metrics.update(result.keys())
        
        # Remove non-numeric keys
        numeric_metrics = {k for k in all_metrics 
                          if isinstance(fold_results[0].get(k), (int, float))}
        
        aggregated = {}
        
        for metric in numeric_metrics:
            values = [result[metric] for result in fold_results 
                     if metric in result]
            
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_min'] = np.min(values)
                aggregated[f'{metric}_max'] = np.max(values)
                aggregated[f'{metric}_values'] = values
        
        # Add summary statistics
        aggregated['n_folds'] = len(fold_results)
        aggregated['cv_score'] = aggregated.get('val_loss_mean', float('inf'))
        aggregated['cv_stability'] = aggregated.get('val_loss_std', float('inf'))
        
        return aggregated
    
    def plot_cv_results(self, cv_results: Dict, save_path: Optional[str] = None):
        """Plot cross-validation results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Validation loss across folds
        if 'val_loss_values' in cv_results:
            axes[0, 0].bar(range(1, self.n_folds + 1), 
                          cv_results['val_loss_values'])
            axes[0, 0].axhline(cv_results['val_loss_mean'], color='red', 
                              linestyle='--', label='Mean')
            axes[0, 0].set_xlabel('Fold')
            axes[0, 0].set_ylabel('Validation Loss')
            axes[0, 0].set_title('Validation Loss Across Folds')
            axes[0, 0].legend()
        
        # Plot 2: MSE across folds (if available)
        if 'mse_values' in cv_results:
            axes[0, 1].bar(range(1, self.n_folds + 1), 
                          cv_results['mse_values'])
            axes[0, 1].axhline(cv_results['mse_mean'], color='red', 
                              linestyle='--', label='Mean')
            axes[0, 1].set_xlabel('Fold')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].set_title('MSE Across Folds')
            axes[0, 1].legend()
        
        # Plot 3: Training vs Validation loss
        if 'train_loss_final_values' in cv_results and 'val_loss_values' in cv_results:
            train_losses = cv_results['train_loss_final_values']
            val_losses = cv_results['val_loss_values']
            
            x = np.arange(1, self.n_folds + 1)
            width = 0.35
            
            axes[1, 0].bar(x - width/2, train_losses, width, label='Train')
            axes[1, 0].bar(x + width/2, val_losses, width, label='Validation')
            axes[1, 0].set_xlabel('Fold')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].set_title('Train vs Validation Loss')
            axes[1, 0].legend()
        
        # Plot 4: Stability analysis
        metric_names = [k.replace('_std', '') for k in cv_results.keys() 
                       if k.endswith('_std') and not k.startswith('n_')]
        
        if metric_names:
            stds = [cv_results[f'{metric}_std'] for metric in metric_names]
            means = [cv_results[f'{metric}_mean'] for metric in metric_names]
            cvs = [std/mean if mean != 0 else 0 for std, mean in zip(stds, means)]
            
            axes[1, 1].bar(range(len(metric_names)), cvs)
            axes[1, 1].set_xlabel('Metric')
            axes[1, 1].set_ylabel('Coefficient of Variation')
            axes[1, 1].set_title('Metric Stability (CV = std/mean)')
            axes[1, 1].set_xticks(range(len(metric_names)))
            axes[1, 1].set_xticklabels(metric_names, rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def save_results(self, cv_results: Dict, fold_results: List[Dict], 
                    save_path: str):
        """Save cross-validation results."""
        
        results_dict = {
            'cv_summary': cv_results,
            'fold_details': fold_results,
            'cv_config': {
                'n_folds': self.n_folds,
                'stratify': self.stratify,
                'random_state': self.random_state
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
    
    def compare_models(self, models: Dict[str, nn.Module], dataset,
                      device: torch.device, metrics_fn: Callable,
                      save_dir: Optional[str] = None) -> Dict[str, Dict]:
        """Compare multiple models using cross-validation."""
        
        print("Comparing models using cross-validation...")
        
        all_results = {}
        
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            cv_results = self.run_cross_validation(
                model, dataset, device, metrics_fn
            )
            
            all_results[model_name] = cv_results
            
            if save_dir:
                save_path = Path(save_dir) / f'{model_name}_cv_results.json'
                self.save_results(cv_results, self.fold_results, str(save_path))
        
        # Create comparison plot
        if save_dir and len(models) > 1:
            self.plot_model_comparison(all_results, save_dir)
        
        return all_results
    
    def plot_model_comparison(self, all_results: Dict[str, Dict], save_dir: str):
        """Plot comparison between models."""
        
        model_names = list(all_results.keys())
        
        # Extract key metrics for comparison
        metrics_to_compare = ['val_loss', 'mse']
        
        fig, axes = plt.subplots(1, len(metrics_to_compare), 
                                figsize=(6 * len(metrics_to_compare), 6))
        
        if len(metrics_to_compare) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics_to_compare):
            means = []
            stds = []
            valid_models = []
            
            for model_name in model_names:
                mean_key = f'{metric}_mean'
                std_key = f'{metric}_std'
                
                if mean_key in all_results[model_name]:
                    means.append(all_results[model_name][mean_key])
                    stds.append(all_results[model_name][std_key])
                    valid_models.append(model_name)
            
            if means:
                x_pos = np.arange(len(valid_models))
                axes[i].bar(x_pos, means, yerr=stds, capsize=5)
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(valid_models, rotation=45)
        
        plt.tight_layout()
        plt.savefig(Path(save_dir) / 'model_comparison.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()

def turbulence_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """Compute turbulence-specific metrics for cross-validation."""
    
    # Flatten arrays
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Basic metrics
    mse = np.mean((pred_flat - target_flat)**2)
    mae = np.mean(np.abs(pred_flat - target_flat))
    rmse = np.sqrt(mse)
    
    # Relative metrics
    target_std = np.std(target_flat)
    if target_std > 0:
        nrmse = rmse / target_std
        relative_error = rmse / np.mean(np.abs(target_flat))
    else:
        nrmse = float('inf')
        relative_error = float('inf')
    
    # Correlation
    correlation = np.corrcoef(pred_flat, target_flat)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # Energy preservation (turbulence-specific)
    pred_energy = np.mean(predictions**2)
    target_energy = np.mean(targets**2)
    energy_error = abs(pred_energy - target_energy) / target_energy
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'nrmse': nrmse,
        'relative_error': relative_error,
        'correlation': correlation,
        'energy_error': energy_error
    }
