import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
from pathlib import Path

class AdversarialAttacker:
    """Adversarial attacks for turbulence models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize adversarial attacker.
        
        Args:
            model: Target model
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def fgsm_attack(self, data: torch.Tensor, target: torch.Tensor, 
                   epsilon: float = 0.01) -> torch.Tensor:
        """Fast Gradient Sign Method attack."""
        data = data.to(self.device).requires_grad_(True)
        target = target.to(self.device)
        
        # Forward pass
        output = self.model(data)
        loss = F.mse_loss(output, target)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Generate adversarial example
        data_grad = data.grad.data
        perturbed_data = data + epsilon * data_grad.sign()
        
        return perturbed_data.detach()
    
    def pgd_attack(self, data: torch.Tensor, target: torch.Tensor,
                   epsilon: float = 0.01, alpha: float = 0.002, 
                   num_iter: int = 10) -> torch.Tensor:
        """Projected Gradient Descent attack."""
        data = data.to(self.device)
        target = target.to(self.device)
        
        # Initialize perturbed data
        perturbed_data = data.clone().detach()
        
        for i in range(num_iter):
            perturbed_data.requires_grad_(True)
            
            # Forward pass
            output = self.model(perturbed_data)
            loss = F.mse_loss(output, target)
            
            # Backward pass
            self.model.zero_grad()
            loss.backward()
            
            # Update perturbation
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data.detach() + alpha * data_grad.sign()
            
            # Project onto epsilon ball
            perturbation = torch.clamp(perturbed_data - data, -epsilon, epsilon)
            perturbed_data = data + perturbation
            
        return perturbed_data.detach()
    
    def physics_aware_attack(self, data: torch.Tensor, target: torch.Tensor,
                           epsilon: float = 0.01, alpha: float = 0.002,
                           num_iter: int = 10, physics_weight: float = 0.1) -> torch.Tensor:
        """Physics-aware adversarial attack for turbulence data."""
        data = data.to(self.device)
        target = target.to(self.device)
        
        perturbed_data = data.clone().detach()
        
        for i in range(num_iter):
            perturbed_data.requires_grad_(True)
            
            # Forward pass
            output = self.model(perturbed_data)
            
            # Prediction loss
            pred_loss = F.mse_loss(output, target)
            
            # Physics constraint loss (energy preservation)
            original_energy = torch.mean(data**2)
            perturbed_energy = torch.mean(perturbed_data**2)
            physics_loss = torch.abs(perturbed_energy - original_energy)
            
            # Combined loss
            total_loss = pred_loss + physics_weight * physics_loss
            
            # Backward pass
            self.model.zero_grad()
            total_loss.backward()
            
            # Update perturbation
            data_grad = perturbed_data.grad.data
            perturbed_data = perturbed_data.detach() + alpha * data_grad.sign()
            
            # Project onto epsilon ball
            perturbation = torch.clamp(perturbed_data - data, -epsilon, epsilon)
            perturbed_data = data + perturbation
            
        return perturbed_data.detach()
    
    def gaussian_noise_attack(self, data: torch.Tensor, 
                            noise_std: float = 0.01) -> torch.Tensor:
        """Gaussian noise attack."""
        noise = torch.randn_like(data) * noise_std
        return data + noise
    
    def spatial_attack(self, data: torch.Tensor, shift_pixels: int = 2) -> torch.Tensor:
        """Spatial transformation attack."""
        # Simple translation attack
        if data.dim() == 4:  # (N, C, H, W)
            shifted_data = torch.roll(data, shifts=(shift_pixels, shift_pixels), dims=(2, 3))
        elif data.dim() == 5:  # (N, C, D, H, W)
            shifted_data = torch.roll(data, shifts=(shift_pixels, shift_pixels, shift_pixels), dims=(2, 3, 4))
        else:
            shifted_data = data
        
        return shifted_data

class RobustnessEvaluator:
    """Evaluate model robustness against adversarial attacks."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize robustness evaluator.
        
        Args:
            model: Target model
            device: Computation device
        """
        self.model = model
        self.device = device
        self.attacker = AdversarialAttacker(model, device)
    
    def evaluate_attack_robustness(self, data_loader, attack_configs: List[Dict]) -> Dict[str, Dict]:
        """Evaluate robustness against multiple attacks."""
        
        results = {}
        
        for attack_config in attack_configs:
            attack_name = attack_config['name']
            attack_params = attack_config.get('params', {})
            
            print(f"Evaluating {attack_name} attack...")
            
            attack_results = self._evaluate_single_attack(
                data_loader, attack_name, attack_params
            )
            
            results[attack_name] = attack_results
        
        return results
    
    def _evaluate_single_attack(self, data_loader, attack_name: str, 
                               attack_params: Dict) -> Dict[str, float]:
        """Evaluate robustness against a single attack."""
        
        clean_losses = []
        adversarial_losses = []
        perturbation_norms = []
        prediction_changes = []
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 50:  # Limit for efficiency
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Clean prediction
            with torch.no_grad():
                clean_pred = self.model(data)
                clean_loss = F.mse_loss(clean_pred, target)
                clean_losses.append(clean_loss.item())
            
            # Generate adversarial example
            if attack_name == 'fgsm':
                adv_data = self.attacker.fgsm_attack(data, target, **attack_params)
            elif attack_name == 'pgd':
                adv_data = self.attacker.pgd_attack(data, target, **attack_params)
            elif attack_name == 'physics_aware':
                adv_data = self.attacker.physics_aware_attack(data, target, **attack_params)
            elif attack_name == 'gaussian':
                adv_data = self.attacker.gaussian_noise_attack(data, **attack_params)
            elif attack_name == 'spatial':
                adv_data = self.attacker.spatial_attack(data, **attack_params)
            else:
                continue
            
            # Adversarial prediction
            with torch.no_grad():
                adv_pred = self.model(adv_data)
                adv_loss = F.mse_loss(adv_pred, target)
                adversarial_losses.append(adv_loss.item())
            
            # Compute metrics
            perturbation = adv_data - data
            perturbation_norm = torch.norm(perturbation).item()
            perturbation_norms.append(perturbation_norm)
            
            pred_change = torch.norm(adv_pred - clean_pred).item()
            prediction_changes.append(pred_change)
        
        # Aggregate results
        return {
            'clean_loss_mean': np.mean(clean_losses),
            'clean_loss_std': np.std(clean_losses),
            'adversarial_loss_mean': np.mean(adversarial_losses),
            'adversarial_loss_std': np.std(adversarial_losses),
            'loss_increase': np.mean(adversarial_losses) - np.mean(clean_losses),
            'perturbation_norm_mean': np.mean(perturbation_norms),
            'perturbation_norm_std': np.std(perturbation_norms),
            'prediction_change_mean': np.mean(prediction_changes),
            'prediction_change_std': np.std(prediction_changes),
            'robustness_score': np.mean(clean_losses) / (np.mean(adversarial_losses) + 1e-8)
        }
    
    def evaluate_uncertainty_robustness(self, data_loader, 
                                       uncertainty_fn: Callable) -> Dict[str, float]:
        """Evaluate robustness of uncertainty estimates."""
        
        clean_uncertainties = []
        adv_uncertainties = []
        uncertainty_changes = []
        
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx >= 20:  # Limit for efficiency
                break
                
            data, target = data.to(self.device), target.to(self.device)
            
            # Clean uncertainty
            clean_uncertainty = uncertainty_fn(data)
            clean_uncertainties.append(clean_uncertainty)
            
            # Generate adversarial example (FGSM)
            adv_data = self.attacker.fgsm_attack(data, target, epsilon=0.01)
            
            # Adversarial uncertainty
            adv_uncertainty = uncertainty_fn(adv_data)
            adv_uncertainties.append(adv_uncertainty)
            
            # Uncertainty change
            uncertainty_change = abs(adv_uncertainty - clean_uncertainty)
            uncertainty_changes.append(uncertainty_change)
        
        return {
            'clean_uncertainty_mean': np.mean(clean_uncertainties),
            'adversarial_uncertainty_mean': np.mean(adv_uncertainties),
            'uncertainty_change_mean': np.mean(uncertainty_changes),
            'uncertainty_stability': 1.0 / (np.mean(uncertainty_changes) + 1e-8)
        }
    
    def comprehensive_robustness_analysis(self, data_loader) -> Dict[str, Dict]:
        """Run comprehensive robustness analysis."""
        
        # Define attack configurations
        attack_configs = [
            {
                'name': 'fgsm',
                'params': {'epsilon': 0.01}
            },
            {
                'name': 'fgsm',
                'params': {'epsilon': 0.05}
            },
            {
                'name': 'pgd',
                'params': {'epsilon': 0.01, 'alpha': 0.002, 'num_iter': 10}
            },
            {
                'name': 'physics_aware',
                'params': {'epsilon': 0.01, 'alpha': 0.002, 'num_iter': 10, 'physics_weight': 0.1}
            },
            {
                'name': 'gaussian',
                'params': {'noise_std': 0.01}
            },
            {
                'name': 'gaussian',
                'params': {'noise_std': 0.05}
            },
            {
                'name': 'spatial',
                'params': {'shift_pixels': 2}
            }
        ]
        
        # Evaluate attacks
        attack_results = self.evaluate_attack_robustness(data_loader, attack_configs)
        
        # Compute overall robustness metrics
        overall_metrics = self._compute_overall_robustness(attack_results)
        
        return {
            'attack_results': attack_results,
            'overall_metrics': overall_metrics
        }
    
    def _compute_overall_robustness(self, attack_results: Dict) -> Dict[str, float]:
        """Compute overall robustness metrics."""
        
        all_robustness_scores = []
        all_loss_increases = []
        all_prediction_changes = []
        
        for attack_name, results in attack_results.items():
            if 'robustness_score' in results:
                all_robustness_scores.append(results['robustness_score'])
            if 'loss_increase' in results:
                all_loss_increases.append(results['loss_increase'])
            if 'prediction_change_mean' in results:
                all_prediction_changes.append(results['prediction_change_mean'])
        
        return {
            'overall_robustness_score': np.mean(all_robustness_scores) if all_robustness_scores else 0.0,
            'average_loss_increase': np.mean(all_loss_increases) if all_loss_increases else 0.0,
            'average_prediction_change': np.mean(all_prediction_changes) if all_prediction_changes else 0.0,
            'robustness_stability': np.std(all_robustness_scores) if all_robustness_scores else 0.0
        }
    
    def plot_robustness_analysis(self, results: Dict, save_path: Optional[str] = None):
        """Plot robustness analysis results."""
        
        attack_results = results['attack_results']
        
        # Extract data for plotting
        attack_names = list(attack_results.keys())
        robustness_scores = [attack_results[name].get('robustness_score', 0) 
                           for name in attack_names]
        loss_increases = [attack_results[name].get('loss_increase', 0) 
                         for name in attack_names]
        prediction_changes = [attack_results[name].get('prediction_change_mean', 0) 
                            for name in attack_names]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Robustness scores
        axes[0, 0].bar(range(len(attack_names)), robustness_scores)
        axes[0, 0].set_xlabel('Attack Type')
        axes[0, 0].set_ylabel('Robustness Score')
        axes[0, 0].set_title('Robustness Scores by Attack Type')
        axes[0, 0].set_xticks(range(len(attack_names)))
        axes[0, 0].set_xticklabels(attack_names, rotation=45)
        
        # Plot 2: Loss increases
        axes[0, 1].bar(range(len(attack_names)), loss_increases)
        axes[0, 1].set_xlabel('Attack Type')
        axes[0, 1].set_ylabel('Loss Increase')
        axes[0, 1].set_title('Loss Increase by Attack Type')
        axes[0, 1].set_xticks(range(len(attack_names)))
        axes[0, 1].set_xticklabels(attack_names, rotation=45)
        
        # Plot 3: Prediction changes
        axes[1, 0].bar(range(len(attack_names)), prediction_changes)
        axes[1, 0].set_xlabel('Attack Type')
        axes[1, 0].set_ylabel('Prediction Change')
        axes[1, 0].set_title('Prediction Change by Attack Type')
        axes[1, 0].set_xticks(range(len(attack_names)))
        axes[1, 0].set_xticklabels(attack_names, rotation=45)
        
        # Plot 4: Overall metrics
        overall_metrics = results['overall_metrics']
        metric_names = ['Overall Robustness', 'Avg Loss Increase', 'Avg Pred Change']
        metric_values = [
            overall_metrics['overall_robustness_score'],
            overall_metrics['average_loss_increase'],
            overall_metrics['average_prediction_change']
        ]
        
        axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_title('Overall Robustness Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_robustness_report(self, results: Dict) -> str:
        """Generate robustness analysis report."""
        
        attack_results = results['attack_results']
        overall_metrics = results['overall_metrics']
        
        report = "# Adversarial Robustness Analysis Report\n\n"
        
        # Overall assessment
        robustness_score = overall_metrics['overall_robustness_score']
        if robustness_score > 0.8:
            assessment = "ROBUST"
        elif robustness_score > 0.6:
            assessment = "MODERATELY ROBUST"
        else:
            assessment = "VULNERABLE"
        
        report += f"## Overall Assessment: {assessment}\n"
        report += f"- Overall Robustness Score: {robustness_score:.3f}\n"
        report += f"- Average Loss Increase: {overall_metrics['average_loss_increase']:.6f}\n"
        report += f"- Average Prediction Change: {overall_metrics['average_prediction_change']:.6f}\n\n"
        
        # Attack-specific results
        report += "## Attack-Specific Results\n\n"
        
        for attack_name, attack_result in attack_results.items():
            report += f"### {attack_name.upper()} Attack\n"
            report += f"- Robustness Score: {attack_result.get('robustness_score', 0):.3f}\n"
            report += f"- Loss Increase: {attack_result.get('loss_increase', 0):.6f}\n"
            report += f"- Prediction Change: {attack_result.get('prediction_change_mean', 0):.6f}\n"
            report += f"- Perturbation Norm: {attack_result.get('perturbation_norm_mean', 0):.6f}\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if robustness_score < 0.6:
            report += "- Consider adversarial training to improve robustness\n"
            report += "- Implement input preprocessing/filtering\n"
            report += "- Use ensemble methods for more robust predictions\n"
        elif robustness_score < 0.8:
            report += "- Model shows moderate robustness\n"
            report += "- Consider targeted improvements for specific attack types\n"
            report += "- Monitor robustness in deployment\n"
        else:
            report += "- Model demonstrates good robustness\n"
            report += "- Continue monitoring for new attack types\n"
            report += "- Consider robustness as a competitive advantage\n"
        
        return report
