import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

class UncertaintyCalibrationDiagnostics:
    """Comprehensive uncertainty calibration diagnostics and recalibration."""
    
    def __init__(self):
        """Initialize calibration diagnostics."""
        self.calibration_methods = {}
    
    def compute_reliability_diagram(self, uncertainties: np.ndarray, 
                                  errors: np.ndarray, 
                                  n_bins: int = 10) -> Dict[str, np.ndarray]:
        """Compute reliability diagram for uncertainty calibration."""
        
        # Sort by uncertainty
        sorted_indices = np.argsort(uncertainties)
        sorted_uncertainties = uncertainties[sorted_indices]
        sorted_errors = errors[sorted_indices]
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Compute bin statistics
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence interval
            if bin_lower == 0:
                in_bin = (sorted_uncertainties >= bin_lower) & (sorted_uncertainties <= bin_upper)
            else:
                in_bin = (sorted_uncertainties > bin_lower) & (sorted_uncertainties <= bin_upper)
            
            if np.sum(in_bin) > 0:
                bin_confidence = np.mean(sorted_uncertainties[in_bin])
                
                # For regression, accuracy is fraction of errors below threshold
                error_threshold = np.percentile(sorted_errors, 68)  # 1-sigma equivalent
                bin_accuracy = np.mean(sorted_errors[in_bin] <= error_threshold)
                
                bin_confidences.append(bin_confidence)
                bin_accuracies.append(bin_accuracy)
                bin_counts.append(np.sum(in_bin))
            else:
                bin_confidences.append(0.5 * (bin_lower + bin_upper))
                bin_accuracies.append(0.0)
                bin_counts.append(0)
        
        return {
            'bin_confidences': np.array(bin_confidences),
            'bin_accuracies': np.array(bin_accuracies),
            'bin_counts': np.array(bin_counts),
            'bin_boundaries': bin_boundaries
        }
    
    def compute_calibration_metrics(self, uncertainties: np.ndarray,
                                  errors: np.ndarray,
                                  n_bins: int = 10) -> Dict[str, float]:
        """Compute calibration metrics."""
        
        reliability_data = self.compute_reliability_diagram(uncertainties, errors, n_bins)
        
        bin_confidences = reliability_data['bin_confidences']
        bin_accuracies = reliability_data['bin_accuracies']
        bin_counts = reliability_data['bin_counts']
        
        # Expected Calibration Error (ECE)
        total_samples = np.sum(bin_counts)
        if total_samples > 0:
            ece = np.sum(bin_counts * np.abs(bin_confidences - bin_accuracies)) / total_samples
        else:
            ece = 0.0
        
        # Maximum Calibration Error (MCE)
        mce = np.max(np.abs(bin_confidences - bin_accuracies))
        
        # Average Calibration Error (ACE)
        ace = np.mean(np.abs(bin_confidences - bin_accuracies))
        
        # Brier Score
        brier_score = np.mean((uncertainties - (errors <= np.percentile(errors, 68)).astype(float))**2)
        
        # Reliability (weighted average of squared differences)
        reliability = np.sum(bin_counts * (bin_confidences - bin_accuracies)**2) / total_samples if total_samples > 0 else 0.0
        
        # Resolution (weighted variance of bin accuracies)
        overall_accuracy = np.mean(errors <= np.percentile(errors, 68))
        resolution = np.sum(bin_counts * (bin_accuracies - overall_accuracy)**2) / total_samples if total_samples > 0 else 0.0
        
        # Sharpness (variance of predictions)
        sharpness = np.var(uncertainties)
        
        return {
            'ece': ece,
            'mce': mce,
            'ace': ace,
            'brier_score': brier_score,
            'reliability': reliability,
            'resolution': resolution,
            'sharpness': sharpness,
            'calibration_score': 1.0 - ece  # Higher is better
        }
    
    def compute_coverage_analysis(self, predictions: np.ndarray,
                                targets: np.ndarray,
                                uncertainties: np.ndarray) -> Dict[str, float]:
        """Compute coverage analysis for uncertainty intervals."""
        
        errors = np.abs(predictions - targets)
        
        # Coverage at different confidence levels
        coverage_levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
        coverage_results = {}
        
        for level in coverage_levels:
            # Determine threshold for this confidence level
            uncertainty_threshold = np.percentile(uncertainties, level * 100)
            
            # Count samples within uncertainty bounds
            within_bounds = uncertainties >= uncertainty_threshold
            
            if np.sum(within_bounds) > 0:
                # Compute actual coverage
                actual_coverage = np.mean(errors[within_bounds] <= uncertainty_threshold)
                coverage_results[f'coverage_{int(level*100)}'] = actual_coverage
                coverage_results[f'expected_{int(level*100)}'] = level
                coverage_results[f'coverage_gap_{int(level*100)}'] = abs(actual_coverage - level)
        
        # Overall coverage quality
        coverage_gaps = [coverage_results[k] for k in coverage_results.keys() if k.startswith('coverage_gap_')]
        coverage_results['mean_coverage_gap'] = np.mean(coverage_gaps) if coverage_gaps else 0.0
        coverage_results['coverage_quality'] = 1.0 - coverage_results['mean_coverage_gap']
        
        return coverage_results
    
    def fit_platt_scaling(self, uncertainties: np.ndarray, 
                         errors: np.ndarray) -> Dict[str, float]:
        """Fit Platt scaling for uncertainty calibration."""
        
        # Convert to binary classification problem
        error_threshold = np.percentile(errors, 68)
        binary_labels = (errors <= error_threshold).astype(int)
        
        # Fit logistic regression
        platt_model = LogisticRegression()
        uncertainties_reshaped = uncertainties.reshape(-1, 1)
        platt_model.fit(uncertainties_reshaped, binary_labels)
        
        # Get calibrated probabilities
        calibrated_probs = platt_model.predict_proba(uncertainties_reshaped)[:, 1]
        
        # Store calibration parameters
        self.calibration_methods['platt'] = {
            'model': platt_model,
            'threshold': error_threshold
        }
        
        return {
            'platt_slope': platt_model.coef_[0, 0],
            'platt_intercept': platt_model.intercept_[0],
            'calibrated_uncertainties': calibrated_probs
        }
    
    def fit_isotonic_regression(self, uncertainties: np.ndarray,
                               errors: np.ndarray) -> Dict[str, np.ndarray]:
        """Fit isotonic regression for uncertainty calibration."""
        
        # Convert to binary classification problem
        error_threshold = np.percentile(errors, 68)
        binary_labels = (errors <= error_threshold).astype(float)
        
        # Fit isotonic regression
        isotonic_model = IsotonicRegression(out_of_bounds='clip')
        calibrated_probs = isotonic_model.fit_transform(uncertainties, binary_labels)
        
        # Store calibration method
        self.calibration_methods['isotonic'] = {
            'model': isotonic_model,
            'threshold': error_threshold
        }
        
        return {
            'calibrated_uncertainties': calibrated_probs,
            'original_uncertainties': uncertainties
        }
    
    def fit_temperature_scaling(self, logits: torch.Tensor,
                               labels: torch.Tensor) -> Dict[str, float]:
        """Fit temperature scaling for neural network calibration."""
        
        # Temperature scaling parameter
        temperature = nn.Parameter(torch.ones(1) * 1.5)
        
        # Optimize temperature
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            scaled_logits = logits / temperature
            loss = nn.CrossEntropyLoss()(scaled_logits, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        
        # Get calibrated probabilities
        with torch.no_grad():
            calibrated_logits = logits / temperature
            calibrated_probs = torch.softmax(calibrated_logits, dim=1)
        
        self.calibration_methods['temperature'] = {
            'temperature': temperature.item(),
            'calibrated_probs': calibrated_probs.numpy()
        }
        
        return {
            'optimal_temperature': temperature.item(),
            'calibrated_probabilities': calibrated_probs.numpy()
        }
    
    def comprehensive_calibration_analysis(self, predictions: np.ndarray,
                                         targets: np.ndarray,
                                         uncertainties: np.ndarray,
                                         logits: Optional[torch.Tensor] = None) -> Dict[str, Dict]:
        """Run comprehensive calibration analysis."""
        
        print("Running comprehensive uncertainty calibration analysis...")
        
        errors = np.abs(predictions - targets)
        
        results = {}
        
        # Basic calibration metrics
        print("  - Computing calibration metrics...")
        results['calibration_metrics'] = self.compute_calibration_metrics(uncertainties, errors)
        
        # Coverage analysis
        print("  - Computing coverage analysis...")
        results['coverage_analysis'] = self.compute_coverage_analysis(predictions, targets, uncertainties)
        
        # Reliability diagram
        print("  - Computing reliability diagram...")
        results['reliability_diagram'] = self.compute_reliability_diagram(uncertainties, errors)
        
        # Recalibration methods
        print("  - Fitting Platt scaling...")
        results['platt_scaling'] = self.fit_platt_scaling(uncertainties, errors)
        
        print("  - Fitting isotonic regression...")
        results['isotonic_regression'] = self.fit_isotonic_regression(uncertainties, errors)
        
        # Temperature scaling (if logits available)
        if logits is not None:
            print("  - Fitting temperature scaling...")
            # Convert targets to class labels for temperature scaling
            target_classes = (errors <= np.percentile(errors, 68)).astype(int)
            target_tensor = torch.tensor(target_classes, dtype=torch.long)
            results['temperature_scaling'] = self.fit_temperature_scaling(logits, target_tensor)
        
        # Evaluate recalibrated uncertainties
        print("  - Evaluating recalibration methods...")
        results['recalibration_evaluation'] = self.evaluate_recalibration_methods(
            predictions, targets, uncertainties
        )
        
        return results
    
    def evaluate_recalibration_methods(self, predictions: np.ndarray,
                                     targets: np.ndarray,
                                     original_uncertainties: np.ndarray) -> Dict[str, Dict]:
        """Evaluate different recalibration methods."""
        
        errors = np.abs(predictions - targets)
        evaluation_results = {}
        
        # Original calibration
        original_metrics = self.compute_calibration_metrics(original_uncertainties, errors)
        evaluation_results['original'] = original_metrics
        
        # Evaluate each calibration method
        for method_name, method_data in self.calibration_methods.items():
            if method_name == 'platt':
                calibrated_uncertainties = method_data['model'].predict_proba(
                    original_uncertainties.reshape(-1, 1)
                )[:, 1]
            elif method_name == 'isotonic':
                calibrated_uncertainties = method_data['model'].transform(original_uncertainties)
            elif method_name == 'temperature':
                calibrated_uncertainties = np.max(method_data['calibrated_probs'], axis=1)
            else:
                continue
            
            # Compute metrics for calibrated uncertainties
            calibrated_metrics = self.compute_calibration_metrics(calibrated_uncertainties, errors)
            evaluation_results[method_name] = calibrated_metrics
            
            # Improvement metrics
            ece_improvement = original_metrics['ece'] - calibrated_metrics['ece']
            evaluation_results[f'{method_name}_improvement'] = {
                'ece_improvement': ece_improvement,
                'calibration_score_improvement': calibrated_metrics['calibration_score'] - original_metrics['calibration_score']
            }
        
        return evaluation_results
    
    def plot_calibration_analysis(self, results: Dict, save_path: Optional[str] = None):
        """Plot calibration analysis results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Reliability diagram
        if 'reliability_diagram' in results:
            reliability_data = results['reliability_diagram']
            bin_confidences = reliability_data['bin_confidences']
            bin_accuracies = reliability_data['bin_accuracies']
            
            axes[0, 0].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
            axes[0, 0].plot(bin_confidences, bin_accuracies, 'ro-', label='Model')
            axes[0, 0].set_xlabel('Confidence')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].set_title('Reliability Diagram')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Calibration metrics comparison
        if 'calibration_metrics' in results:
            metrics = results['calibration_metrics']
            metric_names = ['ECE', 'MCE', 'ACE', 'Reliability']
            metric_values = [metrics['ece'], metrics['mce'], metrics['ace'], metrics['reliability']]
            
            axes[0, 1].bar(metric_names, metric_values)
            axes[0, 1].set_ylabel('Error')
            axes[0, 1].set_title('Calibration Metrics')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Coverage analysis
        if 'coverage_analysis' in results:
            coverage_data = results['coverage_analysis']
            
            levels = [50, 68, 80, 90, 95, 99]
            expected = [coverage_data.get(f'expected_{level}', level/100) for level in levels]
            actual = [coverage_data.get(f'coverage_{level}', 0) for level in levels]
            
            axes[0, 2].plot(expected, expected, 'k--', label='Perfect Coverage')
            axes[0, 2].plot(expected, actual, 'bo-', label='Actual Coverage')
            axes[0, 2].set_xlabel('Expected Coverage')
            axes[0, 2].set_ylabel('Actual Coverage')
            axes[0, 2].set_title('Coverage Analysis')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Plot 4: Recalibration comparison
        if 'recalibration_evaluation' in results:
            eval_data = results['recalibration_evaluation']
            
            methods = []
            ece_values = []
            
            for method, metrics in eval_data.items():
                if not method.endswith('_improvement') and isinstance(metrics, dict):
                    methods.append(method)
                    ece_values.append(metrics.get('ece', 0))
            
            if methods:
                axes[1, 0].bar(methods, ece_values)
                axes[1, 0].set_ylabel('Expected Calibration Error')
                axes[1, 0].set_title('Recalibration Methods Comparison')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Calibration improvement
        if 'recalibration_evaluation' in results:
            eval_data = results['recalibration_evaluation']
            
            improvements = []
            method_names = []
            
            for key, value in eval_data.items():
                if key.endswith('_improvement') and isinstance(value, dict):
                    method_name = key.replace('_improvement', '')
                    method_names.append(method_name)
                    improvements.append(value.get('ece_improvement', 0))
            
            if method_names:
                colors = ['green' if imp > 0 else 'red' for imp in improvements]
                axes[1, 1].bar(method_names, improvements, color=colors)
                axes[1, 1].set_ylabel('ECE Improvement')
                axes[1, 1].set_title('Calibration Improvement')
                axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Overall calibration score
        if 'calibration_metrics' in results:
            calibration_score = results['calibration_metrics']['calibration_score']
            
            # Create a gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            axes[1, 2].plot(theta, r, 'k-', linewidth=2)
            
            # Add score indicator
            score_angle = calibration_score * np.pi
            axes[1, 2].plot([score_angle, score_angle], [0, 1], 'r-', linewidth=3)
            axes[1, 2].fill_between(theta[theta <= score_angle], 0, 1, alpha=0.3, color='green')
            
            axes[1, 2].set_xlim(0, np.pi)
            axes[1, 2].set_ylim(0, 1.2)
            axes[1, 2].set_title(f'Calibration Score: {calibration_score:.3f}')
            axes[1, 2].set_xticks([0, np.pi/2, np.pi])
            axes[1, 2].set_xticklabels(['0', '0.5', '1.0'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_calibration_report(self, results: Dict) -> str:
        """Generate calibration analysis report."""
        
        report = "# Uncertainty Calibration Analysis Report\n\n"
        
        # Overall assessment
        if 'calibration_metrics' in results:
            metrics = results['calibration_metrics']
            calibration_score = metrics['calibration_score']
            
            if calibration_score > 0.9:
                assessment = "EXCELLENT"
            elif calibration_score > 0.8:
                assessment = "GOOD"
            elif calibration_score > 0.6:
                assessment = "MODERATE"
            else:
                assessment = "POOR"
            
            report += f"## Overall Assessment: {assessment}\n"
            report += f"- Calibration Score: {calibration_score:.3f}\n"
            report += f"- Expected Calibration Error: {metrics['ece']:.4f}\n"
            report += f"- Maximum Calibration Error: {metrics['mce']:.4f}\n\n"
        
        # Coverage analysis
        if 'coverage_analysis' in results:
            coverage_data = results['coverage_analysis']
            report += "## Coverage Analysis\n"
            report += f"- Mean Coverage Gap: {coverage_data.get('mean_coverage_gap', 0):.4f}\n"
            report += f"- Coverage Quality: {coverage_data.get('coverage_quality', 0):.3f}\n\n"
            
            # Coverage at different levels
            levels = [68, 90, 95]
            for level in levels:
                expected = coverage_data.get(f'expected_{level}', level/100)
                actual = coverage_data.get(f'coverage_{level}', 0)
                gap = coverage_data.get(f'coverage_gap_{level}', 0)
                report += f"- {level}% Coverage: Expected {expected:.2f}, Actual {actual:.2f}, Gap {gap:.4f}\n"
            
            report += "\n"
        
        # Recalibration results
        if 'recalibration_evaluation' in results:
            eval_data = results['recalibration_evaluation']
            report += "## Recalibration Methods\n"
            
            original_ece = eval_data.get('original', {}).get('ece', 0)
            report += f"- Original ECE: {original_ece:.4f}\n\n"
            
            for method in ['platt', 'isotonic', 'temperature']:
                if method in eval_data:
                    method_ece = eval_data[method]['ece']
                    improvement_key = f'{method}_improvement'
                    
                    if improvement_key in eval_data:
                        improvement = eval_data[improvement_key]['ece_improvement']
                        report += f"### {method.title()} Scaling\n"
                        report += f"- Calibrated ECE: {method_ece:.4f}\n"
                        report += f"- ECE Improvement: {improvement:.4f}\n"
                        report += f"- Status: {'✓ Improved' if improvement > 0 else '✗ No improvement'}\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if 'calibration_metrics' in results:
            ece = results['calibration_metrics']['ece']
            
            if ece > 0.1:
                report += "- High calibration error detected\n"
                report += "- Consider applying recalibration methods\n"
                report += "- Retrain model with calibration-aware loss functions\n"
            elif ece > 0.05:
                report += "- Moderate calibration error\n"
                report += "- Apply post-hoc calibration methods\n"
                report += "- Monitor calibration in deployment\n"
            else:
                report += "- Good calibration achieved\n"
                report += "- Continue monitoring calibration performance\n"
                report += "- Consider calibration as a model selection criterion\n"
        
        return report
