import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
import matplotlib.pyplot as plt
from pathlib import Path

class EnsembleDiversityMetrics:
    """Comprehensive ensemble diversity metrics for model evaluation."""
    
    def __init__(self):
        """Initialize ensemble diversity analyzer."""
        self.diversity_metrics = {}
    
    def compute_pairwise_disagreement(self, predictions: np.ndarray) -> Dict[str, float]:
        """Compute pairwise disagreement between ensemble members."""
        
        n_models = predictions.shape[0]
        n_samples = predictions.shape[1]
        
        # Flatten spatial dimensions if present
        if predictions.ndim > 2:
            pred_flat = predictions.reshape(n_models, n_samples, -1)
            pred_flat = pred_flat.mean(axis=2)  # Average over spatial dimensions
        else:
            pred_flat = predictions
        
        # Compute pairwise distances
        disagreements = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                disagreement = np.mean(np.abs(pred_flat[i] - pred_flat[j]))
                disagreements.append(disagreement)
        
        disagreements = np.array(disagreements)
        
        return {
            'mean_pairwise_disagreement': np.mean(disagreements),
            'std_pairwise_disagreement': np.std(disagreements),
            'max_pairwise_disagreement': np.max(disagreements),
            'min_pairwise_disagreement': np.min(disagreements),
            'disagreement_distribution': disagreements
        }
    
    def compute_diversity_measures(self, predictions: np.ndarray) -> Dict[str, float]:
        """Compute various diversity measures for ensemble."""
        
        n_models = predictions.shape[0]
        
        # Flatten predictions for analysis
        if predictions.ndim > 2:
            pred_flat = predictions.reshape(n_models, -1)
        else:
            pred_flat = predictions
        
        # Q-statistic (Yule's Q)
        q_statistics = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Convert to binary decisions for Q-statistic
                pred_i_binary = (pred_flat[i] > np.median(pred_flat[i])).astype(int)
                pred_j_binary = (pred_flat[j] > np.median(pred_flat[j])).astype(int)
                
                # Compute confusion matrix elements
                n11 = np.sum((pred_i_binary == 1) & (pred_j_binary == 1))
                n10 = np.sum((pred_i_binary == 1) & (pred_j_binary == 0))
                n01 = np.sum((pred_i_binary == 0) & (pred_j_binary == 1))
                n00 = np.sum((pred_i_binary == 0) & (pred_j_binary == 0))
                
                # Q-statistic
                if (n11 * n00 + n10 * n01) > 0:
                    q = (n11 * n00 - n10 * n01) / (n11 * n00 + n10 * n01)
                    q_statistics.append(q)
        
        # Correlation coefficient
        correlation_matrix = np.corrcoef(pred_flat)
        # Remove diagonal elements
        off_diagonal = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        # Disagreement measure
        disagreement_data = self.compute_pairwise_disagreement(predictions)
        
        # Entropy-based diversity
        ensemble_mean = np.mean(pred_flat, axis=0)
        individual_entropies = []
        for i in range(n_models):
            # Discretize predictions for entropy calculation
            pred_discrete = np.digitize(pred_flat[i], np.percentile(pred_flat[i], [25, 50, 75]))
            individual_entropies.append(entropy(np.bincount(pred_discrete) + 1e-8))
        
        ensemble_discrete = np.digitize(ensemble_mean, np.percentile(ensemble_mean, [25, 50, 75]))
        ensemble_entropy = entropy(np.bincount(ensemble_discrete) + 1e-8)
        
        return {
            'mean_q_statistic': np.mean(q_statistics) if q_statistics else 0.0,
            'std_q_statistic': np.std(q_statistics) if q_statistics else 0.0,
            'mean_correlation': np.mean(off_diagonal),
            'std_correlation': np.std(off_diagonal),
            'max_correlation': np.max(off_diagonal),
            'min_correlation': np.min(off_diagonal),
            'mean_pairwise_disagreement': disagreement_data['mean_pairwise_disagreement'],
            'diversity_score': 1.0 - np.mean(np.abs(off_diagonal)),  # Higher is more diverse
            'mean_individual_entropy': np.mean(individual_entropies),
            'ensemble_entropy': ensemble_entropy,
            'entropy_diversity': np.mean(individual_entropies) - ensemble_entropy
        }
    
    def compute_bias_variance_decomposition(self, predictions: np.ndarray, 
                                          targets: np.ndarray) -> Dict[str, float]:
        """Compute bias-variance decomposition for ensemble."""
        
        # Flatten spatial dimensions
        if predictions.ndim > 2:
            pred_flat = predictions.reshape(predictions.shape[0], -1)
            target_flat = targets.flatten()
        else:
            pred_flat = predictions
            target_flat = targets
        
        # Ensemble mean prediction
        ensemble_mean = np.mean(pred_flat, axis=0)
        
        # Bias: squared difference between ensemble mean and true values
        bias_squared = np.mean((ensemble_mean - target_flat) ** 2)
        
        # Variance: expected squared difference between individual predictions and ensemble mean
        individual_variances = []
        for i in range(pred_flat.shape[0]):
            individual_variances.append(np.mean((pred_flat[i] - ensemble_mean) ** 2))
        
        variance = np.mean(individual_variances)
        
        # Noise: irreducible error (estimated from ensemble variance)
        noise = np.var(target_flat)
        
        # Total error
        total_error = np.mean((ensemble_mean - target_flat) ** 2)
        
        return {
            'bias_squared': bias_squared,
            'variance': variance,
            'noise': noise,
            'total_error': total_error,
            'bias_variance_ratio': bias_squared / (variance + 1e-8),
            'explained_variance': 1.0 - total_error / (np.var(target_flat) + 1e-8)
        }
    
    def compute_ensemble_strength(self, predictions: np.ndarray, 
                                targets: np.ndarray) -> Dict[str, float]:
        """Compute ensemble strength metrics."""
        
        n_models = predictions.shape[0]
        
        # Individual model errors
        individual_errors = []
        for i in range(n_models):
            if predictions.ndim > 2:
                pred_flat = predictions[i].flatten()
                target_flat = targets.flatten()
            else:
                pred_flat = predictions[i]
                target_flat = targets
            
            error = np.mean((pred_flat - target_flat) ** 2)
            individual_errors.append(error)
        
        # Ensemble error
        if predictions.ndim > 2:
            ensemble_pred = np.mean(predictions, axis=0).flatten()
            target_flat = targets.flatten()
        else:
            ensemble_pred = np.mean(predictions, axis=0)
            target_flat = targets
        
        ensemble_error = np.mean((ensemble_pred - target_flat) ** 2)
        
        # Ensemble strength
        mean_individual_error = np.mean(individual_errors)
        ensemble_improvement = (mean_individual_error - ensemble_error) / mean_individual_error
        
        # Diversity vs accuracy trade-off
        diversity_data = self.compute_diversity_measures(predictions)
        diversity_score = diversity_data['diversity_score']
        
        return {
            'mean_individual_error': mean_individual_error,
            'ensemble_error': ensemble_error,
            'ensemble_improvement': ensemble_improvement,
            'diversity_score': diversity_score,
            'strength_diversity_product': ensemble_improvement * diversity_score,
            'individual_error_std': np.std(individual_errors),
            'best_individual_error': np.min(individual_errors),
            'worst_individual_error': np.max(individual_errors)
        }
    
    def compute_prediction_intervals(self, predictions: np.ndarray,
                                   confidence_levels: List[float] = [0.68, 0.90, 0.95]) -> Dict[str, np.ndarray]:
        """Compute prediction intervals from ensemble."""
        
        intervals = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(predictions, upper_percentile, axis=0)
            
            intervals[f'lower_{int(confidence*100)}'] = lower_bound
            intervals[f'upper_{int(confidence*100)}'] = upper_bound
            intervals[f'width_{int(confidence*100)}'] = upper_bound - lower_bound
        
        return intervals
    
    def analyze_ensemble_convergence(self, predictions: np.ndarray) -> Dict[str, np.ndarray]:
        """Analyze how ensemble performance changes with number of models."""
        
        n_models = predictions.shape[0]
        convergence_data = {
            'ensemble_sizes': [],
            'ensemble_variances': [],
            'pairwise_disagreements': []
        }
        
        for size in range(2, n_models + 1):
            subset_predictions = predictions[:size]
            
            # Ensemble variance
            ensemble_mean = np.mean(subset_predictions, axis=0)
            ensemble_var = np.var(subset_predictions, axis=0)
            mean_ensemble_var = np.mean(ensemble_var)
            
            # Pairwise disagreement
            disagreement_data = self.compute_pairwise_disagreement(subset_predictions)
            mean_disagreement = disagreement_data['mean_pairwise_disagreement']
            
            convergence_data['ensemble_sizes'].append(size)
            convergence_data['ensemble_variances'].append(mean_ensemble_var)
            convergence_data['pairwise_disagreements'].append(mean_disagreement)
        
        return {
            'ensemble_sizes': np.array(convergence_data['ensemble_sizes']),
            'ensemble_variances': np.array(convergence_data['ensemble_variances']),
            'pairwise_disagreements': np.array(convergence_data['pairwise_disagreements'])
        }
    
    def comprehensive_ensemble_analysis(self, predictions: np.ndarray,
                                      targets: Optional[np.ndarray] = None) -> Dict[str, Dict]:
        """Run comprehensive ensemble diversity analysis."""
        
        print("Running comprehensive ensemble diversity analysis...")
        
        results = {}
        
        # Basic diversity measures
        print("  - Computing diversity measures...")
        results['diversity_measures'] = self.compute_diversity_measures(predictions)
        
        # Ensemble strength
        if targets is not None:
            print("  - Computing ensemble strength...")
            results['ensemble_strength'] = self.compute_ensemble_strength(predictions, targets)
            
            print("  - Computing bias-variance decomposition...")
            results['bias_variance'] = self.compute_bias_variance_decomposition(predictions, targets)
        
        # Prediction intervals
        print("  - Computing prediction intervals...")
        results['prediction_intervals'] = self.compute_prediction_intervals(predictions)
        
        # Convergence analysis
        print("  - Analyzing ensemble convergence...")
        results['convergence_analysis'] = self.analyze_ensemble_convergence(predictions)
        
        # Overall ensemble quality score
        diversity_score = results['diversity_measures']['diversity_score']
        if targets is not None:
            ensemble_improvement = results['ensemble_strength']['ensemble_improvement']
            overall_score = 0.5 * diversity_score + 0.5 * ensemble_improvement
        else:
            overall_score = diversity_score
        
        results['overall'] = {
            'ensemble_quality_score': overall_score,
            'n_models': predictions.shape[0],
            'recommendation': self._get_ensemble_recommendation(results)
        }
        
        return results
    
    def _get_ensemble_recommendation(self, results: Dict) -> str:
        """Get recommendation based on ensemble analysis."""
        
        diversity_score = results['diversity_measures']['diversity_score']
        mean_correlation = results['diversity_measures']['mean_correlation']
        
        if diversity_score > 0.8:
            if mean_correlation < 0.3:
                return "EXCELLENT: High diversity, low correlation. Ensemble is well-balanced."
            else:
                return "GOOD: High diversity but moderate correlation. Consider more diverse training."
        elif diversity_score > 0.6:
            if mean_correlation < 0.5:
                return "MODERATE: Reasonable diversity. Could benefit from more diverse models."
            else:
                return "MODERATE: Limited diversity due to high correlation. Diversify training strategies."
        else:
            return "POOR: Low diversity, high correlation. Ensemble may not provide significant benefits."
    
    def plot_ensemble_analysis(self, results: Dict, save_path: Optional[str] = None):
        """Plot ensemble diversity analysis results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Diversity measures
        if 'diversity_measures' in results:
            diversity_data = results['diversity_measures']
            
            measures = ['Diversity Score', 'Mean Correlation', 'Mean Disagreement']
            values = [
                diversity_data['diversity_score'],
                abs(diversity_data['mean_correlation']),  # Absolute for visualization
                diversity_data['mean_pairwise_disagreement']
            ]
            
            axes[0, 0].bar(measures, values)
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].set_title('Ensemble Diversity Measures')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Bias-Variance decomposition
        if 'bias_variance' in results:
            bv_data = results['bias_variance']
            
            components = ['Bias²', 'Variance', 'Total Error']
            values = [bv_data['bias_squared'], bv_data['variance'], bv_data['total_error']]
            
            axes[0, 1].bar(components, values)
            axes[0, 1].set_ylabel('Error')
            axes[0, 1].set_title('Bias-Variance Decomposition')
        
        # Plot 3: Ensemble strength
        if 'ensemble_strength' in results:
            strength_data = results['ensemble_strength']
            
            errors = ['Mean Individual', 'Ensemble', 'Best Individual']
            values = [
                strength_data['mean_individual_error'],
                strength_data['ensemble_error'],
                strength_data['best_individual_error']
            ]
            
            axes[0, 2].bar(errors, values)
            axes[0, 2].set_ylabel('Error')
            axes[0, 2].set_title('Ensemble vs Individual Performance')
            axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Plot 4: Convergence analysis
        if 'convergence_analysis' in results:
            conv_data = results['convergence_analysis']
            
            axes[1, 0].plot(conv_data['ensemble_sizes'], conv_data['ensemble_variances'], 'bo-', label='Variance')
            axes[1, 0].set_xlabel('Ensemble Size')
            axes[1, 0].set_ylabel('Ensemble Variance')
            axes[1, 0].set_title('Ensemble Convergence')
            axes[1, 0].grid(True)
        
        # Plot 5: Prediction interval widths
        if 'prediction_intervals' in results:
            intervals = results['prediction_intervals']
            
            confidence_levels = [68, 90, 95]
            widths = []
            
            for level in confidence_levels:
                width_key = f'width_{level}'
                if width_key in intervals:
                    widths.append(np.mean(intervals[width_key]))
                else:
                    widths.append(0)
            
            axes[1, 1].bar([f'{level}%' for level in confidence_levels], widths)
            axes[1, 1].set_ylabel('Mean Interval Width')
            axes[1, 1].set_title('Prediction Interval Widths')
        
        # Plot 6: Overall quality score
        if 'overall' in results:
            quality_score = results['overall']['ensemble_quality_score']
            
            # Create a gauge-like visualization
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            axes[1, 2].plot(theta, r, 'k-', linewidth=2)
            
            # Add score indicator
            score_angle = quality_score * np.pi
            axes[1, 2].plot([score_angle, score_angle], [0, 1], 'r-', linewidth=3)
            axes[1, 2].fill_between(theta[theta <= score_angle], 0, 1, alpha=0.3, color='green')
            
            axes[1, 2].set_xlim(0, np.pi)
            axes[1, 2].set_ylim(0, 1.2)
            axes[1, 2].set_title(f'Ensemble Quality: {quality_score:.3f}')
            axes[1, 2].set_xticks([0, np.pi/2, np.pi])
            axes[1, 2].set_xticklabels(['0', '0.5', '1.0'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_ensemble_report(self, results: Dict) -> str:
        """Generate ensemble diversity analysis report."""
        
        report = "# Ensemble Diversity Analysis Report\n\n"
        
        # Overall assessment
        if 'overall' in results:
            quality_score = results['overall']['ensemble_quality_score']
            n_models = results['overall']['n_models']
            recommendation = results['overall']['recommendation']
            
            report += f"## Overall Assessment\n"
            report += f"- Ensemble Size: {n_models} models\n"
            report += f"- Quality Score: {quality_score:.3f}\n"
            report += f"- Recommendation: {recommendation}\n\n"
        
        # Diversity measures
        if 'diversity_measures' in results:
            diversity = results['diversity_measures']
            report += "## Diversity Measures\n"
            report += f"- Diversity Score: {diversity['diversity_score']:.3f}\n"
            report += f"- Mean Correlation: {diversity['mean_correlation']:.3f}\n"
            report += f"- Mean Pairwise Disagreement: {diversity['mean_pairwise_disagreement']:.4f}\n"
            report += f"- Mean Q-Statistic: {diversity['mean_q_statistic']:.3f}\n\n"
        
        # Ensemble strength
        if 'ensemble_strength' in results:
            strength = results['ensemble_strength']
            report += "## Ensemble Performance\n"
            report += f"- Mean Individual Error: {strength['mean_individual_error']:.4f}\n"
            report += f"- Ensemble Error: {strength['ensemble_error']:.4f}\n"
            report += f"- Ensemble Improvement: {strength['ensemble_improvement']:.1%}\n"
            report += f"- Best Individual Error: {strength['best_individual_error']:.4f}\n\n"
        
        # Bias-variance decomposition
        if 'bias_variance' in results:
            bv = results['bias_variance']
            report += "## Bias-Variance Analysis\n"
            report += f"- Bias²: {bv['bias_squared']:.4f}\n"
            report += f"- Variance: {bv['variance']:.4f}\n"
            report += f"- Total Error: {bv['total_error']:.4f}\n"
            report += f"- Bias/Variance Ratio: {bv['bias_variance_ratio']:.3f}\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if 'diversity_measures' in results:
            diversity_score = results['diversity_measures']['diversity_score']
            mean_correlation = results['diversity_measures']['mean_correlation']
            
            if diversity_score < 0.5:
                report += "- **Low Diversity**: Consider using different architectures, training procedures, or data augmentation\n"
                report += "- **High Correlation**: Models are too similar; diversify training strategies\n"
            elif mean_correlation > 0.7:
                report += "- **High Correlation**: Reduce model similarity through diverse initialization or training\n"
            else:
                report += "- **Good Diversity**: Ensemble shows appropriate model diversity\n"
        
        if 'ensemble_strength' in results:
            improvement = results['ensemble_strength']['ensemble_improvement']
            
            if improvement < 0.1:
                report += "- **Limited Ensemble Benefit**: Consider more diverse models or different ensemble strategies\n"
            elif improvement > 0.3:
                report += "- **Strong Ensemble**: Significant improvement over individual models\n"
            else:
                report += "- **Moderate Ensemble Benefit**: Room for improvement in ensemble composition\n"
        
        return report
