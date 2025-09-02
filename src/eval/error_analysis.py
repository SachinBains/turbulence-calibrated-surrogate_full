import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

class ComprehensiveErrorAnalyzer:
    """Comprehensive error analysis and failure mode detection."""
    
    def __init__(self):
        """Initialize error analyzer."""
        self.error_patterns = {}
        self.failure_modes = {}
    
    def compute_error_statistics(self, predictions: np.ndarray, 
                                targets: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive error statistics."""
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        squared_errors = errors ** 2
        
        return {
            'mae': np.mean(abs_errors),
            'mse': np.mean(squared_errors),
            'rmse': np.sqrt(np.mean(squared_errors)),
            'max_error': np.max(abs_errors),
            'min_error': np.min(abs_errors),
            'error_std': np.std(errors),
            'error_skewness': stats.skew(errors.flatten()),
            'error_kurtosis': stats.kurtosis(errors.flatten()),
            'q95_error': np.percentile(abs_errors, 95),
            'q99_error': np.percentile(abs_errors, 99),
            'median_error': np.median(abs_errors)
        }
    
    def detect_outliers(self, predictions: np.ndarray, 
                       targets: np.ndarray,
                       method: str = 'iqr') -> Dict[str, np.ndarray]:
        """Detect outliers in predictions and errors."""
        
        errors = np.abs(predictions - targets)
        
        if method == 'iqr':
            Q1 = np.percentile(errors, 25)
            Q3 = np.percentile(errors, 75)
            IQR = Q3 - Q1
            threshold = Q3 + 1.5 * IQR
            outliers = errors > threshold
            
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(errors.flatten()))
            outliers = z_scores > 3
            
        elif method == 'modified_zscore':
            median = np.median(errors)
            mad = np.median(np.abs(errors - median))
            modified_z_scores = 0.6745 * (errors - median) / mad
            outliers = np.abs(modified_z_scores) > 3.5
        
        return {
            'outlier_mask': outliers,
            'outlier_indices': np.where(outliers),
            'outlier_count': np.sum(outliers),
            'outlier_percentage': np.mean(outliers) * 100
        }
    
    def analyze_error_patterns(self, predictions: np.ndarray,
                              targets: np.ndarray,
                              uncertainties: Optional[np.ndarray] = None) -> Dict:
        """Analyze spatial and temporal error patterns."""
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        patterns = {}
        
        # Spatial error patterns (if data has spatial dimensions)
        if errors.ndim >= 3:
            # Compute error statistics along spatial dimensions
            spatial_mean_error = np.mean(abs_errors, axis=tuple(range(errors.ndim-2)))
            spatial_std_error = np.std(abs_errors, axis=tuple(range(errors.ndim-2)))
            
            patterns['spatial_error_mean'] = spatial_mean_error
            patterns['spatial_error_std'] = spatial_std_error
            patterns['spatial_error_cv'] = spatial_std_error / (spatial_mean_error + 1e-8)
        
        # Error magnitude distribution
        patterns['error_histogram'] = np.histogram(abs_errors.flatten(), bins=50)
        
        # Correlation between errors and uncertainties
        if uncertainties is not None:
            error_unc_corr = np.corrcoef(abs_errors.flatten(), uncertainties.flatten())[0, 1]
            patterns['error_uncertainty_correlation'] = error_unc_corr
        
        return patterns
    
    def cluster_failure_modes(self, predictions: np.ndarray,
                             targets: np.ndarray,
                             n_clusters: int = 5) -> Dict:
        """Cluster samples to identify failure modes."""
        
        # Prepare features for clustering
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # Flatten spatial dimensions
        if errors.ndim > 1:
            error_features = abs_errors.reshape(abs_errors.shape[0], -1)
        else:
            error_features = abs_errors.reshape(-1, 1)
        
        # Standardize features
        scaler = StandardScaler()
        error_features_scaled = scaler.fit_transform(error_features)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(50, error_features_scaled.shape[1]))
        error_features_pca = pca.fit_transform(error_features_scaled)
        
        # Cluster errors
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(error_features_pca)
        
        # Analyze clusters
        cluster_analysis = {}
        for i in range(n_clusters):
            cluster_mask = cluster_labels == i
            cluster_errors = abs_errors[cluster_mask]
            
            cluster_analysis[f'cluster_{i}'] = {
                'size': np.sum(cluster_mask),
                'percentage': np.mean(cluster_mask) * 100,
                'mean_error': np.mean(cluster_errors),
                'std_error': np.std(cluster_errors),
                'max_error': np.max(cluster_errors),
                'sample_indices': np.where(cluster_mask)[0]
            }
        
        return {
            'cluster_labels': cluster_labels,
            'cluster_analysis': cluster_analysis,
            'n_clusters': n_clusters,
            'pca_explained_variance': pca.explained_variance_ratio_
        }
    
    def analyze_prediction_confidence(self, predictions: np.ndarray,
                                    targets: np.ndarray,
                                    uncertainties: np.ndarray) -> Dict:
        """Analyze relationship between prediction confidence and accuracy."""
        
        errors = np.abs(predictions - targets)
        
        # Bin by uncertainty levels
        n_bins = 10
        uncertainty_bins = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
        
        confidence_analysis = {}
        for i in range(n_bins):
            if i == 0:
                mask = uncertainties <= uncertainty_bins[i + 1]
            else:
                mask = (uncertainties > uncertainty_bins[i]) & (uncertainties <= uncertainty_bins[i + 1])
            
            if np.sum(mask) > 0:
                bin_errors = errors[mask]
                bin_uncertainties = uncertainties[mask]
                
                confidence_analysis[f'bin_{i}'] = {
                    'uncertainty_range': [uncertainty_bins[i], uncertainty_bins[i + 1]],
                    'sample_count': np.sum(mask),
                    'mean_error': np.mean(bin_errors),
                    'mean_uncertainty': np.mean(bin_uncertainties),
                    'error_std': np.std(bin_errors)
                }
        
        return confidence_analysis
    
    def comprehensive_error_analysis(self, predictions: np.ndarray,
                                   targets: np.ndarray,
                                   uncertainties: Optional[np.ndarray] = None) -> Dict:
        """Run comprehensive error analysis."""
        
        print("Running comprehensive error analysis...")
        
        results = {}
        
        # Basic error statistics
        print("  - Computing error statistics...")
        results['error_statistics'] = self.compute_error_statistics(predictions, targets)
        
        # Outlier detection
        print("  - Detecting outliers...")
        results['outliers'] = {}
        for method in ['iqr', 'zscore', 'modified_zscore']:
            results['outliers'][method] = self.detect_outliers(predictions, targets, method)
        
        # Error patterns
        print("  - Analyzing error patterns...")
        results['error_patterns'] = self.analyze_error_patterns(predictions, targets, uncertainties)
        
        # Failure mode clustering
        print("  - Clustering failure modes...")
        results['failure_modes'] = self.cluster_failure_modes(predictions, targets)
        
        # Confidence analysis
        if uncertainties is not None:
            print("  - Analyzing prediction confidence...")
            results['confidence_analysis'] = self.analyze_prediction_confidence(
                predictions, targets, uncertainties
            )
        
        # Overall assessment
        error_stats = results['error_statistics']
        outlier_percentage = results['outliers']['iqr']['outlier_percentage']
        
        if error_stats['rmse'] < 0.1 and outlier_percentage < 5:
            assessment = "EXCELLENT"
        elif error_stats['rmse'] < 0.2 and outlier_percentage < 10:
            assessment = "GOOD"
        elif error_stats['rmse'] < 0.5 and outlier_percentage < 20:
            assessment = "MODERATE"
        else:
            assessment = "POOR"
        
        results['overall'] = {
            'assessment': assessment,
            'rmse': error_stats['rmse'],
            'outlier_percentage': outlier_percentage,
            'dominant_failure_mode': self._identify_dominant_failure_mode(results.get('failure_modes', {}))
        }
        
        return results
    
    def _identify_dominant_failure_mode(self, failure_modes: Dict) -> str:
        """Identify the dominant failure mode."""
        
        if 'cluster_analysis' not in failure_modes:
            return "Unknown"
        
        cluster_analysis = failure_modes['cluster_analysis']
        
        # Find cluster with highest error
        max_error = 0
        dominant_cluster = None
        
        for cluster_name, cluster_data in cluster_analysis.items():
            if cluster_data['mean_error'] > max_error:
                max_error = cluster_data['mean_error']
                dominant_cluster = cluster_name
        
        if dominant_cluster:
            cluster_data = cluster_analysis[dominant_cluster]
            if cluster_data['percentage'] > 20:
                return f"High-error cluster ({cluster_data['percentage']:.1f}% of samples)"
            else:
                return f"Isolated failures ({cluster_data['percentage']:.1f}% of samples)"
        
        return "Distributed errors"
    
    def plot_error_analysis(self, results: Dict, save_path: Optional[str] = None):
        """Plot error analysis results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Error distribution
        if 'error_statistics' in results:
            error_stats = results['error_statistics']
            
            metrics = ['MAE', 'RMSE', 'Max Error', 'Q95 Error']
            values = [error_stats['mae'], error_stats['rmse'], 
                     error_stats['max_error'], error_stats['q95_error']]
            
            axes[0, 0].bar(metrics, values)
            axes[0, 0].set_ylabel('Error')
            axes[0, 0].set_title('Error Statistics')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Outlier detection comparison
        if 'outliers' in results:
            methods = list(results['outliers'].keys())
            percentages = [results['outliers'][method]['outlier_percentage'] for method in methods]
            
            axes[0, 1].bar(methods, percentages)
            axes[0, 1].set_ylabel('Outlier Percentage (%)')
            axes[0, 1].set_title('Outlier Detection Methods')
        
        # Plot 3: Failure mode clusters
        if 'failure_modes' in results and 'cluster_analysis' in results['failure_modes']:
            cluster_data = results['failure_modes']['cluster_analysis']
            
            cluster_names = list(cluster_data.keys())
            cluster_sizes = [cluster_data[name]['percentage'] for name in cluster_names]
            cluster_errors = [cluster_data[name]['mean_error'] for name in cluster_names]
            
            scatter = axes[0, 2].scatter(cluster_sizes, cluster_errors, 
                                       s=100, alpha=0.7, c=range(len(cluster_names)))
            axes[0, 2].set_xlabel('Cluster Size (%)')
            axes[0, 2].set_ylabel('Mean Error')
            axes[0, 2].set_title('Failure Mode Clusters')
            
            # Add cluster labels
            for i, name in enumerate(cluster_names):
                axes[0, 2].annotate(f'C{i}', (cluster_sizes[i], cluster_errors[i]))
        
        # Plot 4: Error histogram
        if 'error_patterns' in results and 'error_histogram' in results['error_patterns']:
            hist_data = results['error_patterns']['error_histogram']
            counts, bins = hist_data
            
            axes[1, 0].bar(bins[:-1], counts, width=np.diff(bins), alpha=0.7)
            axes[1, 0].set_xlabel('Absolute Error')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Error Distribution')
        
        # Plot 5: Confidence vs Error
        if 'confidence_analysis' in results:
            conf_data = results['confidence_analysis']
            
            bin_numbers = []
            mean_errors = []
            mean_uncertainties = []
            
            for bin_name, bin_data in conf_data.items():
                if bin_name.startswith('bin_'):
                    bin_numbers.append(int(bin_name.split('_')[1]))
                    mean_errors.append(bin_data['mean_error'])
                    mean_uncertainties.append(bin_data['mean_uncertainty'])
            
            if bin_numbers:
                axes[1, 1].scatter(mean_uncertainties, mean_errors, alpha=0.7)
                axes[1, 1].set_xlabel('Mean Uncertainty')
                axes[1, 1].set_ylabel('Mean Error')
                axes[1, 1].set_title('Uncertainty vs Error')
                axes[1, 1].grid(True)
        
        # Plot 6: Overall assessment
        if 'overall' in results:
            assessment = results['overall']['assessment']
            rmse = results['overall']['rmse']
            
            # Create assessment visualization
            assessments = ['POOR', 'MODERATE', 'GOOD', 'EXCELLENT']
            colors = ['red', 'orange', 'yellow', 'green']
            
            current_idx = assessments.index(assessment) if assessment in assessments else 0
            
            axes[1, 2].bar(assessments, [1, 1, 1, 1], color=['lightgray']*4)
            axes[1, 2].bar([assessment], [1], color=colors[current_idx])
            axes[1, 2].set_ylabel('Assessment')
            axes[1, 2].set_title(f'Overall Assessment\nRMSE: {rmse:.4f}')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_error_report(self, results: Dict) -> str:
        """Generate comprehensive error analysis report."""
        
        report = "# Comprehensive Error Analysis Report\n\n"
        
        # Overall assessment
        if 'overall' in results:
            overall = results['overall']
            report += f"## Overall Assessment: {overall['assessment']}\n"
            report += f"- RMSE: {overall['rmse']:.4f}\n"
            report += f"- Outlier Percentage: {overall['outlier_percentage']:.1f}%\n"
            report += f"- Dominant Failure Mode: {overall['dominant_failure_mode']}\n\n"
        
        # Error statistics
        if 'error_statistics' in results:
            stats = results['error_statistics']
            report += "## Error Statistics\n"
            report += f"- Mean Absolute Error (MAE): {stats['mae']:.4f}\n"
            report += f"- Root Mean Square Error (RMSE): {stats['rmse']:.4f}\n"
            report += f"- Maximum Error: {stats['max_error']:.4f}\n"
            report += f"- 95th Percentile Error: {stats['q95_error']:.4f}\n"
            report += f"- Error Standard Deviation: {stats['error_std']:.4f}\n"
            report += f"- Error Skewness: {stats['error_skewness']:.3f}\n"
            report += f"- Error Kurtosis: {stats['error_kurtosis']:.3f}\n\n"
        
        # Outlier analysis
        if 'outliers' in results:
            report += "## Outlier Detection\n"
            for method, outlier_data in results['outliers'].items():
                percentage = outlier_data['outlier_percentage']
                count = outlier_data['outlier_count']
                report += f"- {method.upper()}: {percentage:.1f}% ({count} samples)\n"
            report += "\n"
        
        # Failure modes
        if 'failure_modes' in results and 'cluster_analysis' in results['failure_modes']:
            report += "## Failure Mode Analysis\n"
            cluster_data = results['failure_modes']['cluster_analysis']
            
            for cluster_name, data in cluster_data.items():
                cluster_id = cluster_name.split('_')[1]
                report += f"### Cluster {cluster_id}\n"
                report += f"- Size: {data['percentage']:.1f}% ({data['size']} samples)\n"
                report += f"- Mean Error: {data['mean_error']:.4f}\n"
                report += f"- Max Error: {data['max_error']:.4f}\n\n"
        
        # Recommendations
        report += "## Recommendations\n\n"
        
        if 'overall' in results:
            assessment = results['overall']['assessment']
            outlier_pct = results['overall']['outlier_percentage']
            
            if assessment == 'POOR':
                report += "- **Critical Issues Detected**: Model requires significant improvement\n"
                report += "- Review training data quality and model architecture\n"
                report += "- Consider ensemble methods or alternative approaches\n"
            elif assessment == 'MODERATE':
                report += "- **Moderate Performance**: Room for improvement\n"
                report += "- Focus on reducing outliers and systematic errors\n"
                report += "- Consider regularization or data augmentation\n"
            else:
                report += "- **Good Performance**: Continue monitoring\n"
                report += "- Focus on edge cases and rare failure modes\n"
            
            if outlier_pct > 10:
                report += "- **High Outlier Rate**: Investigate data quality and preprocessing\n"
            elif outlier_pct > 5:
                report += "- **Moderate Outliers**: Monitor and analyze outlier patterns\n"
        
        return report
