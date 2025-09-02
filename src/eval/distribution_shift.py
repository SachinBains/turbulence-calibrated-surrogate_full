import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class DistributionShiftDetector:
    """Detect distribution shifts in turbulence data."""
    
    def __init__(self, reference_data: torch.Tensor, device: torch.device = None):
        """
        Initialize shift detector with reference data.
        
        Args:
            reference_data: Reference dataset (training data)
            device: Computation device
        """
        self.device = device or torch.device('cpu')
        self.reference_data = reference_data.to(self.device)
        self.reference_stats = self._compute_statistics(reference_data)
        
        # Fit PCA for dimensionality reduction
        ref_flat = reference_data.view(reference_data.size(0), -1).cpu().numpy()
        self.pca = PCA(n_components=min(50, ref_flat.shape[1]))
        self.reference_pca = self.pca.fit_transform(ref_flat)
        
    def _compute_statistics(self, data: torch.Tensor) -> Dict[str, float]:
        """Compute statistical properties of data."""
        data_flat = data.view(data.size(0), -1)
        
        stats = {
            'mean': torch.mean(data_flat, dim=1).cpu().numpy(),
            'std': torch.std(data_flat, dim=1).cpu().numpy(),
            'skewness': self._compute_skewness(data_flat).cpu().numpy(),
            'kurtosis': self._compute_kurtosis(data_flat).cpu().numpy(),
            'energy': torch.mean(data_flat**2, dim=1).cpu().numpy(),
            'gradient_norm': self._compute_gradient_norm(data).cpu().numpy()
        }
        
        return stats
    
    def _compute_skewness(self, data: torch.Tensor) -> torch.Tensor:
        """Compute skewness of data."""
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        skewness = torch.mean(normalized**3, dim=1)
        return skewness
    
    def _compute_kurtosis(self, data: torch.Tensor) -> torch.Tensor:
        """Compute kurtosis of data."""
        mean = torch.mean(data, dim=1, keepdim=True)
        std = torch.std(data, dim=1, keepdim=True)
        normalized = (data - mean) / (std + 1e-8)
        kurtosis = torch.mean(normalized**4, dim=1) - 3  # Excess kurtosis
        return kurtosis
    
    def _compute_gradient_norm(self, data: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradient norm."""
        if data.dim() == 4:  # (N, C, H, W) - 2D
            grad_x = torch.diff(data, dim=2)
            grad_y = torch.diff(data, dim=3)
            grad_norm = torch.sqrt(grad_x[:, :, :, :-1]**2 + grad_y[:, :, :-1, :]**2)
        elif data.dim() == 5:  # (N, C, D, H, W) - 3D
            grad_x = torch.diff(data, dim=2)
            grad_y = torch.diff(data, dim=3)
            grad_z = torch.diff(data, dim=4)
            grad_norm = torch.sqrt(
                grad_x[:, :, :, :, :-1]**2 + 
                grad_y[:, :, :, :-1, :]**2 + 
                grad_z[:, :, :-1, :, :]**2
            )
        else:
            raise ValueError(f"Unsupported data dimension: {data.dim()}")
        
        return torch.mean(grad_norm.view(data.size(0), -1), dim=1)
    
    def detect_shift_statistical(self, test_data: torch.Tensor, 
                                alpha: float = 0.05) -> Dict[str, Any]:
        """Detect distribution shift using statistical tests."""
        test_stats = self._compute_statistics(test_data)
        
        results = {}
        
        for stat_name in self.reference_stats.keys():
            ref_values = self.reference_stats[stat_name]
            test_values = test_stats[stat_name]
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.ks_2samp(ref_values, test_values)
            
            # Mann-Whitney U test
            mw_stat, mw_pvalue = stats.mannwhitneyu(
                ref_values, test_values, alternative='two-sided'
            )
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                (np.var(ref_values) + np.var(test_values)) / 2
            )
            cohens_d = (np.mean(test_values) - np.mean(ref_values)) / pooled_std
            
            results[stat_name] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'ks_significant': ks_pvalue < alpha,
                'mw_statistic': mw_stat,
                'mw_pvalue': mw_pvalue,
                'mw_significant': mw_pvalue < alpha,
                'cohens_d': cohens_d,
                'ref_mean': np.mean(ref_values),
                'test_mean': np.mean(test_values),
                'ref_std': np.std(ref_values),
                'test_std': np.std(test_values)
            }
        
        # Overall shift detection
        significant_tests = sum(
            1 for stat_result in results.values() 
            if stat_result['ks_significant'] or stat_result['mw_significant']
        )
        
        results['summary'] = {
            'total_tests': len(results) - 1,  # Exclude summary itself
            'significant_tests': significant_tests,
            'shift_detected': significant_tests > len(results) // 2,
            'shift_confidence': significant_tests / (len(results) - 1)
        }
        
        return results
    
    def detect_shift_mmd(self, test_data: torch.Tensor, 
                        sigma: float = 1.0) -> Dict[str, float]:
        """Detect distribution shift using Maximum Mean Discrepancy."""
        
        # Flatten data
        ref_flat = self.reference_data.view(self.reference_data.size(0), -1)
        test_flat = test_data.view(test_data.size(0), -1).to(self.device)
        
        # Compute MMD
        mmd_value = self._compute_mmd(ref_flat, test_flat, sigma)
        
        # Compute MMD in PCA space for efficiency
        test_pca = self.pca.transform(test_flat.cpu().numpy())
        ref_pca_tensor = torch.tensor(self.reference_pca, dtype=torch.float32).to(self.device)
        test_pca_tensor = torch.tensor(test_pca, dtype=torch.float32).to(self.device)
        
        mmd_pca = self._compute_mmd(ref_pca_tensor, test_pca_tensor, sigma)
        
        return {
            'mmd_full': mmd_value.item(),
            'mmd_pca': mmd_pca.item(),
            'shift_detected_full': mmd_value > 0.1,  # Threshold
            'shift_detected_pca': mmd_pca > 0.05
        }
    
    def _compute_mmd(self, X: torch.Tensor, Y: torch.Tensor, 
                    sigma: float) -> torch.Tensor:
        """Compute Maximum Mean Discrepancy with RBF kernel."""
        
        def rbf_kernel(X, Y, sigma):
            """RBF kernel computation."""
            X_norm = (X**2).sum(1).view(-1, 1)
            Y_norm = (Y**2).sum(1).view(1, -1)
            dist = X_norm + Y_norm - 2.0 * torch.mm(X, Y.transpose(0, 1))
            return torch.exp(-dist / (2 * sigma**2))
        
        m, n = X.size(0), Y.size(0)
        
        # Kernel matrices
        K_XX = rbf_kernel(X, X, sigma)
        K_YY = rbf_kernel(Y, Y, sigma)
        K_XY = rbf_kernel(X, Y, sigma)
        
        # MMD computation
        mmd = (K_XX.sum() / (m * m) + 
               K_YY.sum() / (n * n) - 
               2 * K_XY.sum() / (m * n))
        
        return mmd
    
    def detect_shift_classifier(self, test_data: torch.Tensor) -> Dict[str, float]:
        """Detect distribution shift using binary classifier approach."""
        
        # Prepare data
        ref_flat = self.reference_data.view(self.reference_data.size(0), -1).cpu().numpy()
        test_flat = test_data.view(test_data.size(0), -1).cpu().numpy()
        
        # Use PCA features
        ref_pca = self.reference_pca
        test_pca = self.pca.transform(test_flat)
        
        # Create binary classification dataset
        X = np.vstack([ref_pca, test_pca])
        y = np.hstack([np.zeros(len(ref_pca)), np.ones(len(test_pca))])
        
        # Train simple classifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
        
        # Fit classifier for feature importance
        clf.fit(X, y)
        
        return {
            'classifier_auc': np.mean(cv_scores),
            'classifier_auc_std': np.std(cv_scores),
            'shift_detected': np.mean(cv_scores) > 0.6,  # Threshold
            'feature_importance': clf.feature_importances_[:10].tolist()  # Top 10
        }
    
    def comprehensive_shift_analysis(self, test_data: torch.Tensor) -> Dict[str, Any]:
        """Run comprehensive distribution shift analysis."""
        
        print("Running comprehensive distribution shift analysis...")
        
        results = {}
        
        # Statistical tests
        print("  - Statistical tests...")
        results['statistical'] = self.detect_shift_statistical(test_data)
        
        # MMD test
        print("  - MMD test...")
        results['mmd'] = self.detect_shift_mmd(test_data)
        
        # Classifier test
        print("  - Classifier test...")
        results['classifier'] = self.detect_shift_classifier(test_data)
        
        # Overall assessment
        shift_indicators = [
            results['statistical']['summary']['shift_detected'],
            results['mmd']['shift_detected_pca'],
            results['classifier']['shift_detected']
        ]
        
        results['overall'] = {
            'shift_detected': sum(shift_indicators) >= 2,
            'confidence': sum(shift_indicators) / len(shift_indicators),
            'methods_agreeing': sum(shift_indicators)
        }
        
        return results
    
    def plot_shift_analysis(self, test_data: torch.Tensor, results: Dict,
                           save_path: Optional[str] = None):
        """Plot distribution shift analysis results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Statistical comparison
        stat_names = ['mean', 'std', 'energy']
        test_stats = self._compute_statistics(test_data)
        
        for i, stat_name in enumerate(stat_names):
            if i < 3:
                ref_values = self.reference_stats[stat_name]
                test_values = test_stats[stat_name]
                
                axes[0, i].hist(ref_values, alpha=0.7, label='Reference', bins=20)
                axes[0, i].hist(test_values, alpha=0.7, label='Test', bins=20)
                axes[0, i].set_xlabel(stat_name.title())
                axes[0, i].set_ylabel('Frequency')
                axes[0, i].set_title(f'{stat_name.title()} Distribution')
                axes[0, i].legend()
        
        # Plot 2: PCA projection
        test_pca = self.pca.transform(
            test_data.view(test_data.size(0), -1).cpu().numpy()
        )
        
        axes[1, 0].scatter(self.reference_pca[:, 0], self.reference_pca[:, 1], 
                          alpha=0.6, label='Reference', s=10)
        axes[1, 0].scatter(test_pca[:, 0], test_pca[:, 1], 
                          alpha=0.6, label='Test', s=10)
        axes[1, 0].set_xlabel('PC1')
        axes[1, 0].set_ylabel('PC2')
        axes[1, 0].set_title('PCA Projection')
        axes[1, 0].legend()
        
        # Plot 3: Shift detection summary
        methods = ['Statistical', 'MMD', 'Classifier']
        detections = [
            results['statistical']['summary']['shift_detected'],
            results['mmd']['shift_detected_pca'],
            results['classifier']['shift_detected']
        ]
        
        colors = ['red' if d else 'green' for d in detections]
        axes[1, 1].bar(methods, [1 if d else 0 for d in detections], color=colors)
        axes[1, 1].set_ylabel('Shift Detected')
        axes[1, 1].set_title('Shift Detection Summary')
        axes[1, 1].set_ylim(0, 1.2)
        
        # Add text annotations
        for i, (method, detection) in enumerate(zip(methods, detections)):
            axes[1, 1].text(i, 0.5, 'YES' if detection else 'NO', 
                           ha='center', va='center', fontweight='bold')
        
        # Plot 4: Confidence scores
        confidences = [
            results['statistical']['summary']['shift_confidence'],
            results['mmd']['mmd_pca'],
            results['classifier']['classifier_auc']
        ]
        
        axes[1, 2].bar(methods, confidences)
        axes[1, 2].set_ylabel('Confidence Score')
        axes[1, 2].set_title('Shift Detection Confidence')
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

class DomainAdaptationAnalyzer:
    """Analyze domain adaptation for turbulence models."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize domain adaptation analyzer.
        
        Args:
            model: Trained model
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def analyze_domain_gap(self, source_data: torch.Tensor, 
                          target_data: torch.Tensor) -> Dict[str, float]:
        """Analyze domain gap between source and target data."""
        
        # Extract features from model
        source_features = self._extract_features(source_data)
        target_features = self._extract_features(target_data)
        
        # Compute domain gap metrics
        feature_shift = self._compute_feature_shift(source_features, target_features)
        prediction_shift = self._compute_prediction_shift(source_data, target_data)
        
        return {
            'feature_shift': feature_shift,
            'prediction_shift': prediction_shift,
            'domain_gap': (feature_shift + prediction_shift) / 2
        }
    
    def _extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract intermediate features from model."""
        features = []
        
        def hook_fn(module, input, output):
            features.append(output.detach())
        
        # Register hook on intermediate layer
        hook = None
        for name, module in self.model.named_modules():
            if 'conv' in name.lower() and len(list(module.children())) == 0:
                hook = module.register_forward_hook(hook_fn)
                break
        
        with torch.no_grad():
            _ = self.model(data.to(self.device))
        
        if hook:
            hook.remove()
        
        return features[0] if features else data
    
    def _compute_feature_shift(self, source_features: torch.Tensor, 
                              target_features: torch.Tensor) -> float:
        """Compute feature distribution shift."""
        source_flat = source_features.view(source_features.size(0), -1)
        target_flat = target_features.view(target_features.size(0), -1)
        
        # Compute mean and covariance differences
        source_mean = torch.mean(source_flat, dim=0)
        target_mean = torch.mean(target_flat, dim=0)
        
        mean_diff = torch.norm(source_mean - target_mean).item()
        
        return mean_diff
    
    def _compute_prediction_shift(self, source_data: torch.Tensor, 
                                 target_data: torch.Tensor) -> float:
        """Compute prediction distribution shift."""
        with torch.no_grad():
            source_pred = self.model(source_data.to(self.device))
            target_pred = self.model(target_data.to(self.device))
        
        source_mean = torch.mean(source_pred)
        target_mean = torch.mean(target_pred)
        
        return torch.abs(source_mean - target_mean).item()
    
    def suggest_adaptation_strategy(self, domain_gap: float) -> Dict[str, Any]:
        """Suggest domain adaptation strategy based on domain gap."""
        
        if domain_gap < 0.1:
            strategy = "minimal"
            recommendations = [
                "Domain gap is small",
                "Consider fine-tuning last layers only",
                "Monitor performance on target domain"
            ]
        elif domain_gap < 0.5:
            strategy = "moderate"
            recommendations = [
                "Moderate domain gap detected",
                "Consider gradual unfreezing",
                "Use domain adversarial training",
                "Apply data augmentation"
            ]
        else:
            strategy = "aggressive"
            recommendations = [
                "Large domain gap detected",
                "Consider full model retraining",
                "Use domain adaptation techniques",
                "Collect more target domain data"
            ]
        
        return {
            'strategy': strategy,
            'domain_gap': domain_gap,
            'recommendations': recommendations
        }
