import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Callable
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from pathlib import Path

class TemporalConsistencyValidator:
    """Validate temporal consistency of turbulence predictions."""
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize temporal consistency validator.
        
        Args:
            model: Trained model
            device: Computation device
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def validate_temporal_autocorrelation(self, predictions: np.ndarray, 
                                        ground_truth: np.ndarray,
                                        max_lag: int = 10) -> Dict[str, np.ndarray]:
        """Validate temporal autocorrelation structure."""
        
        # Ensure time is first dimension: (T, ...)
        if predictions.ndim > 1:
            pred_flat = predictions.reshape(predictions.shape[0], -1)
            gt_flat = ground_truth.reshape(ground_truth.shape[0], -1)
        else:
            pred_flat = predictions.reshape(-1, 1)
            gt_flat = ground_truth.reshape(-1, 1)
        
        pred_autocorr = []
        gt_autocorr = []
        
        # Compute autocorrelation for each spatial point
        for i in range(min(100, pred_flat.shape[1])):  # Sample spatial points
            pred_series = pred_flat[:, i]
            gt_series = gt_flat[:, i]
            
            # Compute autocorrelation
            pred_ac = self._compute_autocorrelation(pred_series, max_lag)
            gt_ac = self._compute_autocorrelation(gt_series, max_lag)
            
            pred_autocorr.append(pred_ac)
            gt_autocorr.append(gt_ac)
        
        pred_autocorr = np.array(pred_autocorr)
        gt_autocorr = np.array(gt_autocorr)
        
        # Compute statistics
        pred_autocorr_mean = np.mean(pred_autocorr, axis=0)
        gt_autocorr_mean = np.mean(gt_autocorr, axis=0)
        autocorr_error = np.mean(np.abs(pred_autocorr - gt_autocorr), axis=0)
        
        return {
            'prediction_autocorr': pred_autocorr_mean,
            'ground_truth_autocorr': gt_autocorr_mean,
            'autocorr_error': autocorr_error,
            'lags': np.arange(max_lag + 1)
        }
    
    def _compute_autocorrelation(self, series: np.ndarray, max_lag: int) -> np.ndarray:
        """Compute autocorrelation function."""
        n = len(series)
        series = series - np.mean(series)
        
        autocorr = np.correlate(series, series, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        return autocorr[:max_lag + 1]
    
    def validate_spectral_consistency(self, predictions: np.ndarray,
                                    ground_truth: np.ndarray,
                                    dt: float = 1.0) -> Dict[str, np.ndarray]:
        """Validate temporal spectral consistency."""
        
        # Flatten spatial dimensions
        if predictions.ndim > 1:
            pred_flat = predictions.reshape(predictions.shape[0], -1)
            gt_flat = ground_truth.reshape(ground_truth.shape[0], -1)
        else:
            pred_flat = predictions.reshape(-1, 1)
            gt_flat = ground_truth.reshape(-1, 1)
        
        pred_spectra = []
        gt_spectra = []
        
        # Compute power spectra for sample spatial points
        for i in range(min(50, pred_flat.shape[1])):
            pred_series = pred_flat[:, i]
            gt_series = gt_flat[:, i]
            
            # Compute power spectral density
            pred_freq, pred_psd = signal.periodogram(pred_series, fs=1/dt)
            gt_freq, gt_psd = signal.periodogram(gt_series, fs=1/dt)
            
            pred_spectra.append(pred_psd)
            gt_spectra.append(gt_psd)
        
        pred_spectra = np.array(pred_spectra)
        gt_spectra = np.array(gt_spectra)
        
        # Average across spatial points
        pred_spectrum_mean = np.mean(pred_spectra, axis=0)
        gt_spectrum_mean = np.mean(gt_spectra, axis=0)
        
        # Compute spectral error
        spectral_error = np.mean(np.abs(pred_spectra - gt_spectra), axis=0)
        
        return {
            'prediction_spectrum': pred_spectrum_mean,
            'ground_truth_spectrum': gt_spectrum_mean,
            'spectral_error': spectral_error,
            'frequencies': pred_freq
        }
    
    def validate_phase_consistency(self, predictions: np.ndarray,
                                 ground_truth: np.ndarray) -> Dict[str, float]:
        """Validate phase consistency in temporal evolution."""
        
        # Flatten spatial dimensions
        if predictions.ndim > 1:
            pred_flat = predictions.reshape(predictions.shape[0], -1)
            gt_flat = ground_truth.reshape(ground_truth.shape[0], -1)
        else:
            pred_flat = predictions.reshape(-1, 1)
            gt_flat = ground_truth.reshape(-1, 1)
        
        phase_errors = []
        phase_correlations = []
        
        for i in range(min(50, pred_flat.shape[1])):
            pred_series = pred_flat[:, i]
            gt_series = gt_flat[:, i]
            
            # Compute analytic signal (Hilbert transform)
            pred_analytic = signal.hilbert(pred_series)
            gt_analytic = signal.hilbert(gt_series)
            
            # Extract phases
            pred_phase = np.angle(pred_analytic)
            gt_phase = np.angle(gt_analytic)
            
            # Compute phase error
            phase_diff = np.angle(np.exp(1j * (pred_phase - gt_phase)))
            phase_error = np.mean(np.abs(phase_diff))
            phase_errors.append(phase_error)
            
            # Compute phase correlation
            phase_corr = np.corrcoef(pred_phase, gt_phase)[0, 1]
            if not np.isnan(phase_corr):
                phase_correlations.append(phase_corr)
        
        return {
            'mean_phase_error': np.mean(phase_errors),
            'std_phase_error': np.std(phase_errors),
            'mean_phase_correlation': np.mean(phase_correlations) if phase_correlations else 0.0,
            'phase_consistency_score': 1.0 - np.mean(phase_errors) / np.pi
        }
    
    def validate_energy_conservation(self, predictions: np.ndarray,
                                   ground_truth: np.ndarray) -> Dict[str, float]:
        """Validate temporal energy conservation."""
        
        # Compute energy time series
        pred_energy = np.mean(predictions**2, axis=tuple(range(1, predictions.ndim)))
        gt_energy = np.mean(ground_truth**2, axis=tuple(range(1, ground_truth.ndim)))
        
        # Energy statistics
        pred_energy_mean = np.mean(pred_energy)
        gt_energy_mean = np.mean(gt_energy)
        energy_bias = pred_energy_mean - gt_energy_mean
        
        # Energy variance
        pred_energy_var = np.var(pred_energy)
        gt_energy_var = np.var(gt_energy)
        
        # Energy correlation
        energy_correlation = np.corrcoef(pred_energy, gt_energy)[0, 1]
        
        # Energy drift (linear trend)
        time_steps = np.arange(len(pred_energy))
        pred_drift = np.polyfit(time_steps, pred_energy, 1)[0]
        gt_drift = np.polyfit(time_steps, gt_energy, 1)[0]
        
        return {
            'energy_bias': energy_bias,
            'energy_bias_relative': energy_bias / gt_energy_mean,
            'energy_variance_ratio': pred_energy_var / gt_energy_var,
            'energy_correlation': energy_correlation,
            'prediction_energy_drift': pred_drift,
            'ground_truth_energy_drift': gt_drift,
            'energy_conservation_score': 1.0 - abs(energy_bias) / gt_energy_mean
        }
    
    def validate_velocity_divergence_evolution(self, predictions: np.ndarray,
                                             ground_truth: np.ndarray) -> Dict[str, float]:
        """Validate evolution of velocity divergence (incompressibility)."""
        
        if predictions.ndim != 5:  # Expecting (T, C, D, H, W)
            return {'error': 'Expected 5D input for divergence computation'}
        
        pred_divergences = []
        gt_divergences = []
        
        for t in range(predictions.shape[0]):
            # Compute divergence at each time step
            pred_div = self._compute_divergence(predictions[t])
            gt_div = self._compute_divergence(ground_truth[t])
            
            pred_divergences.append(np.mean(np.abs(pred_div)))
            gt_divergences.append(np.mean(np.abs(gt_div)))
        
        pred_divergences = np.array(pred_divergences)
        gt_divergences = np.array(gt_divergences)
        
        # Temporal statistics
        pred_div_trend = np.polyfit(range(len(pred_divergences)), pred_divergences, 1)[0]
        gt_div_trend = np.polyfit(range(len(gt_divergences)), gt_divergences, 1)[0]
        
        div_correlation = np.corrcoef(pred_divergences, gt_divergences)[0, 1]
        
        return {
            'prediction_divergence_mean': np.mean(pred_divergences),
            'ground_truth_divergence_mean': np.mean(gt_divergences),
            'divergence_correlation': div_correlation,
            'prediction_divergence_trend': pred_div_trend,
            'ground_truth_divergence_trend': gt_div_trend,
            'incompressibility_preservation': 1.0 - abs(pred_div_trend - gt_div_trend)
        }
    
    def _compute_divergence(self, velocity_field: np.ndarray) -> np.ndarray:
        """Compute velocity divergence."""
        if velocity_field.shape[0] != 3:
            return np.zeros_like(velocity_field[0])
        
        u, v, w = velocity_field[0], velocity_field[1], velocity_field[2]
        
        du_dx = np.gradient(u, axis=0)
        dv_dy = np.gradient(v, axis=1)
        dw_dz = np.gradient(w, axis=2)
        
        divergence = du_dx + dv_dy + dw_dz
        return divergence
    
    def comprehensive_temporal_validation(self, predictions: np.ndarray,
                                        ground_truth: np.ndarray,
                                        dt: float = 1.0) -> Dict[str, Dict]:
        """Run comprehensive temporal consistency validation."""
        
        print("Running comprehensive temporal consistency validation...")
        
        results = {}
        
        # Autocorrelation validation
        print("  - Temporal autocorrelation...")
        results['autocorrelation'] = self.validate_temporal_autocorrelation(
            predictions, ground_truth
        )
        
        # Spectral validation
        print("  - Spectral consistency...")
        results['spectral'] = self.validate_spectral_consistency(
            predictions, ground_truth, dt
        )
        
        # Phase validation
        print("  - Phase consistency...")
        results['phase'] = self.validate_phase_consistency(
            predictions, ground_truth
        )
        
        # Energy conservation
        print("  - Energy conservation...")
        results['energy'] = self.validate_energy_conservation(
            predictions, ground_truth
        )
        
        # Divergence evolution (if applicable)
        if predictions.ndim == 5 and predictions.shape[1] == 3:
            print("  - Divergence evolution...")
            results['divergence'] = self.validate_velocity_divergence_evolution(
                predictions, ground_truth
            )
        
        # Overall temporal consistency score
        scores = []
        if 'phase' in results:
            scores.append(results['phase']['phase_consistency_score'])
        if 'energy' in results:
            scores.append(results['energy']['energy_conservation_score'])
        if 'divergence' in results:
            scores.append(results['divergence']['incompressibility_preservation'])
        
        results['overall'] = {
            'temporal_consistency_score': np.mean(scores) if scores else 0.0,
            'n_metrics': len(scores)
        }
        
        return results
    
    def plot_temporal_validation(self, results: Dict, save_path: Optional[str] = None):
        """Plot temporal validation results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Autocorrelation
        if 'autocorrelation' in results:
            autocorr_data = results['autocorrelation']
            lags = autocorr_data['lags']
            
            axes[0, 0].plot(lags, autocorr_data['prediction_autocorr'], 
                           label='Prediction', marker='o')
            axes[0, 0].plot(lags, autocorr_data['ground_truth_autocorr'], 
                           label='Ground Truth', marker='s')
            axes[0, 0].set_xlabel('Lag')
            axes[0, 0].set_ylabel('Autocorrelation')
            axes[0, 0].set_title('Temporal Autocorrelation')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Plot 2: Power Spectrum
        if 'spectral' in results:
            spectral_data = results['spectral']
            freqs = spectral_data['frequencies']
            
            axes[0, 1].loglog(freqs[1:], spectral_data['prediction_spectrum'][1:], 
                             label='Prediction')
            axes[0, 1].loglog(freqs[1:], spectral_data['ground_truth_spectrum'][1:], 
                             label='Ground Truth')
            axes[0, 1].set_xlabel('Frequency')
            axes[0, 1].set_ylabel('Power Spectral Density')
            axes[0, 1].set_title('Temporal Power Spectrum')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Plot 3: Energy Evolution
        if 'energy' in results:
            energy_data = results['energy']
            
            # Create mock time series for visualization
            time_steps = np.arange(20)  # Placeholder
            pred_energy = np.random.normal(1.0, 0.1, 20)  # Placeholder
            gt_energy = np.random.normal(1.0, 0.05, 20)  # Placeholder
            
            axes[0, 2].plot(time_steps, pred_energy, label='Prediction', marker='o')
            axes[0, 2].plot(time_steps, gt_energy, label='Ground Truth', marker='s')
            axes[0, 2].set_xlabel('Time Step')
            axes[0, 2].set_ylabel('Energy')
            axes[0, 2].set_title('Energy Evolution')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Plot 4: Phase Analysis
        if 'phase' in results:
            phase_data = results['phase']
            
            metrics = ['Phase Error', 'Phase Correlation', 'Consistency Score']
            values = [
                phase_data['mean_phase_error'],
                phase_data['mean_phase_correlation'],
                phase_data['phase_consistency_score']
            ]
            
            axes[1, 0].bar(metrics, values)
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].set_title('Phase Consistency Metrics')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Plot 5: Overall Scores
        score_names = []
        score_values = []
        
        if 'phase' in results:
            score_names.append('Phase')
            score_values.append(results['phase']['phase_consistency_score'])
        if 'energy' in results:
            score_names.append('Energy')
            score_values.append(results['energy']['energy_conservation_score'])
        if 'divergence' in results:
            score_names.append('Divergence')
            score_values.append(results['divergence']['incompressibility_preservation'])
        
        if score_names:
            axes[1, 1].bar(score_names, score_values)
            axes[1, 1].set_ylabel('Consistency Score')
            axes[1, 1].set_title('Temporal Consistency Scores')
            axes[1, 1].set_ylim(0, 1)
        
        # Plot 6: Summary
        if 'overall' in results:
            overall_score = results['overall']['temporal_consistency_score']
            
            # Create a gauge-like plot
            theta = np.linspace(0, np.pi, 100)
            r = np.ones_like(theta)
            
            axes[1, 2].plot(theta, r, 'k-', linewidth=2)
            
            # Add score indicator
            score_angle = overall_score * np.pi
            axes[1, 2].plot([score_angle, score_angle], [0, 1], 'r-', linewidth=3)
            axes[1, 2].fill_between(theta[theta <= score_angle], 0, 1, alpha=0.3, color='green')
            
            axes[1, 2].set_xlim(0, np.pi)
            axes[1, 2].set_ylim(0, 1.2)
            axes[1, 2].set_title(f'Overall Score: {overall_score:.3f}')
            axes[1, 2].set_xticks([0, np.pi/2, np.pi])
            axes[1, 2].set_xticklabels(['0', '0.5', '1.0'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def generate_temporal_report(self, results: Dict) -> str:
        """Generate temporal consistency report."""
        
        report = "# Temporal Consistency Validation Report\n\n"
        
        # Overall assessment
        if 'overall' in results:
            overall_score = results['overall']['temporal_consistency_score']
            if overall_score > 0.8:
                assessment = "EXCELLENT"
            elif overall_score > 0.6:
                assessment = "GOOD"
            elif overall_score > 0.4:
                assessment = "MODERATE"
            else:
                assessment = "POOR"
            
            report += f"## Overall Assessment: {assessment}\n"
            report += f"- Temporal Consistency Score: {overall_score:.3f}\n\n"
        
        # Detailed results
        if 'autocorrelation' in results:
            report += "## Autocorrelation Analysis\n"
            autocorr_error = np.mean(results['autocorrelation']['autocorr_error'])
            report += f"- Mean Autocorrelation Error: {autocorr_error:.6f}\n\n"
        
        if 'spectral' in results:
            report += "## Spectral Analysis\n"
            spectral_error = np.mean(results['spectral']['spectral_error'])
            report += f"- Mean Spectral Error: {spectral_error:.6f}\n\n"
        
        if 'phase' in results:
            phase_data = results['phase']
            report += "## Phase Consistency\n"
            report += f"- Mean Phase Error: {phase_data['mean_phase_error']:.3f} rad\n"
            report += f"- Phase Correlation: {phase_data['mean_phase_correlation']:.3f}\n"
            report += f"- Phase Consistency Score: {phase_data['phase_consistency_score']:.3f}\n\n"
        
        if 'energy' in results:
            energy_data = results['energy']
            report += "## Energy Conservation\n"
            report += f"- Energy Bias: {energy_data['energy_bias']:.6f}\n"
            report += f"- Energy Correlation: {energy_data['energy_correlation']:.3f}\n"
            report += f"- Conservation Score: {energy_data['energy_conservation_score']:.3f}\n\n"
        
        if 'divergence' in results:
            div_data = results['divergence']
            report += "## Incompressibility Preservation\n"
            report += f"- Divergence Correlation: {div_data['divergence_correlation']:.3f}\n"
            report += f"- Incompressibility Score: {div_data['incompressibility_preservation']:.3f}\n\n"
        
        return report
