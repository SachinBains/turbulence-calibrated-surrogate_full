#!/usr/bin/env python3
"""
Step 13: Enhanced Interpretability and Uncertainty Explainability Analysis
Comprehensive analysis using SHAP, ALE, GradCAM, Sobol indices, and advanced
uncertainty explainability methods for turbulence prediction models.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal, ndimage
from typing import Dict, List, Tuple, Any
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.interp.shap_analysis import TurbulenceSHAP
from src.interp.ale import TurbulenceALE
from src.interp.gradcam import VelocityFieldGradCAM
from src.interp.sobol import TurbulenceSobolAnalyzer
from src.interp.grad import integrated_gradients, gradient_shap

def load_model_and_data(config_path: str, device: torch.device) -> Tuple[torch.nn.Module, DataLoader]:
    """Load trained model and dataset."""
    cfg = load_config(config_path)
    exp_id = cfg['experiment_id']
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    # Find best checkpoint
    ckpts = sorted(results_dir.glob('best_*.pth'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {results_dir}")
    
    # Build model
    mcfg = cfg['model']
    model = UNet3D(
        mcfg['in_channels'], 
        mcfg['out_channels'], 
        base_ch=mcfg['base_channels']
    )
    
    # Load weights
    state = torch.load(ckpts[-1], map_location=device)
    model.load_state_dict(state['model'])
    model = model.to(device)
    model.eval()
    
    # Load dataset
    dataset = HITDataset(cfg, 'test', eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return model, loader

def comprehensive_shap_analysis(model: torch.nn.Module, loader: DataLoader, 
                               device: torch.device, output_dir: Path) -> Dict:
    """Perform comprehensive SHAP analysis for predictions and uncertainty."""
    print("Running comprehensive SHAP analysis...")
    
    shap_analyzer = TurbulenceSHAP(model, device)
    
    # Collect data
    test_samples = []
    background_samples = []
    
    for i, (x, _) in enumerate(loader):
        if i < 20:  # Background samples
            background_samples.append(x[0])
        if 20 <= i < 30:  # Test samples
            test_samples.append(x[0])
        if i >= 30:
            break
    
    background_data = torch.stack(background_samples)
    test_data = torch.stack(test_samples)
    
    # Analyze prediction drivers
    pred_results = shap_analyzer.analyze_prediction_drivers(
        background_data, test_data, max_evals=500
    )
    
    # Analyze uncertainty drivers
    uncertainty_results = shap_analyzer.analyze_uncertainty_drivers(
        background_data, test_data[:3], max_evals=200
    )
    
    # Create visualizations
    figures_dir = output_dir / 'shap_analysis'
    figures_dir.mkdir(exist_ok=True)
    
    shap_analyzer.create_spatial_shap_maps(pred_results, str(figures_dir))
    shap_analyzer.create_summary_plots(pred_results, str(figures_dir))
    
    # Compute importance statistics
    pred_importance = shap_analyzer.compute_feature_importance(pred_results['shap_values'])
    uncertainty_importance = shap_analyzer.compute_feature_importance(uncertainty_results['uncertainty_shap'])
    
    return {
        'prediction_importance': pred_importance,
        'uncertainty_importance': uncertainty_importance,
        'shap_results': pred_results,
        'uncertainty_shap_results': uncertainty_results
    }

def comprehensive_ale_analysis(model: torch.nn.Module, loader: DataLoader,
                              device: torch.device, output_dir: Path) -> Dict:
    """Perform comprehensive ALE analysis."""
    print("Running comprehensive ALE analysis...")
    
    ale_analyzer = TurbulenceALE(model, device)
    
    # Collect samples
    samples = []
    for i, (x, _) in enumerate(loader):
        if i >= 50:  # Use 50 samples
            break
        samples.append(x[0])
    
    X = torch.stack(samples)
    
    # Analyze spatial effects
    spatial_results = ale_analyzer.analyze_spatial_effects(X, n_regions=8, n_bins=15)
    
    # Analyze velocity component effects
    velocity_results = ale_analyzer.analyze_velocity_component_effects(X, n_bins=20)
    
    # Create visualizations
    figures_dir = output_dir / 'ale_analysis'
    figures_dir.mkdir(exist_ok=True)
    
    ale_analyzer.plot_ale_results(spatial_results, str(figures_dir / 'spatial'))
    ale_analyzer.plot_ale_results(velocity_results, str(figures_dir / 'velocity'))
    
    # Compute importance rankings
    spatial_ranking = ale_analyzer.compute_ale_importance_ranking(spatial_results)
    velocity_ranking = ale_analyzer.compute_ale_importance_ranking(velocity_results)
    
    return {
        'spatial_effects': spatial_results,
        'velocity_effects': velocity_results,
        'spatial_ranking': spatial_ranking,
        'velocity_ranking': velocity_ranking
    }

def comprehensive_gradcam_analysis(model: torch.nn.Module, loader: DataLoader,
                                  device: torch.device, output_dir: Path) -> Dict:
    """Perform comprehensive GradCAM analysis."""
    print("Running comprehensive GradCAM analysis...")
    
    gradcam_analyzer = VelocityFieldGradCAM(model)
    
    # Get sample input
    sample_input, _ = next(iter(loader))
    sample_input = sample_input.to(device)
    
    # Analyze velocity importance
    cam_results = gradcam_analyzer.analyze_velocity_importance(sample_input)
    
    # Create visualizations
    figures_dir = output_dir / 'gradcam_analysis'
    figures_dir.mkdir(exist_ok=True)
    
    gradcam_analyzer.visualize_cams(sample_input, cam_results, str(figures_dir))
    
    # Get importance statistics
    importance_stats = gradcam_analyzer.get_importance_statistics(cam_results)
    
    # Cleanup
    gradcam_analyzer.cleanup()
    
    return {
        'cam_results': cam_results,
        'importance_statistics': importance_stats
    }

def comprehensive_sobol_analysis(model: torch.nn.Module, loader: DataLoader,
                                device: torch.device, output_dir: Path) -> Dict:
    """Perform comprehensive Sobol sensitivity analysis."""
    print("Running comprehensive Sobol sensitivity analysis...")
    
    sobol_analyzer = TurbulenceSobolAnalyzer(model, device)
    
    # Get sample input
    sample_input, _ = next(iter(loader))
    
    # Analyze velocity sensitivity
    velocity_sensitivity = sobol_analyzer.analyze_velocity_sensitivity(
        sample_input, n_samples=256, perturbation_scale=0.1
    )
    
    # Create visualizations
    figures_dir = output_dir / 'sobol_analysis'
    figures_dir.mkdir(exist_ok=True)
    
    from src.interp.sobol import plot_sobol_results
    plot_sobol_results(velocity_sensitivity, "Velocity Field Sensitivity", 
                      str(figures_dir / 'velocity_sensitivity.png'))
    
    return {
        'velocity_sensitivity': velocity_sensitivity
    }

def enhanced_uncertainty_analysis(predictions: Dict, output_dir: Path) -> Dict:
    """Enhanced uncertainty analysis with correlation and calibration metrics."""
    print("Running enhanced uncertainty analysis...")
    
    results = {}
    
    for method_name, pred_data in predictions.items():
        if 'uncertainty' not in pred_data:
            continue
            
        uncertainty = pred_data['uncertainty']
        errors = pred_data.get('errors', None)
        
        if errors is None:
            continue
            
        method_results = {}
        
        # Uncertainty-error correlation
        flat_uncertainty = uncertainty.flatten()
        flat_errors = errors.flatten()
        
        correlation = np.corrcoef(flat_uncertainty, flat_errors)[0, 1]
        method_results['uncertainty_error_correlation'] = float(correlation)
        
        # Calibration analysis
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_error = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this confidence interval
            in_bin = (flat_uncertainty >= np.percentile(flat_uncertainty, bin_lower * 100)) & \
                    (flat_uncertainty < np.percentile(flat_uncertainty, bin_upper * 100))
            
            if np.sum(in_bin) > 0:
                bin_accuracy = np.mean(flat_errors[in_bin] < np.percentile(flat_errors, 80))
                bin_confidence = (bin_lower + bin_upper) / 2
                calibration_error += np.abs(bin_accuracy - bin_confidence) * np.sum(in_bin)
        
        calibration_error /= len(flat_uncertainty)
        method_results['calibration_error'] = float(calibration_error)
        
        # Sharpness (average uncertainty)
        method_results['mean_uncertainty'] = float(np.mean(flat_uncertainty))
        method_results['uncertainty_std'] = float(np.std(flat_uncertainty))
        
        # Coverage analysis
        sorted_indices = np.argsort(flat_uncertainty)
        sorted_errors = flat_errors[sorted_indices]
        
        # Compute coverage at different uncertainty levels
        coverage_levels = [0.5, 0.8, 0.9, 0.95]
        for level in coverage_levels:
            threshold_idx = int(level * len(sorted_errors))
            if threshold_idx < len(sorted_errors):
                coverage = np.mean(sorted_errors[:threshold_idx] < np.percentile(flat_errors, level * 100))
                method_results[f'coverage_{int(level*100)}'] = float(coverage)
        
        results[method_name] = method_results
    
    return results

def analyze_prediction_patterns(prediction: np.ndarray, method_name: str) -> Dict[str, Any]:
    """Enhanced spatial pattern analysis with turbulence-specific metrics."""
    
    # Remove batch dimension if present
    if prediction.ndim == 4 and prediction.shape[0] == 1:
        pred = prediction[0]  # Shape: (32, 32, 32)
    else:
        pred = prediction
    
    results = {}
    
    # Basic statistics
    results['mean_value'] = float(np.mean(pred))
    results['std_value'] = float(np.std(pred))
    results['min_value'] = float(np.min(pred))
    results['max_value'] = float(np.max(pred))
    results['skewness'] = float(signal.skew(pred.flatten()))
    results['kurtosis'] = float(signal.kurtosis(pred.flatten()))
    
    # Spatial gradients (measure of local variation)
    grad_x = np.gradient(pred, axis=0)
    grad_y = np.gradient(pred, axis=1)
    grad_z = np.gradient(pred, axis=2)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    
    results['gradient_mean'] = float(np.mean(gradient_magnitude))
    results['gradient_std'] = float(np.std(gradient_magnitude))
    results['gradient_max'] = float(np.max(gradient_magnitude))
    
    # Turbulence-specific metrics
    # Approximate enstrophy (vorticity magnitude squared)
    if pred.ndim == 3:
        # Simple finite difference approximation
        dw_dy = np.gradient(pred, axis=1)
        dv_dz = np.gradient(pred, axis=2)
        omega_x = dw_dy - dv_dz  # Simplified vorticity component
        enstrophy = omega_x**2
        results['enstrophy_mean'] = float(np.mean(enstrophy))
        results['enstrophy_std'] = float(np.std(enstrophy))
    
    # Spatial autocorrelation (measure of spatial structure)
    center_slice = pred[pred.shape[0]//2]
    autocorr = signal.correlate(center_slice, center_slice, mode='same')
    autocorr_normalized = autocorr / autocorr.max()
    
    results['spatial_autocorr_peak'] = float(autocorr_normalized.max())
    results['spatial_autocorr_width'] = float(np.sum(autocorr_normalized > 0.5))
    
    # Energy distribution across scales
    fft_pred = np.fft.fftn(pred)
    power_spectrum = np.abs(fft_pred)**2
    
    # Radial average of power spectrum
    center = np.array(pred.shape) // 2
    y, x, z = np.ogrid[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
    
    # Bin by radius
    r_bins = np.arange(0, min(center) + 1)
    power_radial = []
    for i in range(len(r_bins)-1):
        mask = (r >= r_bins[i]) & (r < r_bins[i+1])
        if np.any(mask):
            power_radial.append(np.mean(power_spectrum[mask]))
        else:
            power_radial.append(0)
    
    results['power_spectrum'] = power_radial
    results['spectral_peak_freq'] = float(np.argmax(power_radial[1:]) + 1)  # Skip DC component
    results['spectral_energy_ratio'] = float(np.sum(power_radial[1:5]) / np.sum(power_radial[1:]))
    
    # Feature localization analysis
    # Find regions of high activity
    pred_abs = np.abs(pred)
    threshold_95 = np.percentile(pred_abs, 95)
    high_activity_mask = pred_abs > threshold_95
    
    results['high_activity_fraction'] = float(np.mean(high_activity_mask))
    results['high_activity_clusters'] = int(ndimage.label(high_activity_mask)[1])
    
    # Spatial moments (center of mass, spread)
    total_mass = np.sum(pred_abs)
    if total_mass > 0:
        coords = np.mgrid[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
        center_of_mass = [float(np.sum(coords[i] * pred_abs) / total_mass) for i in range(3)]
        results['center_of_mass'] = center_of_mass
        
        # Spread around center of mass
        spread = np.sum(pred_abs * np.sum([(coords[i] - center_of_mass[i])**2 for i in range(3)], axis=0)) / total_mass
        results['spatial_spread'] = float(spread)
    else:
        results['center_of_mass'] = [16.0, 16.0, 16.0]  # Center of 32x32x32 grid
        results['spatial_spread'] = 0.0
    
    return results

def create_prediction_pattern_analysis(
    interp_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    sample_idx: int
):
    """Create plots analyzing prediction patterns"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    methods = list(interp_results.keys())
    
    # Plot 1: Gradient magnitude comparison
    ax = axes[0, 0]
    gradient_means = [interp_results[m]['gradient_mean'] for m in methods]
    gradient_stds = [interp_results[m]['gradient_std'] for m in methods]
    
    x_pos = np.arange(len(methods))
    ax.bar(x_pos, gradient_means, yerr=gradient_stds, capsize=5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.set_ylabel('Gradient Magnitude')
    ax.set_title('Spatial Gradient Analysis')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Value distribution comparison
    ax = axes[0, 1]
    means = [interp_results[m]['mean_value'] for m in methods]
    stds = [interp_results[m]['std_value'] for m in methods]
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, color='orange')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, rotation=45)
    ax.set_ylabel('Prediction Value')
    ax.set_title('Prediction Statistics')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Spatial autocorrelation
    ax = axes[1, 0]
    autocorr_peaks = [interp_results[m]['spatial_autocorr_peak'] for m in methods]
    autocorr_widths = [interp_results[m]['spatial_autocorr_width'] for m in methods]
    
    ax.scatter(autocorr_peaks, autocorr_widths, s=100, alpha=0.7)
    for i, method in enumerate(methods):
        ax.annotate(method, (autocorr_peaks[i], autocorr_widths[i]), 
                   xytext=(5, 5), textcoords='offset points')
    ax.set_xlabel('Autocorrelation Peak')
    ax.set_ylabel('Autocorrelation Width')
    ax.set_title('Spatial Structure Analysis')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Power spectrum comparison
    ax = axes[1, 1]
    for method in methods:
        power_spec = interp_results[method]['power_spectrum']
        freqs = np.arange(len(power_spec))
        ax.loglog(freqs[1:], power_spec[1:], label=method, marker='o', markersize=4)
    
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    ax.set_title('Power Spectrum Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'prediction_pattern_analysis_sample{sample_idx}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_spatial_analysis(
    interp_results: Dict[str, Dict[str, Any]],
    output_dir: Path,
    sample_idx: int
):
    """Create spatial analysis plots"""
    
    n_methods = len(interp_results)
    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 5*n_methods))
    if n_methods == 1:
        axes = axes.reshape(1, -1)
    
    for i, (method, results) in enumerate(interp_results.items()):
        # Plot power spectrum
        ax = axes[i, 0]
        power_spec = results['power_spectrum']
        freqs = np.arange(len(power_spec))
        ax.loglog(freqs[1:], power_spec[1:], 'o-')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power')
        ax.set_title(f'{method}\nPower Spectrum')
        ax.grid(True, alpha=0.3)
        
        # Plot spatial characteristics summary
        ax = axes[i, 1]
        metrics = ['gradient_mean', 'spatial_autocorr_peak', 'spectral_energy_ratio', 'high_activity_fraction']
        values = [results[m] for m in metrics]
        
        # Normalize values for comparison
        if max(values) > min(values):
            normalized_values = [(v - min(values)) / (max(values) - min(values)) for v in values]
        else:
            normalized_values = [0.5] * len(values)
        
        bars = ax.bar(metrics, normalized_values, alpha=0.7)
        ax.set_ylabel('Normalized Value')
        ax.set_title(f'{method}\nSpatial Characteristics')
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / f'spatial_analysis_sample{sample_idx}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()

def create_interpretability_summary(
    interp_results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> pd.DataFrame:
    """Create summary table of interpretability metrics"""
    
    summary_data = []
    
    for method, results in interp_results.items():
        summary_data.append({
            'Method': method,
            'Mean_Gradient': f"{results['gradient_mean']:.4f}",
            'Spatial_Autocorr': f"{results['spatial_autocorr_peak']:.3f}",
            'Spectral_Peak_Freq': f"{results['spectral_peak_freq']:.1f}",
            'Energy_Ratio': f"{results['spectral_energy_ratio']:.3f}",
            'High_Activity_Fraction': f"{results['high_activity_fraction']:.3f}",
            'Activity_Clusters': f"{results['high_activity_clusters']:.0f}",
            'Spatial_Spread': f"{results['spatial_spread']:.2f}",
            'Prediction_Range': f"{results['max_value'] - results['min_value']:.4f}",
            'Prediction_Std': f"{results['std_value']:.4f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'interpretability_summary.csv', index=False)
    
    # Save detailed results
    with open(output_dir / 'detailed_interpretability_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for method, results in interp_results.items():
            serializable_results[method] = {}
            for key, value in results.items():
                if isinstance(value, np.ndarray):
                    serializable_results[method][key] = value.tolist()
                elif isinstance(value, (np.floating, np.integer)):
                    serializable_results[method][key] = float(value)
                elif isinstance(value, np.bool_):
                    serializable_results[method][key] = bool(value)
                else:
                    serializable_results[method][key] = value
        
        json.dump(serializable_results, f, indent=2)
    
    return summary_df

def main():
    """Main function for Step 13: Interpretability Analysis"""
    
    print("=== Step 13: Interpretability/Feature Analysis ===\n")
    
    # Setup paths
    base_dir = Path.cwd()
    step10_dir = Path(r'C:\Users\Sachi\OneDrive\Desktop\Dissertation\step10_visualization')
    output_dir = base_dir / 'step13_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Check available prediction files
    print("1. Checking available prediction files...")
    available_methods = {}
    
    for results_subdir in step10_dir.glob('results/*/'):
        method_name = results_subdir.name
        mean_file = results_subdir / 'mc_mean_test.npy' if 'mc' in method_name.lower() else results_subdir / 'ens_mean_test.npy'
        
        if mean_file.exists():
            available_methods[method_name] = mean_file
            print(f"  Found: {method_name} -> {mean_file}")
    
    if not available_methods:
        print("ERROR: No prediction files found in step10_visualization/results/")
        return
    
    # For interpretability analysis, we'll focus on prediction patterns
    # since we don't have trained models locally
    print("\n2. Analyzing prediction patterns and spatial characteristics...")
    
    interp_results = {}
    sample_idx = 0
    
    # For interpretability analysis, we'll focus on prediction patterns
    # since we don't have trained models locally
    print("  Focusing on prediction pattern analysis")
    
    # Analyze prediction characteristics for each method
    for method_name, pred_file in available_methods.items():
        print(f"\nAnalyzing {method_name}...")
        
        # Load predictions
        predictions = np.load(pred_file)  # Shape: (N, 1, 32, 32, 32)
        
        if predictions.shape[0] <= sample_idx:
            print(f"  Warning: Sample {sample_idx} not available, using sample 0")
            sample_idx = 0
        
        pred_sample = predictions[sample_idx]  # Shape: (1, 32, 32, 32)
        
        # Analyze spatial patterns in predictions
        results = analyze_prediction_patterns(pred_sample, method_name)
        interp_results[method_name] = results
    
    # Create visualizations
    print("\n3. Creating interpretability visualizations...")
    create_prediction_pattern_analysis(interp_results, output_dir, sample_idx)
    
    # Create spatial analysis
    print("4. Creating spatial importance analysis...")
    create_spatial_analysis(interp_results, output_dir, sample_idx)
    
    # Create summary statistics
    print("5. Creating interpretability summary...")
    summary_df = create_interpretability_summary(interp_results, output_dir)
    
    # Print summary
    print("\n=== INTERPRETABILITY ANALYSIS SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    print(f"\nStep 13 Complete: Interpretability analysis saved to {output_dir}/")
    print("Generated files:")
    print("  - prediction_pattern_analysis_sample0.png")
    print("  - spatial_analysis_sample0.png")
    print("  - interpretability_summary.csv")
    print("  - detailed_interpretability_results.json")
    
    print("\nNext: Step 14 - Automated summary report generation")

if __name__ == "__main__":
    main()
