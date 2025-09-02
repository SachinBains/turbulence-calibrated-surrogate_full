#!/usr/bin/env python3
"""
Uncertainty Calibration Analysis Script
Diagnose and recalibrate uncertainty estimates for turbulence models.
"""

import os
import sys
import argparse
import numpy as np
import torch
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.uncertainty_calibration import UncertaintyCalibrationDiagnostics

def load_model_outputs(results_dir: Path, split: str = 'test') -> tuple:
    """Load model predictions, targets, and uncertainties."""
    
    # Load predictions
    pred_patterns = [
        f'pred_{split}.npy',
        f'mean_{split}.npy', 
        f'mc_mean_{split}.npy'
    ]
    
    predictions = None
    for pattern in pred_patterns:
        pred_path = results_dir / pattern
        if pred_path.exists():
            predictions = np.load(pred_path)
            print(f"Loaded predictions from: {pred_path}")
            break
    
    # Load ground truth
    gt_path = results_dir / f'gt_{split}.npy'
    targets = None
    if gt_path.exists():
        targets = np.load(gt_path)
        print(f"Loaded targets from: {gt_path}")
    
    # Load uncertainties
    uncertainty_patterns = [
        f'var_{split}.npy',
        f'mc_var_{split}.npy',
        f'std_{split}.npy',
        f'mc_std_{split}.npy'
    ]
    
    uncertainties = None
    for pattern in uncertainty_patterns:
        unc_path = results_dir / pattern
        if unc_path.exists():
            uncertainties = np.load(unc_path)
            print(f"Loaded uncertainties from: {unc_path}")
            # Convert variance to standard deviation if needed
            if 'var' in pattern:
                uncertainties = np.sqrt(uncertainties)
            break
    
    # Load logits if available (for temperature scaling)
    logits_path = results_dir / f'logits_{split}.npy'
    logits = None
    if logits_path.exists():
        logits_array = np.load(logits_path)
        logits = torch.tensor(logits_array, dtype=torch.float32)
        print(f"Loaded logits from: {logits_path}")
    
    return predictions, targets, uncertainties, logits

def prepare_data_for_calibration(predictions: np.ndarray, 
                                targets: np.ndarray,
                                uncertainties: np.ndarray,
                                sample_idx: int = 0) -> tuple:
    """Prepare data for calibration analysis."""
    
    # Handle different data shapes
    if predictions.ndim > 1:
        # Flatten spatial dimensions for analysis
        if predictions.ndim == 4:  # (N, C, H, W)
            pred_flat = predictions[sample_idx].flatten()
            target_flat = targets[sample_idx].flatten()
            unc_flat = uncertainties[sample_idx].flatten()
        elif predictions.ndim == 5:  # (N, C, D, H, W)
            pred_flat = predictions[sample_idx].flatten()
            target_flat = targets[sample_idx].flatten()
            unc_flat = uncertainties[sample_idx].flatten()
        else:
            pred_flat = predictions.flatten()
            target_flat = targets.flatten()
            unc_flat = uncertainties.flatten()
    else:
        pred_flat = predictions
        target_flat = targets
        unc_flat = uncertainties
    
    # Remove any invalid values
    valid_mask = np.isfinite(pred_flat) & np.isfinite(target_flat) & np.isfinite(unc_flat)
    valid_mask &= (unc_flat > 0)  # Positive uncertainties only
    
    pred_clean = pred_flat[valid_mask]
    target_clean = target_flat[valid_mask]
    unc_clean = unc_flat[valid_mask]
    
    # Normalize uncertainties to [0, 1] for calibration analysis
    unc_normalized = (unc_clean - np.min(unc_clean)) / (np.max(unc_clean) - np.min(unc_clean) + 1e-8)
    
    return pred_clean, target_clean, unc_normalized

def main():
    parser = argparse.ArgumentParser(description='Uncertainty calibration analysis for turbulence models')
    parser.add_argument('--results_dir', required=True, help='Path to results directory')
    parser.add_argument('--split', default='test', help='Dataset split to analyze')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to analyze')
    parser.add_argument('--output_dir', default='calibration_analysis', help='Output directory')
    parser.add_argument('--n_bins', type=int, default=10, help='Number of bins for reliability diagram')
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running uncertainty calibration analysis")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    predictions, targets, uncertainties, logits = load_model_outputs(results_dir, args.split)
    
    if predictions is None or targets is None or uncertainties is None:
        print("Missing required data files!")
        return
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    print(f"Uncertainties shape: {uncertainties.shape}")
    if logits is not None:
        print(f"Logits shape: {logits.shape}")
    
    # Prepare data for calibration analysis
    pred_clean, target_clean, unc_clean = prepare_data_for_calibration(
        predictions, targets, uncertainties, args.sample_idx
    )
    
    print(f"Prepared data shapes:")
    print(f"  Predictions: {pred_clean.shape}")
    print(f"  Targets: {target_clean.shape}")
    print(f"  Uncertainties: {unc_clean.shape}")
    
    # Initialize calibration diagnostics
    calibrator = UncertaintyCalibrationDiagnostics()
    
    # Run comprehensive calibration analysis
    print(f"\n{'='*60}")
    print("UNCERTAINTY CALIBRATION ANALYSIS")
    print(f"{'='*60}")
    
    results = calibrator.comprehensive_calibration_analysis(
        pred_clean, target_clean, unc_clean, logits
    )
    
    # Print key results
    if 'calibration_metrics' in results:
        metrics = results['calibration_metrics']
        print(f"\nCalibration Metrics:")
        print(f"  Expected Calibration Error (ECE): {metrics['ece']:.4f}")
        print(f"  Maximum Calibration Error (MCE): {metrics['mce']:.4f}")
        print(f"  Average Calibration Error (ACE): {metrics['ace']:.4f}")
        print(f"  Calibration Score: {metrics['calibration_score']:.3f}")
        print(f"  Brier Score: {metrics['brier_score']:.4f}")
        print(f"  Reliability: {metrics['reliability']:.4f}")
        print(f"  Resolution: {metrics['resolution']:.4f}")
        print(f"  Sharpness: {metrics['sharpness']:.4f}")
    
    if 'coverage_analysis' in results:
        coverage = results['coverage_analysis']
        print(f"\nCoverage Analysis:")
        print(f"  Mean Coverage Gap: {coverage.get('mean_coverage_gap', 0):.4f}")
        print(f"  Coverage Quality: {coverage.get('coverage_quality', 0):.3f}")
        
        # Show coverage at key levels
        for level in [68, 90, 95]:
            expected = coverage.get(f'expected_{level}', level/100)
            actual = coverage.get(f'coverage_{level}', 0)
            gap = coverage.get(f'coverage_gap_{level}', 0)
            print(f"  {level}% Coverage: Expected {expected:.2f}, Actual {actual:.2f}, Gap {gap:.4f}")
    
    if 'recalibration_evaluation' in results:
        recal_eval = results['recalibration_evaluation']
        print(f"\nRecalibration Results:")
        
        original_ece = recal_eval.get('original', {}).get('ece', 0)
        print(f"  Original ECE: {original_ece:.4f}")
        
        for method in ['platt', 'isotonic', 'temperature']:
            if method in recal_eval:
                method_ece = recal_eval[method]['ece']
                improvement_key = f'{method}_improvement'
                
                if improvement_key in recal_eval:
                    improvement = recal_eval[improvement_key]['ece_improvement']
                    status = "✓ Improved" if improvement > 0 else "✗ No improvement"
                    print(f"  {method.title()} Scaling: ECE {method_ece:.4f}, Improvement {improvement:.4f} {status}")
    
    # Save results
    results_path = output_dir / 'calibration_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    json_results[key][subkey] = subvalue.tolist()
                elif hasattr(subvalue, 'numpy'):  # torch tensor
                    json_results[key][subkey] = subvalue.numpy().tolist()
                else:
                    json_results[key][subkey] = subvalue
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = output_dir / 'calibration_plots.png'
    calibrator.plot_calibration_analysis(results, str(plot_path))
    print(f"Plots saved to: {plot_path}")
    
    # Generate report
    report = calibrator.generate_calibration_report(results)
    report_path = output_dir / 'calibration_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    print(f"\nUncertainty calibration analysis completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
