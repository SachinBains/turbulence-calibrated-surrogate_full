#!/usr/bin/env python3
"""
Comprehensive Error Analysis Script
Analyze errors and detect failure modes in turbulence model predictions.
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.error_analysis import ComprehensiveErrorAnalyzer

def load_prediction_data(results_dir: Path, split: str = 'test') -> tuple:
    """Load predictions, targets, and uncertainties."""
    
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
    
    return predictions, targets, uncertainties

def prepare_analysis_data(predictions: np.ndarray, 
                         targets: np.ndarray,
                         uncertainties: Optional[np.ndarray],
                         sample_idx: int = 0) -> tuple:
    """Prepare data for error analysis."""
    
    # Handle different data shapes
    if predictions.ndim > 1:
        # Use specific sample or flatten all
        if sample_idx < predictions.shape[0]:
            pred_sample = predictions[sample_idx]
            target_sample = targets[sample_idx]
            unc_sample = uncertainties[sample_idx] if uncertainties is not None else None
        else:
            # Flatten all samples
            pred_sample = predictions.flatten()
            target_sample = targets.flatten()
            unc_sample = uncertainties.flatten() if uncertainties is not None else None
    else:
        pred_sample = predictions
        target_sample = targets
        unc_sample = uncertainties
    
    # Remove invalid values
    valid_mask = np.isfinite(pred_sample) & np.isfinite(target_sample)
    if unc_sample is not None:
        valid_mask &= np.isfinite(unc_sample) & (unc_sample > 0)
    
    pred_clean = pred_sample[valid_mask]
    target_clean = target_sample[valid_mask]
    unc_clean = unc_sample[valid_mask] if unc_sample is not None else None
    
    return pred_clean, target_clean, unc_clean

def main():
    parser = argparse.ArgumentParser(description='Comprehensive error analysis for turbulence models')
    parser.add_argument('--results_dir', required=True, help='Path to results directory')
    parser.add_argument('--split', default='test', help='Dataset split to analyze')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to analyze')
    parser.add_argument('--output_dir', default='error_analysis', help='Output directory')
    parser.add_argument('--n_clusters', type=int, default=5, help='Number of failure mode clusters')
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running comprehensive error analysis")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    predictions, targets, uncertainties = load_prediction_data(results_dir, args.split)
    
    if predictions is None or targets is None:
        print("Missing required prediction or target data!")
        return
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    if uncertainties is not None:
        print(f"Uncertainties shape: {uncertainties.shape}")
    
    # Prepare data for analysis
    pred_clean, target_clean, unc_clean = prepare_analysis_data(
        predictions, targets, uncertainties, args.sample_idx
    )
    
    print(f"Prepared data shapes:")
    print(f"  Predictions: {pred_clean.shape}")
    print(f"  Targets: {target_clean.shape}")
    if unc_clean is not None:
        print(f"  Uncertainties: {unc_clean.shape}")
    
    # Initialize error analyzer
    analyzer = ComprehensiveErrorAnalyzer()
    
    # Run comprehensive error analysis
    print(f"\n{'='*60}")
    print("COMPREHENSIVE ERROR ANALYSIS")
    print(f"{'='*60}")
    
    results = analyzer.comprehensive_error_analysis(pred_clean, target_clean, unc_clean)
    
    # Print key results
    if 'overall' in results:
        overall = results['overall']
        print(f"\nOverall Assessment: {overall['assessment']}")
        print(f"  RMSE: {overall['rmse']:.4f}")
        print(f"  Outlier Percentage: {overall['outlier_percentage']:.1f}%")
        print(f"  Dominant Failure Mode: {overall['dominant_failure_mode']}")
    
    if 'error_statistics' in results:
        stats = results['error_statistics']
        print(f"\nError Statistics:")
        print(f"  MAE: {stats['mae']:.4f}")
        print(f"  RMSE: {stats['rmse']:.4f}")
        print(f"  Max Error: {stats['max_error']:.4f}")
        print(f"  95th Percentile: {stats['q95_error']:.4f}")
        print(f"  Error Std: {stats['error_std']:.4f}")
        print(f"  Skewness: {stats['error_skewness']:.3f}")
        print(f"  Kurtosis: {stats['error_kurtosis']:.3f}")
    
    if 'outliers' in results:
        print(f"\nOutlier Detection:")
        for method, outlier_data in results['outliers'].items():
            percentage = outlier_data['outlier_percentage']
            count = outlier_data['outlier_count']
            print(f"  {method.upper()}: {percentage:.1f}% ({count} samples)")
    
    if 'failure_modes' in results and 'cluster_analysis' in results['failure_modes']:
        print(f"\nFailure Mode Clusters:")
        cluster_data = results['failure_modes']['cluster_analysis']
        
        for cluster_name, data in cluster_data.items():
            cluster_id = cluster_name.split('_')[1]
            print(f"  Cluster {cluster_id}: {data['percentage']:.1f}% samples, "
                  f"Mean Error: {data['mean_error']:.4f}")
    
    # Save results
    results_path = output_dir / 'error_analysis_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    json_results[key][subkey] = subvalue.tolist()
                elif isinstance(subvalue, dict):
                    json_results[key][subkey] = {}
                    for subsubkey, subsubvalue in subvalue.items():
                        if isinstance(subsubvalue, np.ndarray):
                            json_results[key][subkey][subsubkey] = subsubvalue.tolist()
                        else:
                            json_results[key][subkey][subsubkey] = subsubvalue
                else:
                    json_results[key][subkey] = subvalue
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = output_dir / 'error_analysis_plots.png'
    analyzer.plot_error_analysis(results, str(plot_path))
    print(f"Plots saved to: {plot_path}")
    
    # Generate report
    report = analyzer.generate_error_report(results)
    report_path = output_dir / 'error_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    print(f"\nComprehensive error analysis completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
