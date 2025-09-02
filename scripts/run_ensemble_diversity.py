#!/usr/bin/env python3
"""
Ensemble Diversity Analysis Script
Analyze diversity and performance of model ensembles.
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.ensemble_diversity import EnsembleDiversityMetrics

def load_ensemble_predictions(results_dirs: List[Path], split: str = 'test') -> tuple:
    """Load predictions from multiple models to form an ensemble."""
    
    ensemble_predictions = []
    targets = None
    
    for results_dir in results_dirs:
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
        
        if predictions is not None:
            ensemble_predictions.append(predictions)
        
        # Load ground truth (only need once)
        if targets is None:
            gt_path = results_dir / f'gt_{split}.npy'
            if gt_path.exists():
                targets = np.load(gt_path)
                print(f"Loaded targets from: {gt_path}")
    
    if ensemble_predictions:
        # Stack predictions to create ensemble
        ensemble_array = np.stack(ensemble_predictions, axis=0)
        print(f"Created ensemble with {len(ensemble_predictions)} models")
        print(f"Ensemble shape: {ensemble_array.shape}")
        return ensemble_array, targets
    else:
        return None, None

def load_mc_ensemble(results_dir: Path, split: str = 'test', n_samples: int = 10) -> tuple:
    """Load MC dropout samples as ensemble members."""
    
    # Look for MC samples
    mc_patterns = [
        f'mc_samples_{split}.npy',
        f'samples_{split}.npy'
    ]
    
    mc_samples = None
    for pattern in mc_patterns:
        mc_path = results_dir / pattern
        if mc_path.exists():
            mc_samples = np.load(mc_path)
            print(f"Loaded MC samples from: {mc_path}")
            break
    
    # If no MC samples, try to simulate from mean and variance
    if mc_samples is None:
        mean_path = results_dir / f'mc_mean_{split}.npy'
        var_path = results_dir / f'mc_var_{split}.npy'
        
        if mean_path.exists() and var_path.exists():
            mean = np.load(mean_path)
            var = np.load(var_path)
            std = np.sqrt(var)
            
            # Generate samples from Gaussian distribution
            mc_samples = []
            for i in range(n_samples):
                sample = mean + std * np.random.randn(*mean.shape)
                mc_samples.append(sample)
            
            mc_samples = np.stack(mc_samples, axis=0)
            print(f"Generated {n_samples} MC samples from mean and variance")
    
    # Load targets
    targets = None
    gt_path = results_dir / f'gt_{split}.npy'
    if gt_path.exists():
        targets = np.load(gt_path)
        print(f"Loaded targets from: {gt_path}")
    
    return mc_samples, targets

def main():
    parser = argparse.ArgumentParser(description='Ensemble diversity analysis for turbulence models')
    parser.add_argument('--results_dirs', nargs='+', help='Paths to results directories for ensemble')
    parser.add_argument('--mc_results_dir', help='Path to MC dropout results directory')
    parser.add_argument('--split', default='test', help='Dataset split to analyze')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to analyze')
    parser.add_argument('--n_mc_samples', type=int, default=10, help='Number of MC samples to use')
    parser.add_argument('--output_dir', default='ensemble_diversity_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running ensemble diversity analysis")
    print(f"Output directory: {output_dir}")
    
    # Load ensemble data
    ensemble_predictions = None
    targets = None
    
    if args.results_dirs:
        # Load from multiple model directories
        results_dirs = [Path(d) for d in args.results_dirs]
        ensemble_predictions, targets = load_ensemble_predictions(results_dirs, args.split)
        analysis_type = "multi_model"
        output_subdir = output_dir / "multi_model_ensemble"
        
    elif args.mc_results_dir:
        # Load MC dropout samples
        results_dir = Path(args.mc_results_dir)
        ensemble_predictions, targets = load_mc_ensemble(results_dir, args.split, args.n_mc_samples)
        analysis_type = "mc_dropout"
        output_subdir = output_dir / results_dir.name
    
    else:
        print("Must specify either --results_dirs or --mc_results_dir")
        return
    
    if ensemble_predictions is None:
        print("No ensemble predictions found!")
        return
    
    output_subdir.mkdir(parents=True, exist_ok=True)
    
    print(f"Ensemble predictions shape: {ensemble_predictions.shape}")
    if targets is not None:
        print(f"Targets shape: {targets.shape}")
    
    # Extract sample for analysis
    if args.sample_idx < ensemble_predictions.shape[1]:
        sample_predictions = ensemble_predictions[:, args.sample_idx]
        sample_targets = targets[args.sample_idx] if targets is not None else None
        
        print(f"Analyzing sample {args.sample_idx}")
        print(f"Sample predictions shape: {sample_predictions.shape}")
    else:
        print(f"Sample index {args.sample_idx} out of range, using full dataset")
        sample_predictions = ensemble_predictions
        sample_targets = targets
    
    # Initialize diversity analyzer
    analyzer = EnsembleDiversityMetrics()
    
    # Run comprehensive ensemble analysis
    print(f"\n{'='*60}")
    print("ENSEMBLE DIVERSITY ANALYSIS")
    print(f"{'='*60}")
    
    results = analyzer.comprehensive_ensemble_analysis(sample_predictions, sample_targets)
    
    # Print key results
    if 'overall' in results:
        overall = results['overall']
        print(f"\nOverall Assessment:")
        print(f"  Ensemble Size: {overall['n_models']} models")
        print(f"  Quality Score: {overall['ensemble_quality_score']:.3f}")
        print(f"  Recommendation: {overall['recommendation']}")
    
    if 'diversity_measures' in results:
        diversity = results['diversity_measures']
        print(f"\nDiversity Measures:")
        print(f"  Diversity Score: {diversity['diversity_score']:.3f}")
        print(f"  Mean Correlation: {diversity['mean_correlation']:.3f}")
        print(f"  Mean Pairwise Disagreement: {diversity['mean_pairwise_disagreement']:.4f}")
        print(f"  Mean Q-Statistic: {diversity['mean_q_statistic']:.3f}")
    
    if 'ensemble_strength' in results:
        strength = results['ensemble_strength']
        print(f"\nEnsemble Performance:")
        print(f"  Mean Individual Error: {strength['mean_individual_error']:.4f}")
        print(f"  Ensemble Error: {strength['ensemble_error']:.4f}")
        print(f"  Ensemble Improvement: {strength['ensemble_improvement']:.1%}")
        print(f"  Best Individual Error: {strength['best_individual_error']:.4f}")
    
    if 'bias_variance' in results:
        bv = results['bias_variance']
        print(f"\nBias-Variance Decomposition:")
        print(f"  Bias²: {bv['bias_squared']:.4f}")
        print(f"  Variance: {bv['variance']:.4f}")
        print(f"  Total Error: {bv['total_error']:.4f}")
        print(f"  Bias/Variance Ratio: {bv['bias_variance_ratio']:.3f}")
    
    # Save results
    results_path = output_subdir / 'ensemble_diversity_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, np.ndarray):
                    json_results[key][subkey] = subvalue.tolist()
                else:
                    json_results[key][subkey] = subvalue
        else:
            json_results[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = output_subdir / 'ensemble_diversity_plots.png'
    analyzer.plot_ensemble_analysis(results, str(plot_path))
    print(f"Plots saved to: {plot_path}")
    
    # Generate report
    report = analyzer.generate_ensemble_report(results)
    report_path = output_subdir / 'ensemble_diversity_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    print(f"\nEnsemble diversity analysis completed!")
    print(f"Results saved in: {output_subdir}")

if __name__ == '__main__':
    main()
