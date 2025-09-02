#!/usr/bin/env python3
"""
Cross-Validation Script for Turbulence Models
Robust evaluation using k-fold cross-validation with turbulence-specific stratification.
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.eval.cross_validation import TurbulenceCrossValidator, turbulence_metrics

def main():
    parser = argparse.ArgumentParser(description='Cross-validation for turbulence models')
    parser.add_argument('--config', required=True, help='Path to experiment config')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of CV folds')
    parser.add_argument('--stratify', action='store_true', help='Use stratified CV')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--output_dir', default='cv_results', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    device = pick_device()
    
    print(f"Running cross-validation for experiment: {exp_id}")
    print(f"Device: {device}")
    print(f"Folds: {args.n_folds}")
    print(f"Stratified: {args.stratify}")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset (use train split for CV)
    dataset = HITDataset(cfg, 'train', eval_mode=False)
    print(f"Dataset size: {len(dataset)}")
    
    # Build model
    mcfg = cfg['model']
    model = UNet3D(
        mcfg['in_channels'], 
        mcfg['out_channels'], 
        base_ch=mcfg['base_channels']
    )
    
    # Initialize cross-validator
    cv = TurbulenceCrossValidator(
        n_folds=args.n_folds,
        stratify=args.stratify,
        random_state=42
    )
    
    # Run cross-validation
    cv_results = cv.run_cross_validation(
        model=model,
        dataset=dataset,
        device=device,
        metrics_fn=turbulence_metrics,
        batch_size=args.batch_size
    )
    
    # Print results
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION RESULTS")
    print(f"{'='*60}")
    
    key_metrics = ['val_loss', 'mse', 'correlation', 'energy_error']
    
    for metric in key_metrics:
        mean_key = f'{metric}_mean'
        std_key = f'{metric}_std'
        
        if mean_key in cv_results:
            mean_val = cv_results[mean_key]
            std_val = cv_results[std_key]
            print(f"{metric.upper()}: {mean_val:.6f} ± {std_val:.6f}")
    
    print(f"\nCV Score (lower is better): {cv_results['cv_score']:.6f}")
    print(f"CV Stability (lower is better): {cv_results['cv_stability']:.6f}")
    
    # Save results
    results_path = output_dir / 'cv_results.json'
    cv.save_results(cv_results, cv.fold_results, str(results_path))
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = output_dir / 'cv_plots.png'
    cv.plot_cv_results(cv_results, str(plot_path))
    print(f"Plots saved to: {plot_path}")

if __name__ == '__main__':
    main()
