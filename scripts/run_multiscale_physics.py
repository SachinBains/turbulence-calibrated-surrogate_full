#!/usr/bin/env python3
"""
Multi-Scale Physics Validation Script
Validate physics across multiple scales for turbulence predictions.
"""

import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.eval.multiscale_physics import MultiScalePhysicsValidator

def load_velocity_data(results_dir: Path, split: str = 'test') -> tuple:
    """Load velocity predictions and ground truth."""
    
    # Try to load existing predictions
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
    ground_truth = None
    if gt_path.exists():
        ground_truth = np.load(gt_path)
        print(f"Loaded ground truth from: {gt_path}")
    
    return predictions, ground_truth

def prepare_velocity_field(data: np.ndarray, sample_idx: int = 0) -> np.ndarray:
    """Prepare velocity field for multi-scale analysis."""
    
    if data.ndim == 4:  # (N, C, H, W) - 2D case, extend to 3D
        sample = data[sample_idx]
        if sample.shape[0] == 1:  # Single component, create 3-component field
            # Create artificial 3-component field for demonstration
            u = sample[0]
            v = np.zeros_like(u)
            w = np.zeros_like(u)
            # Add to third dimension
            velocity_field = np.stack([u[:, :, np.newaxis], v[:, :, np.newaxis], w[:, :, np.newaxis]], axis=0)
        else:
            velocity_field = sample
            # Add third spatial dimension if needed
            if velocity_field.ndim == 3:  # (C, H, W)
                velocity_field = velocity_field[:, :, :, np.newaxis]
    
    elif data.ndim == 5:  # (N, C, D, H, W) - 3D case
        sample = data[sample_idx]
        if sample.shape[0] == 1:  # Single component
            # Create 3-component field
            u = sample[0]
            v = np.zeros_like(u)
            w = np.zeros_like(u)
            velocity_field = np.stack([u, v, w], axis=0)
        else:
            velocity_field = sample
    
    else:
        raise ValueError(f"Unsupported data shape: {data.shape}")
    
    # Ensure we have 3 velocity components
    if velocity_field.shape[0] != 3:
        # Pad with zeros if needed
        if velocity_field.shape[0] == 1:
            u = velocity_field[0]
            v = np.zeros_like(u)
            w = np.zeros_like(u)
            velocity_field = np.stack([u, v, w], axis=0)
        else:
            raise ValueError(f"Expected 1 or 3 velocity components, got {velocity_field.shape[0]}")
    
    return velocity_field

def main():
    parser = argparse.ArgumentParser(description='Multi-scale physics validation for turbulence models')
    parser.add_argument('--results_dir', required=True, help='Path to results directory')
    parser.add_argument('--split', default='test', help='Dataset split to analyze')
    parser.add_argument('--sample_idx', type=int, default=0, help='Sample index to analyze')
    parser.add_argument('--viscosity', type=float, default=1e-4, help='Kinematic viscosity')
    parser.add_argument('--output_dir', default='multiscale_analysis', help='Output directory')
    parser.add_argument('--compare_with_reference', action='store_true', 
                       help='Compare predictions with ground truth')
    
    args = parser.parse_args()
    
    # Setup paths
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) / results_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running multi-scale physics validation")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    predictions, ground_truth = load_velocity_data(results_dir, args.split)
    
    if predictions is None:
        print("No predictions found!")
        return
    
    print(f"Predictions shape: {predictions.shape}")
    if ground_truth is not None:
        print(f"Ground truth shape: {ground_truth.shape}")
    
    # Prepare velocity fields
    pred_velocity = prepare_velocity_field(predictions, args.sample_idx)
    print(f"Prepared prediction velocity field shape: {pred_velocity.shape}")
    
    ref_velocity = None
    if ground_truth is not None and args.compare_with_reference:
        ref_velocity = prepare_velocity_field(ground_truth, args.sample_idx)
        print(f"Prepared reference velocity field shape: {ref_velocity.shape}")
    
    # Initialize validator
    validator = MultiScalePhysicsValidator()
    
    # Run comprehensive multi-scale validation
    print(f"\n{'='*60}")
    print("MULTI-SCALE PHYSICS VALIDATION")
    print(f"{'='*60}")
    
    results = validator.comprehensive_multiscale_validation(
        pred_velocity, ref_velocity, args.viscosity
    )
    
    # Print results
    if 'overall' in results:
        overall_score = results['overall']['multiscale_physics_score']
        print(f"\nOverall Multi-Scale Physics Score: {overall_score:.3f}")
    
    if 'energy_cascade' in results:
        cascade_data = results['energy_cascade']
        print(f"\nEnergy Cascade:")
        print(f"  Total Energy: {cascade_data['total_energy']:.6f}")
        if 'inertial_slope' in cascade_data:
            print(f"  Inertial Range Slope: {cascade_data['inertial_slope']:.3f}")
            print(f"  Kolmogorov Deviation: {cascade_data['kolmogorov_deviation']:.3f}")
        if 'energy_ratio' in cascade_data:
            print(f"  Energy Ratio (pred/ref): {cascade_data['energy_ratio']:.3f}")
    
    if 'vorticity_dynamics' in results:
        vort_data = results['vorticity_dynamics']
        print(f"\nVorticity Dynamics:")
        print(f"  Mean Vorticity Magnitude: {vort_data['mean_vorticity_magnitude']:.6f}")
        print(f"  Total Enstrophy: {vort_data['total_enstrophy']:.6f}")
        if 'enstrophy_ratio' in vort_data:
            print(f"  Enstrophy Ratio (pred/ref): {vort_data['enstrophy_ratio']:.3f}")
    
    if 'dissipation_scales' in results:
        diss_data = results['dissipation_scales']
        print(f"\nDissipation Scales:")
        print(f"  Kolmogorov Length: {diss_data['kolmogorov_length']:.6f}")
        print(f"  Taylor Microscale: {diss_data['taylor_microscale']:.6f}")
        print(f"  Reynolds Number (Taylor): {diss_data['reynolds_taylor']:.1f}")
    
    # Save results
    results_path = output_dir / 'multiscale_physics_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = output_dir / 'multiscale_physics_plots.png'
    validator.plot_multiscale_validation(results, str(plot_path))
    print(f"Plots saved to: {plot_path}")
    
    # Generate report
    report = validator.generate_multiscale_report(results)
    report_path = output_dir / 'multiscale_physics_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    print(f"\nMulti-scale physics validation completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
