#!/usr/bin/env python3
"""
Q-criterion Analysis Script
Computes and visualizes Q-criterion isosurfaces for turbulence data.
"""

import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.physics.q_criterion import QCriterionAnalyzer, TurbulenceVisualization
from torch.utils.data import DataLoader

def load_predictions_and_truth(results_dir: Path, split: str = 'test') -> Dict[str, np.ndarray]:
    """Load prediction arrays and ground truth."""
    data = {}
    
    # Load ground truth
    gt_path = results_dir / f'gt_{split}.npy'
    if gt_path.exists():
        data['ground_truth'] = np.load(gt_path)
        print(f"Loaded ground truth: {data['ground_truth'].shape}")
    
    # Load predictions (try different naming conventions)
    pred_patterns = [
        f'pred_{split}.npy',
        f'mean_{split}.npy', 
        f'mc_mean_{split}.npy',
        f'ens_mean_{split}.npy'
    ]
    
    for pattern in pred_patterns:
        pred_path = results_dir / pattern
        if pred_path.exists():
            pred_name = pattern.replace(f'_{split}.npy', '').replace('.npy', '')
            data[pred_name] = np.load(pred_path)
            print(f"Loaded {pred_name}: {data[pred_name].shape}")
    
    return data

def analyze_sample_q_criterion(velocity_fields: Dict[str, np.ndarray], 
                              sample_idx: int, output_dir: Path) -> Dict[str, Any]:
    """Analyze Q-criterion for a specific sample."""
    
    # Create output directory
    sample_dir = output_dir / f'sample_{sample_idx}'
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize visualization
    viz = TurbulenceVisualization()
    
    results = {}
    
    for name, vel_field in velocity_fields.items():
        print(f"\nAnalyzing Q-criterion for {name} (sample {sample_idx})...")
        
        # Extract single sample
        if vel_field.ndim == 4:  # (N, C, D, H, W)
            sample_vel = vel_field[sample_idx]
        elif vel_field.ndim == 5:  # (N, D, H, W, C)
            sample_vel = vel_field[sample_idx].transpose(3, 0, 1, 2)
        else:
            print(f"Unexpected velocity field shape for {name}: {vel_field.shape}")
            continue
        
        # Create method-specific output directory
        method_dir = sample_dir / name
        method_dir.mkdir(exist_ok=True)
        
        # Perform comprehensive analysis
        try:
            analysis_results = viz.create_comprehensive_visualization(
                sample_vel, save_dir=str(method_dir)
            )
            
            if 'error' not in analysis_results:
                results[name] = analysis_results
                
                # Save Q-field for later analysis
                np.save(method_dir / 'q_field.npy', analysis_results['q_field'])
                
                # Save statistics
                with open(method_dir / 'statistics.json', 'w') as f:
                    json.dump({
                        'isosurface_statistics': analysis_results['isosurface_statistics'],
                        'q_statistics': analysis_results['q_statistics']
                    }, f, indent=2)
                
                print(f"  - Found {analysis_results['n_isosurfaces']} isosurfaces")
                print(f"  - Q-criterion range: [{analysis_results['q_statistics']['min_q']:.4f}, "
                      f"{analysis_results['q_statistics']['max_q']:.4f}]")
                print(f"  - Positive Q fraction: {analysis_results['q_statistics']['positive_q_fraction']:.3f}")
            else:
                print(f"  - Error: {analysis_results['error']}")
                
        except Exception as e:
            print(f"  - Error analyzing {name}: {e}")
            continue
    
    # Create comparison if multiple methods
    if len(results) > 1:
        print("\nCreating comparison analysis...")
        try:
            comparison_vel_fields = {}
            for name in results.keys():
                if name in velocity_fields:
                    vel_field = velocity_fields[name]
                    if vel_field.ndim == 4:
                        comparison_vel_fields[name] = vel_field[sample_idx]
                    elif vel_field.ndim == 5:
                        comparison_vel_fields[name] = vel_field[sample_idx].transpose(3, 0, 1, 2)
            
            comparison_results = viz.compare_q_criterion(
                comparison_vel_fields, save_dir=str(sample_dir / 'comparison')
            )
            
            # Save comparison statistics
            comparison_stats = {}
            for name, comp_result in comparison_results.items():
                if comp_result['statistics']:
                    comparison_stats[name] = comp_result['statistics']
            
            with open(sample_dir / 'comparison_statistics.json', 'w') as f:
                json.dump(comparison_stats, f, indent=2)
                
            print("  - Comparison analysis completed")
            
        except Exception as e:
            print(f"  - Error in comparison: {e}")
    
    return results

def run_q_criterion_analysis(config_path: str, sample_indices: list = None, 
                           split: str = 'test') -> None:
    """Run comprehensive Q-criterion analysis."""
    
    # Load configuration
    cfg = load_config(config_path)
    exp_id = cfg['experiment_id']
    
    # Setup paths
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    output_dir = Path('step13_analysis') / 'q_criterion' / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Running Q-criterion analysis for experiment: {exp_id}")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load velocity field data
    velocity_data = load_predictions_and_truth(results_dir, split)
    
    if not velocity_data:
        print("No velocity data found!")
        return
    
    # Determine sample indices
    if sample_indices is None:
        # Use first few samples
        first_key = list(velocity_data.keys())[0]
        n_samples = velocity_data[first_key].shape[0]
        sample_indices = list(range(min(3, n_samples)))
    
    print(f"Analyzing samples: {sample_indices}")
    
    # Analyze each sample
    all_results = {}
    
    for sample_idx in sample_indices:
        print(f"\n{'='*60}")
        print(f"ANALYZING SAMPLE {sample_idx}")
        print(f"{'='*60}")
        
        sample_results = analyze_sample_q_criterion(
            velocity_data, sample_idx, output_dir
        )
        
        if sample_results:
            all_results[f'sample_{sample_idx}'] = sample_results
    
    # Create summary statistics
    print(f"\n{'='*60}")
    print("CREATING SUMMARY")
    print(f"{'='*60}")
    
    summary_stats = {}
    
    for sample_name, sample_results in all_results.items():
        sample_stats = {}
        
        for method_name, method_results in sample_results.items():
            if 'q_statistics' in method_results:
                sample_stats[method_name] = method_results['q_statistics']
        
        if sample_stats:
            summary_stats[sample_name] = sample_stats
    
    # Save summary
    summary_path = output_dir / 'summary_statistics.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Create aggregate analysis
    if len(all_results) > 1:
        create_aggregate_analysis(summary_stats, output_dir)
    
    print(f"\nQ-criterion analysis completed!")
    print(f"Results saved in: {output_dir}")

def create_aggregate_analysis(summary_stats: Dict, output_dir: Path) -> None:
    """Create aggregate analysis across samples and methods."""
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Collect data for analysis
    data_rows = []
    
    for sample_name, sample_data in summary_stats.items():
        for method_name, method_stats in sample_data.items():
            row = {
                'sample': sample_name,
                'method': method_name,
                **method_stats
            }
            data_rows.append(row)
    
    if not data_rows:
        return
    
    df = pd.DataFrame(data_rows)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean Q-criterion by method
    if 'mean_q' in df.columns:
        df.boxplot(column='mean_q', by='method', ax=axes[0, 0])
        axes[0, 0].set_title('Mean Q-criterion Distribution')
        axes[0, 0].set_xlabel('Method')
        axes[0, 0].set_ylabel('Mean Q')
    
    # Plot 2: Positive Q fraction by method
    if 'positive_q_fraction' in df.columns:
        df.boxplot(column='positive_q_fraction', by='method', ax=axes[0, 1])
        axes[0, 1].set_title('Positive Q Fraction Distribution')
        axes[0, 1].set_xlabel('Method')
        axes[0, 1].set_ylabel('Positive Q Fraction')
    
    # Plot 3: Q-criterion range (max - min)
    if 'max_q' in df.columns and 'min_q' in df.columns:
        df['q_range'] = df['max_q'] - df['min_q']
        df.boxplot(column='q_range', by='method', ax=axes[1, 0])
        axes[1, 0].set_title('Q-criterion Range Distribution')
        axes[1, 0].set_xlabel('Method')
        axes[1, 0].set_ylabel('Q Range')
    
    # Plot 4: Standard deviation
    if 'std_q' in df.columns:
        df.boxplot(column='std_q', by='method', ax=axes[1, 1])
        axes[1, 1].set_title('Q-criterion Std Distribution')
        axes[1, 1].set_xlabel('Method')
        axes[1, 1].set_ylabel('Q Std')
    
    plt.suptitle('Q-criterion Analysis Summary', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / 'aggregate_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save aggregate statistics
    aggregate_stats = {}
    
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        
        method_stats = {}
        for col in ['mean_q', 'std_q', 'positive_q_fraction', 'max_q', 'min_q']:
            if col in method_data.columns:
                method_stats[f'{col}_mean'] = float(method_data[col].mean())
                method_stats[f'{col}_std'] = float(method_data[col].std())
                method_stats[f'{col}_min'] = float(method_data[col].min())
                method_stats[f'{col}_max'] = float(method_data[col].max())
        
        aggregate_stats[method] = method_stats
    
    with open(output_dir / 'aggregate_statistics.json', 'w') as f:
        json.dump(aggregate_stats, f, indent=2)
    
    print("Aggregate analysis completed")

def main():
    parser = argparse.ArgumentParser(description='Q-criterion analysis for turbulence data')
    parser.add_argument('--config', required=True, help='Path to experiment config')
    parser.add_argument('--samples', nargs='+', type=int, default=None,
                       help='Sample indices to analyze (default: first 3)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test',
                       help='Dataset split to analyze')
    
    args = parser.parse_args()
    
    run_q_criterion_analysis(args.config, args.samples, args.split)

if __name__ == '__main__':
    main()
