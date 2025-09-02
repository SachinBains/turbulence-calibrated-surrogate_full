#!/usr/bin/env python3
"""
Temporal Consistency Validation Script
Validate temporal consistency of turbulence predictions for time-series data.
"""

import os
import sys
import argparse
import torch
import numpy as np
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.eval.temporal_consistency import TemporalConsistencyValidator
from torch.utils.data import DataLoader

def load_temporal_predictions(results_dir: Path, split: str = 'test') -> tuple:
    """Load temporal predictions and ground truth."""
    
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

def generate_temporal_sequence(model: torch.nn.Module, dataset, device: torch.device,
                             n_samples: int = 20, sequence_length: int = 10) -> tuple:
    """Generate temporal sequence predictions."""
    
    model.eval()
    predictions = []
    ground_truths = []
    
    print(f"Generating temporal sequences...")
    
    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            sample_input, sample_target = dataset[i]
            
            # Generate sequence by iterative prediction
            current_input = sample_input.unsqueeze(0).to(device)
            sequence_preds = []
            sequence_targets = []
            
            for t in range(sequence_length):
                # Predict next step
                pred = model(current_input)
                sequence_preds.append(pred.cpu().numpy()[0])
                
                # Use prediction as next input (autoregressive)
                current_input = pred
                
                # For ground truth, we'll use the same target (simplified)
                sequence_targets.append(sample_target.numpy())
            
            predictions.append(np.array(sequence_preds))
            ground_truths.append(np.array(sequence_targets))
    
    return np.array(predictions), np.array(ground_truths)

def main():
    parser = argparse.ArgumentParser(description='Temporal consistency validation for turbulence models')
    parser.add_argument('--config', required=True, help='Path to experiment config')
    parser.add_argument('--split', default='test', help='Dataset split to analyze')
    parser.add_argument('--n_samples', type=int, default=20, help='Number of samples for temporal analysis')
    parser.add_argument('--sequence_length', type=int, default=10, help='Length of temporal sequences')
    parser.add_argument('--dt', type=float, default=1.0, help='Time step for spectral analysis')
    parser.add_argument('--output_dir', default='temporal_analysis', help='Output directory')
    parser.add_argument('--generate_sequences', action='store_true', 
                       help='Generate new temporal sequences instead of using existing predictions')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    device = pick_device()
    
    print(f"Running temporal consistency validation for experiment: {exp_id}")
    print(f"Device: {device}")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load or generate temporal data
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
    
    if args.generate_sequences:
        print("Generating new temporal sequences...")
        
        # Load model
        ckpts = sorted(results_dir.glob('best_*.pth'))
        if not ckpts:
            print(f"No checkpoint found in {results_dir}")
            return
        
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
        
        # Load dataset
        dataset = HITDataset(cfg, args.split, eval_mode=True)
        
        # Generate temporal sequences
        predictions, ground_truth = generate_temporal_sequence(
            model, dataset, device, args.n_samples, args.sequence_length
        )
        
        print(f"Generated predictions shape: {predictions.shape}")
        print(f"Generated ground truth shape: {ground_truth.shape}")
        
    else:
        print("Loading existing predictions...")
        predictions, ground_truth = load_temporal_predictions(results_dir, args.split)
        
        if predictions is None or ground_truth is None:
            print("Could not load predictions or ground truth. Try --generate_sequences")
            return
        
        print(f"Loaded predictions shape: {predictions.shape}")
        print(f"Loaded ground truth shape: {ground_truth.shape}")
        
        # For existing predictions, create pseudo-temporal sequences
        if predictions.ndim == 4:  # (N, C, H, W)
            # Reshape to create temporal dimension
            n_samples = min(args.n_samples, predictions.shape[0])
            seq_len = min(args.sequence_length, predictions.shape[0] // n_samples)
            
            predictions = predictions[:n_samples * seq_len].reshape(n_samples, seq_len, *predictions.shape[1:])
            ground_truth = ground_truth[:n_samples * seq_len].reshape(n_samples, seq_len, *ground_truth.shape[1:])
    
    # Initialize temporal consistency validator
    # Use dummy model for validation (only needs predictions)
    dummy_model = torch.nn.Identity()
    validator = TemporalConsistencyValidator(dummy_model, device)
    
    # Run temporal consistency validation
    print(f"\n{'='*60}")
    print("TEMPORAL CONSISTENCY VALIDATION")
    print(f"{'='*60}")
    
    # Process each sample sequence
    all_results = []
    
    for sample_idx in range(predictions.shape[0]):
        print(f"\nProcessing sample {sample_idx + 1}/{predictions.shape[0]}...")
        
        sample_pred = predictions[sample_idx]  # (T, C, H, W)
        sample_gt = ground_truth[sample_idx]   # (T, C, H, W)
        
        # Run validation for this sample
        sample_results = validator.comprehensive_temporal_validation(
            sample_pred, sample_gt, args.dt
        )
        
        all_results.append(sample_results)
    
    # Aggregate results across samples
    print(f"\nAggregating results across {len(all_results)} samples...")
    
    aggregated_results = {}
    
    # Aggregate each metric type
    for metric_type in ['autocorrelation', 'spectral', 'phase', 'energy', 'divergence', 'overall']:
        if metric_type in all_results[0]:
            if metric_type == 'overall':
                scores = [result[metric_type]['temporal_consistency_score'] for result in all_results]
                aggregated_results[metric_type] = {
                    'temporal_consistency_score_mean': np.mean(scores),
                    'temporal_consistency_score_std': np.std(scores),
                    'temporal_consistency_score_min': np.min(scores),
                    'temporal_consistency_score_max': np.max(scores)
                }
            else:
                # For other metrics, aggregate key statistics
                aggregated_results[metric_type] = {}
                
                # Get all keys from first sample
                sample_keys = all_results[0][metric_type].keys()
                
                for key in sample_keys:
                    if isinstance(all_results[0][metric_type][key], (int, float)):
                        values = [result[metric_type][key] for result in all_results 
                                if key in result[metric_type]]
                        if values:
                            aggregated_results[metric_type][f'{key}_mean'] = np.mean(values)
                            aggregated_results[metric_type][f'{key}_std'] = np.std(values)
    
    # Print aggregated results
    if 'overall' in aggregated_results:
        overall_mean = aggregated_results['overall']['temporal_consistency_score_mean']
        overall_std = aggregated_results['overall']['temporal_consistency_score_std']
        print(f"\nOverall Temporal Consistency Score: {overall_mean:.3f} ± {overall_std:.3f}")
    
    if 'phase' in aggregated_results:
        phase_score = aggregated_results['phase'].get('phase_consistency_score_mean', 0)
        print(f"Phase Consistency Score: {phase_score:.3f}")
    
    if 'energy' in aggregated_results:
        energy_score = aggregated_results['energy'].get('energy_conservation_score_mean', 0)
        print(f"Energy Conservation Score: {energy_score:.3f}")
    
    # Save results
    results_path = output_dir / 'temporal_consistency_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'aggregated_results': aggregated_results,
            'individual_results': all_results,
            'config': {
                'n_samples': len(all_results),
                'sequence_length': args.sequence_length,
                'dt': args.dt
            }
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots using first sample for visualization
    if all_results:
        plot_path = output_dir / 'temporal_consistency_plots.png'
        validator.plot_temporal_validation(all_results[0], str(plot_path))
        print(f"Plots saved to: {plot_path}")
    
    # Generate report
    if all_results:
        report = validator.generate_temporal_report(all_results[0])
        
        # Add aggregated summary
        if 'overall' in aggregated_results:
            overall_mean = aggregated_results['overall']['temporal_consistency_score_mean']
            report += f"\n## Aggregated Results (n={len(all_results)} samples)\n"
            report += f"- Mean Temporal Consistency Score: {overall_mean:.3f}\n"
            report += f"- Standard Deviation: {aggregated_results['overall']['temporal_consistency_score_std']:.3f}\n"
        
        report_path = output_dir / 'temporal_consistency_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Report saved to: {report_path}")
    
    print(f"\nTemporal consistency validation completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
