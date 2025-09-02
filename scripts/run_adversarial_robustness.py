#!/usr/bin/env python3
"""
Adversarial Robustness Testing Script
Test model robustness against adversarial attacks for turbulence data.
"""

import os
import sys
import argparse
import torch
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import load_config
from src.utils.devices import pick_device
from src.dataio.hit_dataset import HITDataset
from src.models.unet3d import UNet3D
from src.eval.adversarial_robustness import RobustnessEvaluator
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Adversarial robustness testing for turbulence models')
    parser.add_argument('--config', required=True, help='Path to experiment config')
    parser.add_argument('--split', default='test', help='Dataset split to test')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--output_dir', default='robustness_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    device = pick_device()
    
    print(f"Running adversarial robustness testing for experiment: {exp_id}")
    print(f"Device: {device}")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    results_dir = Path(cfg['paths']['results_dir']) / exp_id
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
    model.eval()
    
    print(f"Loaded model from: {ckpts[-1]}")
    
    # Load dataset
    dataset = HITDataset(cfg, args.split, eval_mode=True)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Initialize robustness evaluator
    evaluator = RobustnessEvaluator(model, device)
    
    # Run comprehensive robustness analysis
    print(f"\n{'='*60}")
    print("ADVERSARIAL ROBUSTNESS ANALYSIS")
    print(f"{'='*60}")
    
    results = evaluator.comprehensive_robustness_analysis(data_loader)
    
    # Print results
    overall_metrics = results['overall_metrics']
    print(f"\nOverall Robustness Score: {overall_metrics['overall_robustness_score']:.3f}")
    print(f"Average Loss Increase: {overall_metrics['average_loss_increase']:.6f}")
    print(f"Average Prediction Change: {overall_metrics['average_prediction_change']:.6f}")
    print(f"Robustness Stability: {overall_metrics['robustness_stability']:.6f}")
    
    # Print attack-specific results
    print(f"\nAttack-Specific Results:")
    attack_results = results['attack_results']
    
    for attack_name, attack_result in attack_results.items():
        print(f"\n{attack_name.upper()}:")
        print(f"  Robustness Score: {attack_result.get('robustness_score', 0):.3f}")
        print(f"  Loss Increase: {attack_result.get('loss_increase', 0):.6f}")
        print(f"  Prediction Change: {attack_result.get('prediction_change_mean', 0):.6f}")
    
    # Save results
    results_path = output_dir / 'robustness_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = output_dir / 'robustness_plots.png'
    evaluator.plot_robustness_analysis(results, str(plot_path))
    print(f"Plots saved to: {plot_path}")
    
    # Generate report
    report = evaluator.generate_robustness_report(results)
    report_path = output_dir / 'robustness_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved to: {report_path}")
    
    print(f"\nAdversarial robustness analysis completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
