#!/usr/bin/env python3
"""
Distribution Shift Detection Script
Detect and analyze distribution shifts in turbulence data for robust deployment.
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
from src.eval.distribution_shift import DistributionShiftDetector, DomainAdaptationAnalyzer
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Distribution shift detection for turbulence models')
    parser.add_argument('--config', required=True, help='Path to experiment config')
    parser.add_argument('--test_split', default='test', help='Test split to analyze')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--output_dir', default='shift_analysis', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    device = pick_device()
    
    print(f"Running distribution shift analysis for experiment: {exp_id}")
    print(f"Device: {device}")
    
    # Setup output directory
    output_dir = Path(args.output_dir) / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    train_dataset = HITDataset(cfg, 'train', eval_mode=True)
    test_dataset = HITDataset(cfg, args.test_split, eval_mode=True)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Sample data for analysis
    n_train = min(args.n_samples, len(train_dataset))
    n_test = min(args.n_samples, len(test_dataset))
    
    # Collect reference (training) data
    train_data = []
    for i in range(n_train):
        sample, _ = train_dataset[i]
        train_data.append(sample)
    
    reference_data = torch.stack(train_data)
    
    # Collect test data
    test_data = []
    for i in range(n_test):
        sample, _ = test_dataset[i]
        test_data.append(sample)
    
    test_data = torch.stack(test_data)
    
    print(f"Reference data shape: {reference_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Initialize shift detector
    shift_detector = DistributionShiftDetector(reference_data, device)
    
    # Run comprehensive shift analysis
    print("\n" + "="*60)
    print("DISTRIBUTION SHIFT ANALYSIS")
    print("="*60)
    
    shift_results = shift_detector.comprehensive_shift_analysis(test_data)
    
    # Print results
    print(f"\nOverall shift detected: {shift_results['overall']['shift_detected']}")
    print(f"Confidence: {shift_results['overall']['confidence']:.3f}")
    print(f"Methods agreeing: {shift_results['overall']['methods_agreeing']}/3")
    
    print(f"\nStatistical tests:")
    stat_summary = shift_results['statistical']['summary']
    print(f"  Significant tests: {stat_summary['significant_tests']}/{stat_summary['total_tests']}")
    print(f"  Shift confidence: {stat_summary['shift_confidence']:.3f}")
    
    print(f"\nMMD test:")
    mmd_results = shift_results['mmd']
    print(f"  MMD (PCA): {mmd_results['mmd_pca']:.6f}")
    print(f"  Shift detected: {mmd_results['shift_detected_pca']}")
    
    print(f"\nClassifier test:")
    clf_results = shift_results['classifier']
    print(f"  AUC: {clf_results['classifier_auc']:.3f} ± {clf_results['classifier_auc_std']:.3f}")
    print(f"  Shift detected: {clf_results['shift_detected']}")
    
    # Save results
    results_path = output_dir / 'shift_analysis.json'
    with open(results_path, 'w') as f:
        json.dump(shift_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_path}")
    
    # Create plots
    plot_path = output_dir / 'shift_analysis_plots.png'
    shift_detector.plot_shift_analysis(test_data, shift_results, str(plot_path))
    print(f"Plots saved to: {plot_path}")
    
    # Domain adaptation analysis (if model available)
    try:
        # Load model
        results_dir = Path(cfg['paths']['results_dir']) / exp_id
        ckpts = sorted(results_dir.glob('best_*.pth'))
        
        if ckpts:
            print(f"\n" + "="*60)
            print("DOMAIN ADAPTATION ANALYSIS")
            print("="*60)
            
            # Build and load model
            mcfg = cfg['model']
            model = UNet3D(
                mcfg['in_channels'], 
                mcfg['out_channels'], 
                base_ch=mcfg['base_channels']
            )
            
            state = torch.load(ckpts[-1], map_location=device)
            model.load_state_dict(state['model'])
            
            # Initialize domain adaptation analyzer
            domain_analyzer = DomainAdaptationAnalyzer(model, device)
            
            # Analyze domain gap
            domain_gap_results = domain_analyzer.analyze_domain_gap(
                reference_data[:20], test_data[:20]  # Use subset for efficiency
            )
            
            print(f"Feature shift: {domain_gap_results['feature_shift']:.6f}")
            print(f"Prediction shift: {domain_gap_results['prediction_shift']:.6f}")
            print(f"Overall domain gap: {domain_gap_results['domain_gap']:.6f}")
            
            # Get adaptation recommendations
            adaptation_strategy = domain_analyzer.suggest_adaptation_strategy(
                domain_gap_results['domain_gap']
            )
            
            print(f"\nRecommended strategy: {adaptation_strategy['strategy']}")
            print("Recommendations:")
            for rec in adaptation_strategy['recommendations']:
                print(f"  - {rec}")
            
            # Save domain adaptation results
            domain_results = {
                'domain_gap': domain_gap_results,
                'adaptation_strategy': adaptation_strategy
            }
            
            domain_path = output_dir / 'domain_adaptation.json'
            with open(domain_path, 'w') as f:
                json.dump(domain_results, f, indent=2, default=str)
            print(f"\nDomain adaptation results saved to: {domain_path}")
            
    except Exception as e:
        print(f"\nDomain adaptation analysis failed: {e}")
    
    print(f"\nDistribution shift analysis completed!")
    print(f"Results saved in: {output_dir}")

if __name__ == '__main__':
    main()
