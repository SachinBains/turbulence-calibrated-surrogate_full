#!/usr/bin/env python3
"""
Train Gaussian Process Regression for turbulence modeling with uncertainty quantification.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import time

from src.utils.config import load_config
from src.dataio.hit_dataset import HITDataset
from src.models.gpr import TurbulenceGPR

def main():
    parser = argparse.ArgumentParser(description='Train GPR for turbulence modeling')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train', help='Dataset split')
    parser.add_argument('--n_components', type=int, default=30, help='PCA components for dimensionality reduction')
    parser.add_argument('--kernel', choices=['rbf', 'matern'], default='rbf', help='GP kernel type')
    parser.add_argument('--alpha', type=float, default=1e-6, help='GP noise parameter')
    parser.add_argument('--n_train', type=int, default=100, help='Number of training samples')
    parser.add_argument('--n_test', type=int, default=50, help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    
    # Setup output directories
    results_dir = Path(cfg['paths']['results_dir']) / exp_id / 'gpr'
    figures_dir = Path('figures') / exp_id / 'gpr'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training GPR for experiment: {exp_id}")
    print(f"Components: {args.n_components}, Kernel: {args.kernel}, Alpha: {args.alpha}")
    print(f"Training samples: {args.n_train}, Test samples: {args.n_test}")
    
    # Load dataset
    dataset = HITDataset(cfg, args.split, eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Collect training data
    print("Loading training data...")
    X_train = []
    y_train = []
    
    for i, (x, y) in enumerate(loader):
        if i >= args.n_train:
            break
        
        X_train.append(x[0].numpy())  # Remove batch dimension
        y_train.append(y[0].numpy())
        
        if (i + 1) % 20 == 0:
            print(f"  Loaded {i+1}/{args.n_train} training samples")
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    
    # Initialize and train GPR
    print("Training GPR model...")
    start_time = time.time()
    
    gpr = TurbulenceGPR(
        n_components=args.n_components,
        kernel_type=args.kernel,
        alpha=args.alpha,
        normalize_y=True
    )
    
    gpr.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate on training data
    print("Evaluating on training data...")
    train_score = gpr.score(X_train, y_train)
    train_likelihood = gpr.get_marginal_likelihood()
    
    print(f"Training R² score: {train_score:.4f}")
    print(f"Log marginal likelihood: {train_likelihood:.4f}")
    
    # Load test data
    print("Loading test data...")
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    X_test = []
    y_test = []
    
    for i, (x, y) in enumerate(test_loader):
        if i >= args.n_test:
            break
        
        X_test.append(x[0].numpy())
        y_test.append(y[0].numpy())
    
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Make predictions
    print("Making predictions...")
    y_pred, y_std = gpr.predict(X_test, return_std=True)
    
    # Evaluate predictions
    test_score = gpr.score(X_test, y_test)
    
    # Compute metrics
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"Test R² score: {test_score:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Compute uncertainty calibration
    errors = np.abs(y_test - y_pred)
    mean_error = np.mean(errors)
    mean_uncertainty = np.mean(y_std)
    
    print(f"Mean prediction error: {mean_error:.4f}")
    print(f"Mean predicted uncertainty: {mean_uncertainty:.4f}")
    
    # Save results
    results = {
        'train_r2': float(train_score),
        'test_r2': float(test_score),
        'test_rmse': float(rmse),
        'test_mae': float(mae),
        'log_marginal_likelihood': float(train_likelihood),
        'mean_error': float(mean_error),
        'mean_uncertainty': float(mean_uncertainty),
        'training_time': float(training_time),
        'n_components': args.n_components,
        'kernel_type': args.kernel,
        'alpha': args.alpha,
        'n_train': args.n_train,
        'n_test': args.n_test
    }
    
    results_file = results_dir / 'gpr_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualizations
    print("Creating visualizations...")
    
    # 1. Prediction vs Truth scatter plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Flatten arrays for plotting
    y_test_flat = y_test.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)
    y_std_flat = y_std.reshape(-1)
    errors_flat = errors.reshape(-1)
    
    # Sample for plotting (too many points otherwise)
    n_plot = min(5000, len(y_test_flat))
    plot_indices = np.random.choice(len(y_test_flat), n_plot, replace=False)
    
    # Prediction vs Truth
    axes[0, 0].scatter(y_test_flat[plot_indices], y_pred_flat[plot_indices], 
                      alpha=0.5, s=1)
    min_val = min(y_test_flat.min(), y_pred_flat.min())
    max_val = max(y_test_flat.max(), y_pred_flat.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('True Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title(f'GPR Predictions (R² = {test_score:.3f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Uncertainty vs Error
    axes[0, 1].scatter(y_std_flat[plot_indices], errors_flat[plot_indices], 
                      alpha=0.5, s=1)
    axes[0, 1].set_xlabel('Predicted Uncertainty (σ)')
    axes[0, 1].set_ylabel('Prediction Error')
    axes[0, 1].set_title('Uncertainty vs Error')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Error histogram
    axes[1, 0].hist(errors_flat, bins=50, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Prediction Error')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Error Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty histogram
    axes[1, 1].hist(y_std_flat, bins=50, alpha=0.7, density=True)
    axes[1, 1].set_xlabel('Predicted Uncertainty')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Uncertainty Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    analysis_plot = figures_dir / 'gpr_analysis.png'
    plt.savefig(analysis_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Sample prediction visualization
    sample_idx = 0
    if len(y_test.shape) == 4:  # (N, C, H, W)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show first channel of first sample
        channel = 0
        slice_idx = y_test.shape[2] // 2  # Middle slice
        
        # Ground truth
        axes[0, 0].imshow(y_test[sample_idx, channel, slice_idx], cmap='viridis')
        axes[0, 0].set_title('Ground Truth')
        axes[0, 0].axis('off')
        
        # Prediction
        axes[0, 1].imshow(y_pred[sample_idx, channel, slice_idx], cmap='viridis')
        axes[0, 1].set_title('GPR Prediction')
        axes[0, 1].axis('off')
        
        # Error
        error_slice = np.abs(y_test[sample_idx, channel, slice_idx] - 
                           y_pred[sample_idx, channel, slice_idx])
        axes[0, 2].imshow(error_slice, cmap='Reds')
        axes[0, 2].set_title('Absolute Error')
        axes[0, 2].axis('off')
        
        # Uncertainty
        axes[1, 0].imshow(y_std[sample_idx, channel, slice_idx], cmap='Blues')
        axes[1, 0].set_title('Predicted Uncertainty')
        axes[1, 0].axis('off')
        
        # Error vs Uncertainty scatter
        error_flat = error_slice.flatten()
        std_flat = y_std[sample_idx, channel, slice_idx].flatten()
        axes[1, 1].scatter(std_flat, error_flat, alpha=0.5, s=1)
        axes[1, 1].set_xlabel('Uncertainty')
        axes[1, 1].set_ylabel('Error')
        axes[1, 1].set_title('Local Uncertainty vs Error')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Remove empty subplot
        axes[1, 2].remove()
        
        plt.tight_layout()
        sample_plot = figures_dir / 'gpr_sample_prediction.png'
        plt.savefig(sample_plot, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Figures saved to: {figures_dir}")
    print("\nGPR training and evaluation completed!")

if __name__ == '__main__':
    main()
