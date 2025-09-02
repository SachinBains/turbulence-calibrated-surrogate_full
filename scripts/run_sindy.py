#!/usr/bin/env python3
"""
Train SINDy model for physics discovery in turbulence data.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.dataio.hit_dataset import HITDataset
from src.models.sindy import TurbulenceSINDy

def compute_time_derivatives(velocity_sequence, dt=1.0):
    """Compute time derivatives using finite differences."""
    if len(velocity_sequence.shape) == 4:  # (T, C, H, W)
        # Use central differences where possible
        derivatives = np.zeros_like(velocity_sequence)
        
        # Forward difference for first point
        derivatives[0] = (velocity_sequence[1] - velocity_sequence[0]) / dt
        
        # Central differences for middle points
        for t in range(1, len(velocity_sequence) - 1):
            derivatives[t] = (velocity_sequence[t+1] - velocity_sequence[t-1]) / (2 * dt)
        
        # Backward difference for last point
        derivatives[-1] = (velocity_sequence[-1] - velocity_sequence[-2]) / dt
        
        return derivatives
    else:
        raise ValueError("Expected 4D velocity sequence (T, C, H, W)")

def extract_spatial_samples(velocity_fields, n_samples=1000):
    """Extract random spatial samples from velocity fields."""
    samples = []
    derivatives = []
    
    for t in range(len(velocity_fields)):
        field = velocity_fields[t]  # (C, H, W)
        
        # Random spatial sampling
        h, w = field.shape[1], field.shape[2]
        sample_indices = np.random.choice(h * w, n_samples, replace=False)
        
        # Flatten spatial dimensions
        field_flat = field.reshape(field.shape[0], -1)  # (C, H*W)
        
        # Extract samples
        for idx in sample_indices:
            sample = field_flat[:, idx]  # (C,) - velocity components
            samples.append(sample)
    
    return np.array(samples)

def main():
    parser = argparse.ArgumentParser(description='Train SINDy for turbulence physics discovery')
    parser.add_argument('--config', required=True, help='Config file path')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='train', help='Dataset split')
    parser.add_argument('--alpha', type=float, default=0.01, help='SINDy regularization strength')
    parser.add_argument('--n_samples', type=int, default=5000, help='Number of spatial samples per timestep')
    parser.add_argument('--n_timesteps', type=int, default=50, help='Number of timesteps to use')
    parser.add_argument('--dt', type=float, default=1.0, help='Time step for derivatives')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load config
    cfg = load_config(args.config)
    exp_id = cfg['experiment_id']
    
    # Setup output directories
    results_dir = Path(cfg['paths']['results_dir']) / exp_id / 'sindy'
    figures_dir = Path('figures') / exp_id / 'sindy'
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Training SINDy for experiment: {exp_id}")
    print(f"Alpha: {args.alpha}, Samples: {args.n_samples}, Timesteps: {args.n_timesteps}")
    
    # Load dataset
    dataset = HITDataset(cfg, args.split, eval_mode=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Collect velocity sequences
    print("Loading velocity data...")
    velocity_sequences = []
    count = 0
    
    for x, _ in loader:
        if count >= args.n_timesteps:
            break
        
        # Extract velocity components (assume first 3 channels are u, v, w)
        velocity = x[0, :3].numpy()  # (3, H, W)
        velocity_sequences.append(velocity)
        count += 1
        
        if count % 10 == 0:
            print(f"  Loaded {count}/{args.n_timesteps} timesteps")
    
    velocity_sequences = np.array(velocity_sequences)  # (T, 3, H, W)
    print(f"Loaded velocity sequences: {velocity_sequences.shape}")
    
    # Compute time derivatives
    print("Computing time derivatives...")
    time_derivatives = compute_time_derivatives(velocity_sequences, args.dt)
    
    # Extract spatial samples
    print("Extracting spatial samples...")
    velocity_samples = extract_spatial_samples(velocity_sequences, args.n_samples)
    derivative_samples = extract_spatial_samples(time_derivatives, args.n_samples)
    
    print(f"Extracted samples: {velocity_samples.shape}, {derivative_samples.shape}")
    
    # Train SINDy
    print("Training SINDy model...")
    sindy = TurbulenceSINDy(alpha=args.alpha)
    sindy.fit(velocity_samples, derivative_samples)
    
    # Get discovered equations
    equations = sindy.get_turbulence_equations()
    
    print("\nDiscovered Turbulence Equations:")
    print("=" * 50)
    for i, eq in enumerate(equations):
        print(f"{i+1}. {eq}")
    
    # Evaluate model
    print("\nEvaluating SINDy model...")
    r2_score = sindy.score(velocity_samples, derivative_samples)
    print(f"R² Score: {r2_score:.4f}")
    
    # Save results
    results = {
        'equations': equations,
        'r2_score': float(r2_score),
        'alpha': args.alpha,
        'n_samples': args.n_samples,
        'n_timesteps': args.n_timesteps,
        'coefficients': sindy.model.coef_.tolist(),
        'feature_names': sindy.model.feature_names_
    }
    
    results_file = results_dir / 'sindy_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Create visualization
    print("Creating visualizations...")
    
    # Plot coefficient magnitudes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    component_names = ['u', 'v', 'w']
    
    for i in range(3):
        coefs = sindy.model.coef_[:, i]
        feature_names = sindy.model.feature_names_
        
        # Only plot non-zero coefficients
        nonzero_mask = np.abs(coefs) > 1e-6
        if np.any(nonzero_mask):
            nonzero_coefs = coefs[nonzero_mask]
            nonzero_names = [feature_names[j] for j in range(len(feature_names)) if nonzero_mask[j]]
            
            axes[i].barh(range(len(nonzero_coefs)), nonzero_coefs)
            axes[i].set_yticks(range(len(nonzero_names)))
            axes[i].set_yticklabels(nonzero_names, fontsize=8)
            axes[i].set_xlabel('Coefficient Value')
            axes[i].set_title(f'd{component_names[i]}/dt')
            axes[i].grid(True, alpha=0.3)
        else:
            axes[i].text(0.5, 0.5, 'No significant\ncoefficients', 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(f'd{component_names[i]}/dt')
    
    plt.tight_layout()
    coef_plot = figures_dir / 'sindy_coefficients.png'
    plt.savefig(coef_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot predictions vs truth
    n_test = min(1000, len(velocity_samples))
    test_indices = np.random.choice(len(velocity_samples), n_test, replace=False)
    
    X_test = velocity_samples[test_indices]
    y_test = derivative_samples[test_indices]
    y_pred = sindy.predict(X_test)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i in range(3):
        axes[i].scatter(y_test[:, i], y_pred[:, i], alpha=0.5, s=1)
        
        # Perfect prediction line
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        axes[i].set_xlabel(f'True d{component_names[i]}/dt')
        axes[i].set_ylabel(f'Predicted d{component_names[i]}/dt')
        axes[i].set_title(f'{component_names[i]} Component (R² = {r2_score:.3f})')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    pred_plot = figures_dir / 'sindy_predictions.png'
    plt.savefig(pred_plot, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Figures saved to: {figures_dir}")
    print("\nSINDy training completed!")

if __name__ == '__main__':
    main()
