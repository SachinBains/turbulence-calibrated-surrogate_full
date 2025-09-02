#!/usr/bin/env python3
"""
Validation script to check pipeline reproduction
"""

import sys
from pathlib import Path
import pandas as pd
import json

def validate_reproduction():
    """Validate that all pipeline steps completed successfully"""
    
    base_dir = Path.cwd()
    
    # Expected outputs for each step
    expected_outputs = {
        'step9_analysis': ['aggregated_metrics.csv', 'training_logs_summary.csv'],
        'step10_analysis': ['method_statistics_summary.csv'],
        'step11_analysis': ['performance_comparison_table.csv', 'uncertainty_quality_metrics.csv'],
        'step12_analysis': ['physics_properties_summary.csv', 'detailed_physics_results.json'],
        'step13_analysis': ['interpretability_summary.csv', 'detailed_interpretability_results.json'],
        'step14_summary': ['turbulence_analysis_report.md', 'summary_table.tex']
    }
    
    print("=== Pipeline Validation ===\n")
    
    all_valid = True
    
    for step_dir, expected_files in expected_outputs.items():
        step_path = base_dir / step_dir
        print(f"Checking {step_dir}...")
        
        if not step_path.exists():
            print(f"  ERROR: Directory {step_dir} does not exist")
            all_valid = False
            continue
        
        missing_files = []
        for expected_file in expected_files:
            file_path = step_path / expected_file
            if file_path.exists():
                print(f"  ✓ {expected_file}")
            else:
                print(f"  ✗ {expected_file} (missing)")
                missing_files.append(expected_file)
                all_valid = False
        
        if missing_files:
            print(f"  Missing files: {missing_files}")
        print()
    
    if all_valid:
        print("🎉 All pipeline steps validated successfully!")
        print("\nReproduction appears to be complete.")
    else:
        print("⚠️  Some validation checks failed.")
        print("\nPlease check missing files and re-run failed steps.")
    
    return all_valid

if __name__ == "__main__":
    validate_reproduction()
