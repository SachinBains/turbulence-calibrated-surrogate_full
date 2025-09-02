# Turbulence Surrogate Analysis - Reproducibility Guide
**Generated**: 2025-09-01 20:57:44

## Environment Setup

### Python Environment
- **Python Version**: 3.12.7
- **Platform**: win32

### Required Packages
Install dependencies using:
```bash
pip install -r requirements.txt
```

**Critical Package Versions:**
- numpy: 1.26.4
- pandas: 2.2.2
- matplotlib: 3.9.2
- scipy: 1.13.1
- torch: 2.7.1+cpu
- seaborn: 0.13.2

## Data Requirements

### CSF3 Data Access
1. Access to CSF3 HPC system with trained model artifacts
2. Download prediction arrays from CSF3:
   ```bash
   # On CSF3:
   cd /path/to/artifacts
   tar -czf predictions.tar.gz results/*/mc_*_test.npy results/*/ens_*_test.npy
   scp predictions.tar.gz local_machine:/path/to/step10_visualization/
   ```

### Local Data Structure
Ensure the following directory structure:
```
turbulence-calibrated-surrogate_full/
├── scripts/           # Analysis scripts
├── src/              # Source code
├── configs/          # Experiment configurations
├── step9_analysis/   # Aggregated metrics
├── step10_analysis/  # Visualization results
├── step11_analysis/  # Quantitative comparison
├── step12_analysis/  # Physics validation
├── step13_analysis/  # Interpretability analysis
└── step14_summary/   # Final reports
```

## Execution Pipeline

Run the analysis pipeline in order:

### Step 9: Aggregate experiment metrics and logs
```bash
python scripts/step9_aggregate_results.py
```

### Step 10: Generate error and uncertainty visualizations
```bash
python scripts/step10_error_uncertainty_maps.py
```

### Step 11: Quantitative UQ method comparison
```bash
python scripts/step11_quantitative_comparison.py
```

### Step 12: Physics consistency validation
```bash
python scripts/step12_physics_validation.py
```

### Step 13: Interpretability and feature analysis
```bash
python scripts/step13_interpretability_analysis.py
```

### Step 14: Generate comprehensive summary report
```bash
python scripts/step14_summary_report.py
```

## Expected Outputs

After successful execution, you should have:
- **CSV files**: Aggregated metrics and summary tables
- **PNG files**: Visualization plots and comparison figures
- **JSON files**: Detailed analysis results
- **TEX files**: LaTeX tables for publication
- **MD/HTML files**: Comprehensive analysis reports

## Validation

To validate successful reproduction:
1. Check that all output directories contain expected files
2. Verify file checksums match the provided manifest
3. Review generated plots for consistency
4. Compare summary metrics with reference values

## Troubleshooting

### Common Issues
- **Missing prediction files**: Ensure CSF3 data is properly downloaded
- **Unicode errors**: Use UTF-8 encoding for all text files
- **Memory issues**: Process data in batches if needed
- **Package conflicts**: Use virtual environment with exact versions

### Contact Information
For questions about reproduction, refer to:
- Original experiment configurations in `configs/`
- Detailed logs in CSF3 artifacts
- Pipeline validation results in `step15_backup/`
