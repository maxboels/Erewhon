# Dataset Tools

This directory contains tools for working with datasets.

## Directory Structure

```
datasets/
├── analysis/              # Dataset analysis and visualization tools
│   ├── analyze_dataset.py
│   ├── quick_dataset_analysis.py
│   ├── simple_analysis.py
│   └── plot_training_signals.py
└── local_dataset_loader.py  # Dataset loading utilities
```

## Analysis Tools

### analyze_dataset.py
Comprehensive dataset analysis tool for examining episode data quality and statistics.

### quick_dataset_analysis.py
Quick overview of dataset characteristics and metrics.

### simple_analysis.py
Simple analysis utilities for basic dataset inspection.

### plot_training_signals.py
Visualization tool for plotting training signals from datasets.

## Dataset Loader

### local_dataset_loader.py
Utilities for loading local datasets in various formats.

## Usage

These tools can be used independently or as part of the training pipeline to:
- Validate dataset quality before training
- Visualize data distributions
- Debug data loading issues
- Generate dataset statistics reports
