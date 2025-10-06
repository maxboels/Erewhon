# Project Reorganization Summary

**Date**: October 6, 2025

This document describes the recent reorganization of the project structure to improve modularity and maintainability.

## 🎯 Goals

1. **Separation of Concerns**: Keep policy-specific code separate from general utilities
2. **Better Organization**: Group related functionality together
3. **Improved Discoverability**: Make it easier to find tools based on their purpose

## 📊 Changes Made

### Dataset Tools → `/src/datasets/`

**Moved Files:**
- `src/policies/ACT/analyze_dataset.py` → `src/datasets/analysis/analyze_dataset.py`
- `src/policies/ACT/quick_dataset_analysis.py` → `src/datasets/analysis/quick_dataset_analysis.py`
- `src/policies/ACT/simple_analysis.py` → `src/datasets/analysis/simple_analysis.py`
- `src/policies/ACT/plot_training_signals.py` → `src/datasets/analysis/plot_training_signals.py`
- `src/policies/ACT/local_dataset_loader.py` → `src/datasets/local_dataset_loader.py`

**Rationale**: These tools analyze and load datasets independent of the ACT policy. They can be used with any policy implementation.

### Episode Visualization → `/src/robots/rover/src/eval/`

**Moved Files:**
- `src/policies/ACT/create_episode_animation.py` → `src/robots/rover/src/eval/create_episode_animation.py`
- `src/policies/ACT/episode_frame_viewer.py` → `src/robots/rover/src/eval/episode_frame_viewer.py`

**Rationale**: These tools visualize episode data recorded by the rover. They work with the episode format created by `episode_recorder.py` and belong with other evaluation tools.

### ACT Policy → `/src/policies/ACT/`

**Remaining Files (ACT-specific):**
- Training: `*_trainer.py`, `train_local_act.py`
- Inference: `inference_act.py`
- Testing: `test_*.py`
- Setup: `setup_lerobot.py`
- Pipeline: `tracer_pipeline.py`
- Config: `config/`
- Framework: `lerobot/`

**Rationale**: These files are specific to the ACT policy implementation and training.

## 📁 New Directory Structure

```
src/
├── datasets/                       # Dataset utilities
│   ├── analysis/                   # Analysis and visualization
│   │   ├── analyze_dataset.py
│   │   ├── quick_dataset_analysis.py
│   │   ├── simple_analysis.py
│   │   └── plot_training_signals.py
│   ├── local_dataset_loader.py
│   └── README.md
│
├── policies/
│   └── ACT/                        # ACT policy (focused)
│       ├── Training scripts
│       ├── Inference scripts
│       ├── Testing scripts
│       ├── config/
│       ├── lerobot/
│       └── README.md (updated)
│
└── robots/
    └── rover/
        └── src/
            ├── eval/               # Episode evaluation tools
            │   ├── validate_episode_data.py
            │   ├── clean_episode_data.py
            │   ├── pwm_statistics.py
            │   ├── episode_analyzer.py
            │   ├── create_episode_animation.py  ← MOVED
            │   └── episode_frame_viewer.py      ← MOVED
            ├── record/
            │   ├── episode_recorder.py
            │   └── episode_analyzer.py
            └── README.md (updated)
```

## 📝 Updated Documentation

### Rover README (`/src/robots/rover/README.md`)

**Added:**
- Complete usage examples for `create_episode_animation.py`
- Documentation for `episode_frame_viewer.py`
- Updated project structure diagram
- Updated key files table

**New sections:**
- Animation creation with various formats (MP4/GIF)
- Combined multi-episode animations
- Interactive frame viewer usage

### ACT README (`/src/policies/ACT/README.md`)

**Updated:**
- Reorganized to focus on ACT-specific content
- Added "Related Tools" section pointing to moved files
- Clearer directory structure documentation

### Dataset README (`/src/datasets/README.md`)

**Created:**
- New README documenting dataset tools
- Usage examples for analysis tools
- Description of each tool's purpose

## 🔧 Migration Guide

### For Users

If you have scripts or documentation referencing old paths, update them as follows:

**Dataset Analysis:**
```bash
# OLD
python3 src/policies/ACT/analyze_dataset.py

# NEW
python3 src/datasets/analysis/analyze_dataset.py
```

**Episode Animation:**
```bash
# OLD
python3 src/policies/ACT/create_episode_animation.py --data_dir ./episodes

# NEW
python3 src/robots/rover/src/eval/create_episode_animation.py --data_dir ./episodes
```

**Episode Frame Viewer:**
```bash
# OLD
python3 src/policies/ACT/episode_frame_viewer.py --episode-dir ./episodes/episode_20251006_220059

# NEW
python3 src/robots/rover/src/eval/episode_frame_viewer.py --episode-dir ./episodes/episode_20251006_220059
```

### For Developers

**Import Paths**: If you're importing these modules in code, update import statements:

```python
# OLD
from policies.ACT.local_dataset_loader import load_dataset

# NEW
from datasets.local_dataset_loader import load_dataset
```

## ✅ Benefits

1. **Clearer Purpose**: Each directory has a focused purpose
2. **Easier Navigation**: Tools are where you'd expect them
3. **Better Reusability**: Dataset and visualization tools can be used across different policies
4. **Reduced Clutter**: ACT folder is now focused on ACT-specific code
5. **Improved Documentation**: Each area has its own README

## 🔍 Quick Reference

| Tool Type | Location |
|-----------|----------|
| ACT Training | `/src/policies/ACT/` |
| Dataset Analysis | `/src/datasets/analysis/` |
| Dataset Loading | `/src/datasets/` |
| Episode Recording | `/src/robots/rover/src/record/` |
| Episode Validation | `/src/robots/rover/src/eval/` |
| Episode Visualization | `/src/robots/rover/src/eval/` |

## 📞 Questions?

If you have questions about where to find a specific tool or where to add new code, refer to:
- This document for the overall structure
- Individual README files in each directory for specific tools
- The rover README for data collection and evaluation workflow
