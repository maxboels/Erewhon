# Sparse Checkout Setup for Raspberry Pi

This guide shows how to set up the repository on Raspberry Pi with only the `src/robots/rover` directory checked out, including the new quantization documentation.

## 🎯 What is Sparse Checkout?

Sparse checkout allows you to clone only a subset of the repository, saving disk space and reducing sync time on the Pi.

**Without Sparse Checkout:**
- Full repo: ~2 GB (includes all policies, training data, etc.)
- Sync time: 5-10 minutes

**With Sparse Checkout:**
- Rover only: ~50 MB (just what Pi needs)
- Sync time: 30 seconds

## 📦 What Gets Checked Out

With sparse checkout, the Pi will have:

```
~/Erewhon/
└── src/
    └── robots/
        └── rover/
            ├── README.md
            ├── requirements.txt
            ├── requirements_pi.txt
            ├── PI_DEPLOYMENT_COMMANDS.md
            ├── QUANTIZATION_DEPLOYMENT_SUMMARY.md
            ├── docs/
            │   ├── quantization/           # ⭐ NEW!
            │   │   ├── README.md
            │   │   ├── QUANTIZATION_GUIDE.md
            │   │   ├── QUANTIZATION_WORKFLOW.md
            │   │   ├── ONNX_AND_FORMATS_EXPLAINED.md
            │   │   └── HAILO_DEPLOYMENT_GUIDE.md
            │   └── ...
            ├── episodes/
            ├── models/
            └── src/
                └── inference/
                    └── act_inference_quantized.py
```

## 🚀 Initial Setup on Raspberry Pi

### Option 1: Fresh Clone with Sparse Checkout

```bash
# 1. Create project directory
mkdir -p ~/projects
cd ~/projects

# 2. Initialize repo with sparse checkout
git clone --no-checkout https://github.com/maxboels/Erewhon.git
cd Erewhon

# 3. Enable sparse checkout
git sparse-checkout init --cone

# 4. Set sparse checkout paths
git sparse-checkout set src/robots/rover

# 5. Checkout main branch
git checkout main

# 6. Verify
ls -la src/robots/rover/docs/quantization/
```

**Expected Output:**
```
README.md
QUANTIZATION_GUIDE.md
QUANTIZATION_WORKFLOW.md
ONNX_AND_FORMATS_EXPLAINED.md
HAILO_DEPLOYMENT_GUIDE.md
```

### Option 2: Update Existing Sparse Checkout

If you already have the repo cloned with sparse checkout:

```bash
cd ~/projects/Erewhon

# Update sparse checkout configuration to include new docs
git sparse-checkout set src/robots/rover

# Pull latest changes
git pull origin main

# Verify new docs are present
ls -la src/robots/rover/docs/quantization/
```

## 🔄 Updating the Pi Repository

### Regular Updates

```bash
cd ~/projects/Erewhon
git pull origin main
```

This will only update the `src/robots/rover` directory, saving bandwidth.

### Check What's Included

```bash
# See sparse checkout configuration
git sparse-checkout list

# Should show:
# src/robots/rover
```

### Temporarily View Other Files

If you need to access files outside the sparse checkout:

```bash
# Add another path temporarily
git sparse-checkout add src/policies/ACT

# Do what you need...

# Remove it again
git sparse-checkout set src/robots/rover
```

## 📚 Accessing Quantization Documentation

After setup, all quantization docs are available locally:

```bash
cd ~/src/robots/rover/docs/quantization

# View README
cat README.md

# View specific guide
less QUANTIZATION_GUIDE.md

# Or use a text editor
nano QUANTIZATION_WORKFLOW.md
```

## 🔧 Configuration Files

### Git Sparse Checkout Config

Located at `.git/info/sparse-checkout`:

```
src/robots/rover
```

### Deployment Benefits

Having docs on the Pi means:
- ✅ **Offline access** - No need for internet to check guides
- ✅ **Version matched** - Docs match the deployed code
- ✅ **Quick reference** - Commands available at terminal
- ✅ **Troubleshooting** - Guides available when debugging

## 📋 Quick Reference Commands

### Navigate to Docs
```bash
cd ~/src/robots/rover/docs/quantization
ls -lh  # List all guides
```

### Search Documentation
```bash
cd ~/src/robots/rover
grep -r "benchmark" docs/quantization/
grep -r "QNNPACK" docs/quantization/
```

### View Specific Sections
```bash
# Installation commands
cat docs/quantization/README.md | grep -A 10 "Quick Start"

# Performance expectations
cat docs/quantization/QUANTIZATION_GUIDE.md | grep -A 20 "Performance Gains"

# Troubleshooting
cat docs/quantization/README.md | grep -A 30 "Troubleshooting"
```

## 🎯 Deployment Workflow with Sparse Checkout

### On Development Machine

```bash
# 1. Quantize model
cd ~/projects/Erewhon
python src/policies/ACT/quantize_act_model.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --mode static \
    --output outputs/lerobot_act/best_model_static_quantized.pth

# 2. Deploy to Pi (script handles sparse checkout)
./deploy_to_pi.sh \
    outputs/lerobot_act/best_model_static_quantized.pth \
    mboels@raspberrypi
```

### On Raspberry Pi

```bash
# 1. Update repo (gets latest docs too!)
cd ~/projects/Erewhon
git pull origin main

# 2. Check for new docs
ls -la src/robots/rover/docs/quantization/

# 3. Read updated guides
cat src/robots/rover/docs/quantization/README.md

# 4. Run inference with reference to docs
cd src/robots/rover
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_static_quantized.pth \
    --benchmark
```

## ⚠️ Important Notes

### What's NOT in Sparse Checkout

The following are NOT available on Pi (and that's okay!):

- ❌ Training scripts (`src/policies/ACT/official_lerobot_trainer.py`)
- ❌ Original training data (`outputs/lerobot_act/`)
- ❌ LeRobot source code (`src/policies/ACT/lerobot/`)
- ❌ Other robot configurations (`src/robots/drone/`)

These files stay on your development machine where they belong.

### What IS in Sparse Checkout

The Pi has everything needed for inference:

- ✅ Inference scripts (`src/inference/`)
- ✅ Deployed models (`models/`)
- ✅ Documentation (`docs/`, including quantization guides)
- ✅ Requirements (`requirements_pi.txt`)
- ✅ Calibration episodes (if needed)

### Disk Space Savings

| What | Full Clone | Sparse Checkout | Savings |
|------|-----------|-----------------|---------|
| Repository | ~2.0 GB | ~50 MB | **97.5%** |
| Models | Included | Deployed separately | - |
| Training data | Included | Not needed | - |
| **Total on Pi** | **~2.0 GB** | **~150 MB** | **92.5%** |

## 🐛 Troubleshooting

### Docs Not Showing Up

```bash
# Check sparse checkout is enabled
git sparse-checkout list

# Re-init if needed
git sparse-checkout init --cone
git sparse-checkout set src/robots/rover
git pull origin main
```

### Want to See All Files Temporarily

```bash
# Disable sparse checkout
git sparse-checkout disable

# Work with full repo...

# Re-enable when done
git sparse-checkout init --cone
git sparse-checkout set src/robots/rover
```

### Accidentally Modified Files Outside Sparse Checkout

```bash
# Reset to clean state
git reset --hard HEAD
git clean -fd

# Re-apply sparse checkout
git sparse-checkout set src/robots/rover
```

---

**Last Updated:** October 11, 2025  
**Sparse Checkout Path:** `src/robots/rover`  
**Includes:** Code, docs, models, inference scripts  
**Excludes:** Training scripts, full dataset, other robots
