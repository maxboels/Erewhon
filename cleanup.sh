#!/bin/bash

# 🧹 Cleanup Script - Remove Testing/Debug Files
# Date: October 8, 2025
# This script removes ~35 test/debug/redundant files while keeping core functionality

set -e  # Exit on error

echo "🧹 Starting cleanup process..."
echo ""

# Create archive directory (optional - if you want to keep files)
# mkdir -p archive/removed_files_$(date +%Y%m%d)

echo "📂 Current directory: $(pwd)"
echo ""

# ============================================================
# 1. Remove Test Scripts
# ============================================================
echo "🗑️  Step 1: Removing test scripts..."

rm -f test_dataloader.py
echo "  ✅ Removed: test_dataloader.py"

cd src/policies/ACT

rm -f test_official_act.py
echo "  ✅ Removed: test_official_act.py"

rm -f test_inference.py
echo "  ✅ Removed: test_inference.py"

rm -f test_tracer_integration.py
echo "  ✅ Removed: test_tracer_integration.py"

echo ""

# ============================================================
# 2. Remove Old/Deprecated Implementations
# ============================================================
echo "🗑️  Step 2: Removing deprecated implementations..."

# Already in deprecated folder, but let's remove redundant files
rm -f hybrid_lerobot_trainer.py
echo "  ✅ Removed: hybrid_lerobot_trainer.py"

rm -f official_act_trainer.py
echo "  ✅ Removed: official_act_trainer.py"

rm -f train_local_act.py
echo "  ✅ Removed: train_local_act.py"

rm -f tracer_pipeline.py
echo "  ✅ Removed: tracer_pipeline.py"

rm -f inference_act.py
echo "  ✅ Removed: inference_act.py"

rm -f state_aware_inference.py
echo "  ✅ Removed: state_aware_inference.py"

rm -f setup_lerobot.py
echo "  ✅ Removed: setup_lerobot.py (one-time setup, no longer needed)"

echo ""

# ============================================================
# 3. Remove Redundant Documentation
# ============================================================
echo "🗑️  Step 3: Removing redundant documentation..."

rm -f ACTION_PLAN_FULL_ACT.md
rm -f ACTION_REQUIRED.md
rm -f ANSWER_WHY_SIMPLIFIED.md
rm -f DONE.md
rm -f ENHANCED_TRAINING_SUMMARY.md
rm -f FINAL_SUMMARY.md
rm -f IMPLEMENTATION_COMPLETE.md
rm -f LEROBOT_VS_CUSTOM_STATE_HANDLING.md
rm -f MIGRATION_SUMMARY.md
rm -f QUICK_COMMANDS.md
rm -f README_LEROBOT_ACT.md
rm -f README_tracer.md
rm -f README_training.md
rm -f RESOLUTION_CONFIG.md
rm -f RESOLUTION_CORRECTION.md
rm -f STATE_AWARE_IMPLEMENTATION.md
rm -f STATE_AWARE_TRAINING_GUIDE.md
rm -f STATE_HANDLING_RESOLVED.md
rm -f STATE_INPUT_ANALYSIS.md
rm -f TRAINING_SUMMARY.md
rm -f VISUAL_GUIDE.md
rm -f WHY_USE_FULL_LEROBOT_ACT.md

echo "  ✅ Removed: 22 redundant markdown files"

echo ""

# ============================================================
# 4. Remove Analysis Scripts
# ============================================================
echo "🗑️  Step 4: Removing analysis scripts..."

cd ../../datasets/analysis

rm -f analyze_dataset.py
rm -f quick_dataset_analysis.py
rm -f simple_analysis.py
rm -f plot_training_signals.py

echo "  ✅ Removed: 4 analysis scripts"

# Remove empty analysis directory
cd ..
rmdir analysis 2>/dev/null && echo "  ✅ Removed empty analysis/ directory" || echo "  ⚠️  analysis/ directory not empty or doesn't exist"

# Remove debug markdown
rm -f DATALOADER_ANALYSIS.md
echo "  ✅ Removed: DATALOADER_ANALYSIS.md"

echo ""

# ============================================================
# 5. Summary
# ============================================================
cd ../../..  # Back to root

echo "✅ Cleanup complete!"
echo ""
echo "📊 Summary:"
echo "  - Removed ~35 test/debug/redundant files"
echo "  - Kept core functionality:"
echo "    • official_lerobot_trainer.py (training)"
echo "    • lerobot_act_inference.py (inference)"
echo "    • local_dataset_loader.py (dataloader)"
echo "    • Essential docs (README.md, QUICKSTART, guides)"
echo "    • Config files"
echo "    • Network files"
echo "    • Full lerobot package"
echo ""
echo "📂 Final structure:"
echo "  src/datasets/ - Dataloader + docs"
echo "  src/policies/ACT/ - Training + inference + docs"
echo ""
echo "🎉 Codebase is now clean and focused!"
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Commit: git add -A && git commit -m 'Clean up: Remove test/debug files'"
echo "  3. Train model: python src/policies/ACT/official_lerobot_trainer.py"
