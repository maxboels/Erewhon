#!/bin/bash

# Update Raspberry Pi with Latest Documentation
# Run this on the Raspberry Pi to get the latest quantization docs
# Since src/robots/rover is already in sparse checkout, a simple git pull will get everything!

set -e

echo "=================================================="
echo "🔄 UPDATING RASPBERRY PI REPOSITORY"
echo "=================================================="
echo ""

# Check if we're in the right directory
if [ ! -d ".git" ]; then
    echo "❌ Error: Not in a git repository"
    echo "Please run this from ~/projects/Erewhon/ or wherever you cloned the repo"
    exit 1
fi

echo "📥 Pulling latest changes from main branch..."
echo "(Since src/robots/rover is in sparse checkout, docs will update automatically)"
git pull origin main

echo ""
echo "✅ Update complete!"
echo ""

# Show what's new
echo "=================================================="
echo "📚 AVAILABLE DOCUMENTATION"
echo "=================================================="
echo ""

if [ -d "src/robots/rover/docs/quantization" ]; then
    echo "Quantization Guides:"
    ls -1 src/robots/rover/docs/quantization/*.md | while read file; do
        basename "$file"
    done | sed 's/^/  ✓ /'
    echo ""
fi

echo "Deployment Guides:"
[ -f "src/robots/rover/PI_DEPLOYMENT_COMMANDS.md" ] && echo "  ✓ PI_DEPLOYMENT_COMMANDS.md"
[ -f "src/robots/rover/QUANTIZATION_DEPLOYMENT_SUMMARY.md" ] && echo "  ✓ QUANTIZATION_DEPLOYMENT_SUMMARY.md"
[ -f "src/robots/rover/SPARSE_CHECKOUT_SETUP.md" ] && echo "  ✓ SPARSE_CHECKOUT_SETUP.md"
[ -f "src/robots/rover/README.md" ] && echo "  ✓ README.md"

echo ""
echo "=================================================="
echo "🎯 QUICK START"
echo "=================================================="
echo ""
echo "View quantization documentation:"
echo "  cd src/robots/rover/docs/quantization"
echo "  ls -lh"
echo "  cat README.md"
echo ""
echo "View deployment commands:"
echo "  cat src/robots/rover/PI_DEPLOYMENT_COMMANDS.md"
echo ""
echo "Run inference benchmark:"
echo "  cd src/robots/rover"
echo "  python3 src/inference/act_inference_quantized.py \\"
echo "      --checkpoint models/best_model_static_quantized.pth \\"
echo "      --benchmark"
echo ""
echo "=================================================="
