#!/bin/bash

# Re-quantize and Deploy Model to Raspberry Pi
# This script fixes the quantization format and deploys to Pi

set -e

echo "=================================================="
echo "🔧 RE-QUANTIZE AND DEPLOY TO RASPBERRY PI"
echo "=================================================="
echo ""

# Configuration
ORIGINAL_MODEL="outputs/lerobot_act/lerobot_act_20251009_101545/best_model.pth"
QUANTIZED_MODEL="outputs/lerobot_act/best_model_quantized.pth"
CALIBRATION_DATA="src/robots/rover/episodes"
RPI_HOST="${1:-mboels@raspberrypi}"

echo "📋 Configuration:"
echo "   Original model: $ORIGINAL_MODEL"
echo "   Output: $QUANTIZED_MODEL"
echo "   Mode: dynamic (more reliable)"
echo "   Pi host: $RPI_HOST"
echo ""

# Step 1: Re-quantize with fixed format
echo "=================================================="
echo "Step 1: Re-quantizing model with fixed format"
echo "=================================================="
echo ""

python src/policies/ACT/quantize_act_model.py \
    --checkpoint "$ORIGINAL_MODEL" \
    --mode dynamic \
    --output "$QUANTIZED_MODEL"

echo ""
echo "✅ Quantization complete!"
echo ""

# Step 2: Verify the new format
echo "=================================================="
echo "Step 2: Verifying new quantized model format"
echo "=================================================="
echo ""

python3 -c "
import torch
ckpt = torch.load('$QUANTIZED_MODEL', map_location='cpu', weights_only=False)
print(f'✅ Checkpoint keys: {list(ckpt.keys())}')
print(f'✅ Quantization mode: {ckpt.get(\"quantization_mode\", \"not found\")}')
print(f'✅ Has full model: {\"model\" in ckpt}')
print(f'✅ Has state_dict: {\"model_state_dict\" in ckpt}')
print(f'✅ Model type: {type(ckpt.get(\"model\", \"N/A\"))}')
"

echo ""

# Step 3: Transfer to Pi
echo "=================================================="
echo "Step 3: Transferring to Raspberry Pi"
echo "=================================================="
echo ""

# Create models directory on Pi
echo "Creating models directory on Pi..."
ssh "$RPI_HOST" "mkdir -p ~/src/robots/rover/models"

# Transfer model
echo "Transferring quantized model..."
scp "$QUANTIZED_MODEL" "$RPI_HOST":~/src/robots/rover/models/best_model_quantized.pth

echo ""
echo "✅ Transfer complete!"
echo ""

# Step 4: Verify on Pi
echo "=================================================="
echo "Step 4: Verifying deployment on Pi"
echo "=================================================="
echo ""

ssh "$RPI_HOST" << 'EOF'
echo "Checking quantized model on Pi..."
ls -lh ~/src/robots/rover/models/best_model_quantized.pth

echo ""
echo "Verifying model format..."
python3 << 'PYEOF'
import torch
import sys

try:
    ckpt = torch.load('/home/mboels/src/robots/rover/models/best_model_quantized.pth', map_location='cpu', weights_only=False)
    print(f"✅ Model loaded successfully!")
    print(f"✅ Quantization mode: {ckpt.get('quantization_mode', 'not found')}")
    print(f"✅ Has full model: {'model' in ckpt}")
    print(f"✅ Has state_dict: {'model_state_dict' in ckpt}")
    
    if 'model' in ckpt:
        model = ckpt['model']
        print(f"✅ Model type: {type(model).__name__}")
        print(f"✅ Model is in eval mode: {not model.training}")
    else:
        print("⚠️  WARNING: Model object not found in checkpoint!")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)
PYEOF

EOF

echo ""
echo "=================================================="
echo "✅ RE-QUANTIZATION AND DEPLOYMENT COMPLETE!"
echo "=================================================="
echo ""
echo "📋 Next Steps on Raspberry Pi:"
echo ""
echo "1. SSH to Pi:"
echo "   ssh $RPI_HOST"
echo ""
echo "2. Run benchmark:"
echo "   cd ~/src/robots/rover"
echo "   python3 src/inference/act_inference_quantized.py \\"
echo "       --checkpoint models/best_model_quantized.pth \\"
echo "       --benchmark \\"
echo "       --iterations 1000"
echo ""
echo "3. Expected results (dynamic quantization):"
echo "   - Mean latency: ~50-60ms"
echo "   - P95 latency: ~70-80ms"
echo "   - Control rate: 15-20 Hz"
echo "   - Model size: ~170 MB (6x compression)"
echo ""
echo "=================================================="
