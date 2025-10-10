#!/bin/bash

# Deploy Quantized ACT Model to Raspberry Pi
# Works with sparse checkout (src/robots/rover only)
#
# Usage:
#   ./deploy_to_pi.sh [checkpoint] [ssh_host]
#
# Example:
#   ./deploy_to_pi.sh outputs/lerobot_act/best_model.pth mboels@raspberrypi

set -e

# Configuration
CHECKPOINT="${1:-outputs/lerobot_act/best_model.pth}"
RPI_HOST="${2:-mboels@raspberrypi}"
RPI_DEPLOY_DIR="src/robots/rover/models"  # Within sparse checkout

echo "=" | tr '=' '=' | head -c 80; echo
echo "🚀 DEPLOY QUANTIZED MODEL TO RASPBERRY PI"
echo "=" | tr '=' '=' | head -c 80; echo
echo "Checkpoint:  $CHECKPOINT"
echo "Pi Host:     $RPI_HOST"
echo "Pi Path:     ~/$RPI_DEPLOY_DIR/"
echo "=" | tr '=' '=' | head -c 80; echo
echo ""

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Error: Checkpoint not found: $CHECKPOINT"
    echo ""
    echo "Please provide a valid checkpoint path or quantize first:"
    echo "  python src/policies/ACT/quantize_act_model.py --checkpoint best_model.pth --mode static --output quantized_model.pth"
    exit 1
fi

# Check if already quantized by inspecting checkpoint metadata
CHECKPOINT_SIZE=$(stat -f%z "$CHECKPOINT" 2>/dev/null || stat -c%s "$CHECKPOINT" 2>/dev/null)
CHECKPOINT_SIZE_MB=$((CHECKPOINT_SIZE / 1024 / 1024))

echo "📦 Model Info:"
echo "   Size: ${CHECKPOINT_SIZE_MB} MB"

# Check if checkpoint has quantization metadata
IS_QUANTIZED=$(python3 -c "import torch; ckpt = torch.load('"$CHECKPOINT"', map_location='cpu', weights_only=False); print(ckpt.get('quantization_mode', 'none'))" 2>/dev/null || echo "error")

if [ "$IS_QUANTIZED" = "static" ] || [ "$IS_QUANTIZED" = "dynamic" ] || [ "$IS_QUANTIZED" = "mixed" ]; then
    echo "   ✅ Model is quantized (mode: $IS_QUANTIZED)"
elif [ "$IS_QUANTIZED" = "none" ] && [ "$CHECKPOINT_SIZE_MB" -gt 200 ]; then
    echo "   ⚠️  Model appears to be FP32 (not quantized)"
    echo ""
    read -p "Do you want to quantize it first? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "🔧 Quantizing model with static quantization..."
        python src/policies/ACT/quantize_act_model.py \
            --checkpoint "$CHECKPOINT" \
            --mode static \
            --calibration_data src/robots/rover/episodes \
            --num_calibration_batches 200 \
            --output "${CHECKPOINT%.pth}_quantized.pth"
        
        CHECKPOINT="${CHECKPOINT%.pth}_quantized.pth"
        echo "✅ Quantization complete: $CHECKPOINT"
        echo ""
    fi
else
    echo "   ℹ️  Proceeding with deployment..."
fi

echo ""

# Check Pi connectivity
echo "🔍 Checking Raspberry Pi connectivity..."
if ! ping -c 1 -W 2 ${RPI_HOST#*@} >/dev/null 2>&1; then
    echo "⚠️  Warning: Cannot ping Pi, but will try SSH anyway"
fi

# Create deployment directory on Pi
echo "📁 Creating deployment directory on Pi..."
ssh "$RPI_HOST" "mkdir -p $RPI_DEPLOY_DIR" || {
    echo "❌ Failed to create directory on Pi"
    echo "   Make sure you can SSH: ssh $RPI_HOST"
    exit 1
}

echo "✅ Directory ready"
echo ""

# Transfer model
echo "📤 Transferring model to Pi..."
MODEL_NAME=$(basename "$CHECKPOINT")
scp "$CHECKPOINT" "$RPI_HOST:~/$RPI_DEPLOY_DIR/$MODEL_NAME" || {
    echo "❌ Transfer failed"
    exit 1
}

echo "✅ Model transferred successfully"
echo ""

# Verify transfer
echo "🔍 Verifying transfer..."
REMOTE_SIZE=$(ssh "$RPI_HOST" "stat -c%s ~/$RPI_DEPLOY_DIR/$MODEL_NAME 2>/dev/null || stat -f%z ~/$RPI_DEPLOY_DIR/$MODEL_NAME 2>/dev/null")
REMOTE_SIZE_MB=$((REMOTE_SIZE / 1024 / 1024))

if [ "$REMOTE_SIZE_MB" -eq "$CHECKPOINT_SIZE_MB" ]; then
    echo "✅ Transfer verified (${REMOTE_SIZE_MB} MB)"
else
    echo "⚠️  Size mismatch: Local ${CHECKPOINT_SIZE_MB}MB, Remote ${REMOTE_SIZE_MB}MB"
fi

echo ""

# Check dependencies on Pi
echo "🔍 Checking Python dependencies on Pi..."
ssh "$RPI_HOST" "python3 -c 'import torch, cv2, numpy' 2>/dev/null" && {
    echo "✅ Dependencies installed"
} || {
    echo "⚠️  Some dependencies missing"
    echo ""
    echo "Install on Pi with:"
    echo "  ssh $RPI_HOST"
    echo "  pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    echo "  pip3 install opencv-python numpy"
}

echo ""

# Summary
echo "=" | tr '=' '=' | head -c 80; echo
echo "✅ DEPLOYMENT COMPLETE!"
echo "=" | tr '=' '=' | head -c 80; echo
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. SSH to Raspberry Pi:"
echo "   ssh $RPI_HOST"
echo ""
echo "2. Benchmark the model:"
echo "   cd ~/$RPI_DEPLOY_DIR/.."
echo "   python3 src/inference/act_inference_quantized.py \\"
echo "       --checkpoint models/$MODEL_NAME \\"
echo "       --benchmark"
echo ""
echo "3. Test with camera:"
echo "   python3 src/inference/act_inference_quantized.py \\"
echo "       --checkpoint models/$MODEL_NAME \\"
echo "       --camera_id 0"
echo ""
echo "4. Deploy to RC car:"
echo "   python3 src/inference/act_inference_quantized.py \\"
echo "       --checkpoint models/$MODEL_NAME \\"
echo "       --camera_id 0 \\"
echo "       --arduino_port /dev/ttyUSB0 \\"
echo "       --control_freq 30"
echo ""
echo "=" | tr '=' '=' | head -c 80; echo
echo ""
echo "📁 Model location on Pi: ~/$RPI_DEPLOY_DIR/$MODEL_NAME"
echo ""
