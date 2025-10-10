#!/bin/bash

# ACT Model Quantization and Deployment Script
# Complete pipeline from trained model to Raspberry Pi deployment

set -e  # Exit on error

# Configuration
CHECKPOINT="${1:-outputs/lerobot_act/best_model.pth}"
DATA_DIR="${2:-src/robots/rover/episodes}"
OUTPUT_DIR="outputs/lerobot_act/quantized"
RPI_HOST="${3:-mboels@raspberrypi}"  # Change to match your SSH setup

echo "=" | tr '=' '=' | head -c 80; echo
echo "🔧 ACT MODEL QUANTIZATION & DEPLOYMENT PIPELINE"
echo "=" | tr '=' '=' | head -c 80; echo
echo "Checkpoint:  $CHECKPOINT"
echo "Data Dir:    $DATA_DIR"
echo "Output Dir:  $OUTPUT_DIR"
echo "Pi Host:     $RPI_HOST"
echo "=" | tr '=' '=' | head -c 80; echo
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Step 1: Dynamic Quantization (fastest, for initial testing)
echo "📦 Step 1: Creating dynamic quantized model..."
echo "─" | tr '─' '─' | head -c 80; echo

python src/policies/ACT/quantize_act_model.py \
    --checkpoint "$CHECKPOINT" \
    --mode dynamic \
    --output "$OUTPUT_DIR/model_dynamic_quant.pth" \
    --benchmark

echo ""
echo "✅ Dynamic quantization complete!"
echo ""

# Step 2: Static Quantization (best accuracy, for production)
echo "📦 Step 2: Creating static quantized model (with calibration)..."
echo "─" | tr '─' '─' | head -c 80; echo

python src/policies/ACT/quantize_act_model.py \
    --checkpoint "$CHECKPOINT" \
    --mode static \
    --calibration_data "$DATA_DIR" \
    --num_calibration_batches 200 \
    --output "$OUTPUT_DIR/model_static_quant.pth" \
    --benchmark

echo ""
echo "✅ Static quantization complete!"
echo ""

# Step 3: Accuracy Comparison
echo "📊 Step 3: Comparing accuracy..."
echo "─" | tr '─' '─' | head -c 80; echo

python src/policies/ACT/quantize_act_model.py \
    --checkpoint "$CHECKPOINT" \
    --mode static \
    --calibration_data "$DATA_DIR" \
    --num_calibration_batches 200 \
    --output "$OUTPUT_DIR/model_final.pth" \
    --compare \
    --test_data "$DATA_DIR"

echo ""
echo "✅ Accuracy validation complete!"
echo ""

# Step 4: Test Inference
echo "🧪 Step 4: Testing quantized inference..."
echo "─" | tr '─' '─' | head -c 80; echo

# Find a test image
TEST_IMAGE=$(find "$DATA_DIR" -name "frame_0000.jpg" -type f | head -n 1)

if [ -n "$TEST_IMAGE" ]; then
    echo "Using test image: $TEST_IMAGE"
    python src/policies/ACT/lerobot_act_inference_quantized.py \
        --checkpoint "$OUTPUT_DIR/model_final.pth" \
        --test_image "$TEST_IMAGE"
else
    echo "⚠️  No test image found, skipping inference test"
fi

echo ""
echo "✅ Inference test complete!"
echo ""

# Step 5: Create deployment package
echo "📦 Step 5: Creating deployment package..."
echo "─" | tr '─' '─' | head -c 80; echo

DEPLOY_DIR="$OUTPUT_DIR/deploy"
mkdir -p "$DEPLOY_DIR"

# Copy necessary files
cp "$OUTPUT_DIR/model_final.pth" "$DEPLOY_DIR/model.pth"
cp src/policies/ACT/lerobot_act_inference_rpi5.py "$DEPLOY_DIR/"
cp src/policies/ACT/lerobot_act_inference_quantized.py "$DEPLOY_DIR/"

# Copy LeRobot source (needed for model)
cp -r src/policies/ACT/lerobot "$DEPLOY_DIR/"

# Create deployment README
cat > "$DEPLOY_DIR/README.md" << 'EOF'
# ACT Model Deployment Package

## Files Included

- `model.pth` - Quantized INT8 model (optimized for Pi 5)
- `lerobot_act_inference_rpi5.py` - Optimized inference script
- `lerobot_act_inference_quantized.py` - General quantized inference
- `lerobot/` - LeRobot source code (required)

## Deployment to Raspberry Pi 5

### 1. Transfer Files

```bash
# From your development machine
scp -r deploy/ pi@raspberrypi:~/act_model/
```

### 2. Install Dependencies (on Pi)

```bash
ssh pi@raspberrypi

# Install PyTorch for ARM
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip3 install numpy opencv-python pillow pyserial tqdm
```

### 3. Benchmark Performance

```bash
cd ~/act_model

python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --benchmark \
    --iterations 1000
```

**Expected Results:**
- Mean latency: 20-40ms
- P95 latency: <50ms
- Throughput: 25-50 FPS

### 4. Test with Camera

```bash
python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --camera_id 0 \
    --control_freq 30
```

Press 'q' to quit.

### 5. Deploy to RC Car

```bash
python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0 \
    --control_freq 30
```

## Troubleshooting

### Slow Inference (>100ms)

1. Check QNNPACK is enabled:
   ```python
   import torch.backends.quantized
   print(torch.backends.quantized.engine)  # Should be 'qnnpack'
   ```

2. Check thread count:
   ```python
   import torch
   print(torch.get_num_threads())  # Should be 4
   ```

3. Verify model is quantized:
   ```bash
   ls -lh model.pth  # Should be ~14MB (not ~54MB)
   ```

### Camera Not Opening

```bash
# List available cameras
ls /dev/video*

# Test with v4l2
v4l2-ctl --list-devices
```

### Arduino Not Responding

```bash
# Check port
ls /dev/ttyUSB* /dev/ttyACM*

# Test connection
python3 -c "import serial; s=serial.Serial('/dev/ttyUSB0', 115200); print('OK')"
```

## Performance Tips

- Run headless (--no_display) for better performance
- Reduce image resolution if needed
- Use action chunking for smoother control
- Monitor CPU temperature: `vcgencmd measure_temp`

EOF

echo "✅ Deployment package created in: $DEPLOY_DIR"
echo ""

# Step 6: Transfer to Pi (if host provided)
if ping -c 1 -W 1 raspberrypi >/dev/null 2>&1; then
    echo "📡 Step 6: Transferring to Raspberry Pi..."
    echo "─" | tr '─' '─' | head -c 80; echo
    
    echo "Transferring deployment package to $RPI_HOST..."
    scp -r "$DEPLOY_DIR" "$RPI_HOST:~/act_model/" || {
        echo "⚠️  Transfer failed. You can manually copy files:"
        echo "   scp -r $DEPLOY_DIR $RPI_HOST:~/act_model/"
    }
    
    echo ""
    echo "✅ Transfer complete!"
else
    echo "⚠️  Step 6: Skipping transfer (Pi not reachable)"
    echo "   Manually transfer with: scp -r $DEPLOY_DIR $RPI_HOST:~/act_model/"
fi

# Summary
echo ""
echo "=" | tr '=' '=' | head -c 80; echo
echo "✅ QUANTIZATION & DEPLOYMENT PIPELINE COMPLETE!"
echo "=" | tr '=' '=' | head -c 80; echo
echo ""
echo "📊 Summary:"
echo "  ✅ Dynamic quantized model:  $OUTPUT_DIR/model_dynamic_quant.pth"
echo "  ✅ Static quantized model:   $OUTPUT_DIR/model_static_quant.pth"
echo "  ✅ Final production model:   $OUTPUT_DIR/model_final.pth"
echo "  ✅ Deployment package:       $DEPLOY_DIR/"
echo ""
echo "📋 Next Steps:"
echo ""
echo "1. SSH to Raspberry Pi:"
echo "   ssh $RPI_HOST"
echo ""
echo "2. Install dependencies:"
echo "   pip3 install torch torchvision numpy opencv-python pillow pyserial tqdm"
echo ""
echo "3. Benchmark on Pi:"
echo "   cd ~/act_model"
echo "   python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --benchmark"
echo ""
echo "4. Deploy to RC car:"
echo "   python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --camera_id 0 --arduino_port /dev/ttyUSB0"
echo ""
echo "=" | tr '=' '=' | head -c 80; echo
echo ""
echo "📚 Documentation: src/policies/ACT/QUANTIZATION_GUIDE.md"
echo ""
