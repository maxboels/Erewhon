# Raspberry Pi Deployment - Quick Commands

## 🚀 Install Dependencies (Copy & Paste)

```bash
# SSH to Pi
ssh mboels@raspberrypi

# Navigate to rover directory
cd ~/src/robots/rover

# Install all dependencies with correct versions (RECOMMENDED)
pip3 install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy==2.2.6 opencv-python==4.12.0.88 pyserial==3.5

# Verify installation
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); import cv2; print(f'OpenCV {cv2.__version__}'); import numpy; print(f'NumPy {numpy.__version__}')"
```

## 🧪 Benchmark Model

```bash
cd ~/src/robots/rover

python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_static_quantized.pth \
    --benchmark \
    --iterations 1000
```

## 📹 Test with Camera

```bash
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_static_quantized.pth \
    --camera_id 0
```

## 🚗 Deploy to RC Car

```bash
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_static_quantized.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0 \
    --control_freq 30
```

---

## ⚠️ Why These Specific Versions?

### Your Development Environment (conda lerobot):
- **PyTorch**: 2.8.0
- **TorchVision**: 0.23.0  
- **NumPy**: 2.2.6
- **OpenCV**: 4.12.0.88
- **PySerial**: 3.5

### Critical Reasons:

1. **Quantization Compatibility**: The model was quantized with PyTorch 2.8.0
   - Different versions have different quantization APIs
   - Loading will fail or produce incorrect results with version mismatch

2. **State Dict Format**: Model weights saved in PyTorch 2.8.0 format
   - Older versions may not support new layer types
   - Newer versions may have breaking changes

3. **QNNPACK Backend**: ARM optimizations are version-specific
   - Quantized operations behavior changes between versions
   - Performance characteristics differ

4. **NumPy Compatibility**: NumPy 2.x has breaking changes from 1.x
   - Your model expects NumPy 2.2.6 array behaviors

### What Happens with Wrong Versions?

❌ **Too Old (e.g., PyTorch 2.0):**
- Cannot load quantized model (missing APIs)
- QNNPACK backend not available or incompatible
- Model loading crashes

❌ **Too New (e.g., PyTorch 2.9+):**
- Potential breaking changes in quantization
- Different default behaviors
- Untested compatibility

❌ **Different NumPy (e.g., 1.x):**
- Array operations may behave differently
- Type conversions may fail
- Subtle bugs in preprocessing

✅ **Exact Match (PyTorch 2.8.0, NumPy 2.2.6):**
- Guaranteed compatibility
- Known performance characteristics
- Reproducible results

---

## 🔍 Verification Commands

### Check Installed Versions:
```bash
python3 -c "import torch; print('PyTorch:', torch.__version__)"
python3 -c "import torchvision; print('TorchVision:', torchvision.__version__)"
python3 -c "import numpy; print('NumPy:', numpy.__version__)"
python3 -c "import cv2; print('OpenCV:', cv2.__version__)"
python3 -c "import serial; print('PySerial:', serial.VERSION)"
```

### Expected Output:
```
PyTorch: 2.8.0
TorchVision: 0.23.0
NumPy: 2.2.6
OpenCV: 4.12.0.88
PySerial: 3.5
```

### Check QNNPACK Backend:
```bash
python3 -c "import torch.backends.quantized; print('Backend:', torch.backends.quantized.engine)"
```

Expected: `qnnpack`

---

## 📋 Installation Troubleshooting

### Issue: PyTorch 2.8.0 not available for ARM

If you get "No matching distribution found":

```bash
# Check available PyTorch versions for CPU
pip3 index versions torch --index-url https://download.pytorch.org/whl/cpu

# If 2.8.0 is not available, use the closest compatible version
# (Check PyTorch version compatibility matrix)
```

### Issue: Installation hangs on Raspberry Pi

```bash
# Use fewer workers to reduce memory usage
pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# Or install one package at a time
pip3 install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip3 install torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
```

### Issue: Out of memory during installation

```bash
# Increase swap space temporarily
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile  # Set CONF_SWAPSIZE=2048
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

---

## ✅ Ready to Test!

Once dependencies are installed with the correct versions, you're ready to benchmark and deploy! 🚀
