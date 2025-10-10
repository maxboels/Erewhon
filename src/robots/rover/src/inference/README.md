# ACT Model Inference for Raspberry Pi 5

**Quantized ACT model deployment for autonomous RC car control**

---

## 🎯 Overview

This directory contains the standalone inference script for running quantized ACT models on Raspberry Pi 5. It's designed to work with the **sparse checkout** setup where only `src/robots/rover` is pulled to the Pi.

## 📁 Directory Structure

```
src/robots/rover/
├── src/
│   └── inference/
│       └── act_inference_quantized.py  # Standalone inference script
├── models/                              # Deployed models (created on first deploy)
│   └── *.pth                           # Quantized model checkpoints
├── episodes/                            # Training data
└── README.md
```

---

## 🚀 Quick Deployment

### From Your Laptop (Development Machine):

#### 1. Quantize Your Model

```bash
# Navigate to project root
cd /home/maxboels/projects/Erewhon

# Quantize the trained model
python src/policies/ACT/quantize_act_model.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --mode static \
    --calibration_data src/robots/rover/episodes \
    --num_calibration_batches 200 \
    --output outputs/lerobot_act/best_model_quantized.pth \
    --benchmark
```

**Expected output:**
```
✅ Quantized model saved to outputs/lerobot_act/best_model_quantized.pth
   Original size: 54.32 MB
   Quantized size: 13.81 MB
   Compression: 3.93x
```

#### 2. Deploy to Raspberry Pi

```bash
# Deploy quantized model to Pi
./deploy_to_pi.sh \
    outputs/lerobot_act/best_model_quantized.pth \
    mboels@raspberrypi
```

**This will:**
- ✅ Transfer model to `~/src/robots/rover/models/` on Pi
- ✅ Verify file transfer
- ✅ Check dependencies
- ✅ Provide next steps

---

### On Raspberry Pi:

#### 3. Install Dependencies (One-Time)

```bash
# PyTorch for ARM (CPU)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Other dependencies
pip3 install opencv-python numpy pyserial
```

#### 4. Benchmark Performance

```bash
cd ~/src/robots/rover

python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_quantized.pth \
    --benchmark \
    --iterations 1000
```

**Expected Results:**
```
📊 RASPBERRY PI 5 INFERENCE BENCHMARK
Mean latency:  35.23 ± 2.15 ms
P95 latency:   42.67 ms
Throughput:    28.4 FPS

🎮 CONTROL FREQUENCY CAPABILITY
✅ Can achieve 30Hz control loop
   Recommended: 23 Hz
```

#### 5. Test with Camera

```bash
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_quantized.pth \
    --camera_id 0 \
    --control_freq 30
```

**Press 'q' to quit**

#### 6. Deploy to RC Car

```bash
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_quantized.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0 \
    --control_freq 30
```

---

## 🔧 Troubleshooting

### Issue: Permission denied for camera

```bash
sudo usermod -a -G video $USER
# Log out and back in
```

### Issue: Permission denied for Arduino

```bash
sudo usermod -a -G dialout $USER
# Log out and back in
```

### Issue: Camera not found

```bash
# List cameras
ls /dev/video*

# Try different ID
python3 src/inference/act_inference_quantized.py --checkpoint models/*.pth --camera_id 1
```

### Issue: Arduino not responding

```bash
# Check port
ls /dev/ttyUSB* /dev/ttyACM*

# Try different port
python3 src/inference/act_inference_quantized.py --checkpoint models/*.pth --arduino_port /dev/ttyACM0
```

### Issue: Slow inference (>100ms)

```bash
# Check model size (should be ~14MB)
ls -lh models/*.pth

# Check QNNPACK backend
python3 -c "import torch.backends.quantized; print(torch.backends.quantized.engine)"
# Should print: qnnpack

# Check CPU temperature (throttles if >80°C)
vcgencmd measure_temp
```

---

## 📊 Performance Expectations

### On Raspberry Pi 5:

| Metric | Expected Value |
|--------|----------------|
| **Mean Latency** | 30-40ms |
| **P95 Latency** | <50ms |
| **Throughput** | 25-35 FPS |
| **Control Rate** | 22-30 Hz |
| **Model Size** | ~14 MB |
| **Power Draw** | ~11W |

### vs Original FP32:

| | Original | Quantized |
|---|----------|-----------|
| **Latency** | ~600ms ❌ | ~40ms ✅ |
| **Size** | 54 MB | 14 MB |
| **Usable** | No | Yes! |

---

## 🗂️ Model Management

### List deployed models:

```bash
ls -lh ~/src/robots/rover/models/
```

### Deploy new model:

From laptop:
```bash
./deploy_to_pi.sh new_model_quantized.pth mboels@raspberrypi
```

### Test different models:

```bash
# Benchmark both
python3 src/inference/act_inference_quantized.py --checkpoint models/model_v1.pth --benchmark
python3 src/inference/act_inference_quantized.py --checkpoint models/model_v2.pth --benchmark

# Use best performing one
python3 src/inference/act_inference_quantized.py --checkpoint models/model_v2.pth --camera_id 0
```

---

## 🚀 Advanced Usage

### Headless Mode (No Display)

```bash
# Good for remote SSH sessions
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/*.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0 \
    --no_display
```

### Custom Control Frequency

```bash
# Lower frequency for stability
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/*.pth \
    --camera_id 0 \
    --control_freq 20

# Higher frequency (if latency allows)
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/*.pth \
    --camera_id 0 \
    --control_freq 40
```

### Monitor System Resources

```bash
# CPU usage
htop

# Temperature (should be <80°C)
watch -n 1 vcgencmd measure_temp

# Check for throttling
vcgencmd get_throttled
# 0x0 = OK
```

---

## 📚 Workflow Summary

### Development Machine:
1. Train ACT model → `best_model.pth`
2. Quantize → `best_model_quantized.pth`
3. Deploy → `./deploy_to_pi.sh`

### Raspberry Pi:
1. Receive model in `~/src/robots/rover/models/`
2. Benchmark performance
3. Test with camera
4. Deploy to RC car

### Sparse Checkout:
- ✅ Pi only has `src/robots/rover`
- ✅ Model deployed within sparse checkout
- ✅ Inference script is standalone (no external dependencies)
- ✅ Everything works within the rover directory

---

## 🔗 Related Documentation

- **Quantization Guide:** See main repo `/src/policies/ACT/QUANTIZATION_GUIDE.md`
- **Deployment Script:** See main repo `/deploy_to_pi.sh`
- **Training Guide:** See main repo `/TRAINING_GUIDE.md`

---

## ✅ Quick Reference

### Deploy from laptop:
```bash
./deploy_to_pi.sh outputs/lerobot_act/best_model_quantized.pth mboels@raspberrypi
```

### Run on Pi:
```bash
cd ~/src/robots/rover
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_quantized.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0
```

**That's it! 🚀**
