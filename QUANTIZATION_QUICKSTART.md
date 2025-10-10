# Quick Start Guide - Quantized ACT Deployment

## 🚀 Your Setup

**SSH Connection:** `ssh mboels@raspberrypi` or `ssh raspberrypi`

---

## Step-by-Step Deployment

### 1. Quantize Your Model (One Command!)

```bash
cd /home/maxboels/projects/Erewhon

# Quantize and prepare deployment package
./deploy_quantized_model.sh \
    outputs/lerobot_act/best_model.pth \
    src/robots/rover/episodes \
    mboels@raspberrypi
```

**Or if your SSH config already has the username:**
```bash
./deploy_quantized_model.sh \
    outputs/lerobot_act/best_model.pth \
    src/robots/rover/episodes \
    raspberrypi
```

**What this does:**
- ✅ Creates dynamic quantized model (quick test)
- ✅ Creates static quantized model (production)
- ✅ Validates accuracy (<2% loss)
- ✅ Benchmarks performance (3-5x speedup)
- ✅ Creates deployment package
- ✅ Transfers to your Pi automatically

**Expected output:**
```
✅ Dynamic quantization complete!
✅ Static quantization complete!
✅ Accuracy validation complete!
   Average MSE: 0.000234 ✅
✅ Deployment package created
✅ Transfer complete!
```

---

### 2. SSH to Your Raspberry Pi

```bash
ssh mboels@raspberrypi
# or
ssh raspberrypi
```

---

### 3. Install Dependencies (One-Time Setup)

```bash
# Install PyTorch for ARM (CPU version)
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip3 install numpy opencv-python pillow pyserial tqdm
```

**Verify installation:**
```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

---

### 4. Benchmark on Raspberry Pi

```bash
cd ~/act_model

python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
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

---

### 5. Test with Camera (No Arduino Yet)

```bash
python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --camera_id 0 \
    --control_freq 30
```

**Controls:**
- Press **'q'** to quit
- You'll see steering/throttle predictions
- No movement yet (Arduino not connected)

---

### 6. Deploy to RC Car (Full Autonomous)

```bash
python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0 \
    --control_freq 30
```

**What happens:**
- 🎥 Camera captures frames (30 FPS)
- 🤖 Model predicts steering/throttle (~30-40ms)
- 📡 Sends PWM commands to Arduino
- 🚗 RC car drives autonomously!

**Monitor stats:**
- Top-left: Latency and FPS
- Bottom: Steering/throttle values
- Visual: Steering indicator and throttle bar

---

## 🔧 Troubleshooting

### Issue: "Permission denied" for camera

```bash
# Add user to video group
sudo usermod -a -G video mboels
# Log out and back in
```

### Issue: "Permission denied" for Arduino

```bash
# Add user to dialout group
sudo usermod -a -G dialout mboels
# Log out and back in
```

### Issue: Camera not found

```bash
# List available cameras
ls /dev/video*

# Test camera
v4l2-ctl --list-devices

# Try different camera ID
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --camera_id 1
```

### Issue: Arduino not responding

```bash
# Check which port Arduino is on
ls /dev/ttyUSB* /dev/ttyACM*

# Test connection
python3 -c "import serial; s=serial.Serial('/dev/ttyUSB0', 115200); print('OK')"

# Try different port
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --arduino_port /dev/ttyACM0
```

### Issue: Slow inference (>100ms)

```bash
# Check if quantized
ls -lh ~/act_model/model.pth
# Should be ~14MB, not ~54MB

# Check QNNPACK backend
python3 -c "import torch.backends.quantized; print(torch.backends.quantized.engine)"
# Should print: qnnpack

# Check CPU temperature (throttling if >80°C)
vcgencmd measure_temp
```

---

## 📊 Performance Expectations

### On Your Raspberry Pi 5:

| Metric | Expected |
|--------|----------|
| **Mean Latency** | 30-40ms |
| **P95 Latency** | <50ms |
| **Throughput** | 25-35 FPS |
| **Control Rate** | 22-30 Hz |
| **Model Size** | 14 MB |
| **Power Draw** | ~11W |

### Comparison to Original:

| | Original (FP32) | Quantized (INT8) |
|---|----------------|------------------|
| **Latency** | ~600ms ❌ | ~40ms ✅ |
| **Model Size** | 54 MB | 14 MB |
| **Control Rate** | 2 Hz | 25-30 Hz |
| **Usable?** | No | Yes! |

---

## 📋 Common Commands

### Test Specific Features:

```bash
# Benchmark only
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --benchmark

# Camera only (no Arduino)
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --camera_id 0

# Headless (no display, lower CPU usage)
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --camera_id 0 --arduino_port /dev/ttyUSB0 --no_display

# Different control frequency
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth --camera_id 0 --control_freq 20
```

### Monitor System:

```bash
# CPU usage
htop

# Temperature
watch -n 1 vcgencmd measure_temp

# GPU memory
vcgencmd get_mem gpu

# Check for throttling
vcgencmd get_throttled
# 0x0 = OK, anything else = throttling detected
```

---

## 🎯 Success Checklist

- [ ] Model quantized successfully
- [ ] Accuracy validation passed (MSE < 0.01)
- [ ] Files transferred to Pi
- [ ] Dependencies installed
- [ ] Benchmark shows <50ms P95 latency
- [ ] Camera working
- [ ] Arduino connected
- [ ] Autonomous control smooth (30Hz)

---

## 📚 Documentation

- **Complete Guide:** `src/policies/ACT/QUANTIZATION_GUIDE.md`
- **Workflow:** `src/policies/ACT/QUANTIZATION_WORKFLOW.md`
- **Hailo HAT:** `src/policies/ACT/HAILO_DEPLOYMENT_GUIDE.md`
- **ONNX Explained:** `src/policies/ACT/ONNX_AND_FORMATS_EXPLAINED.md`

---

## 🚀 When Hailo HAT Arrives

When your Hailo AI HAT arrives, upgrade for 3x better performance:

```bash
# Export to ONNX
python export_act_to_onnx.py --checkpoint best_model.pth --output act.onnx

# (Compile on x86_64 machine with Hailo tools)

# Deploy HEF to Pi
scp act.hef mboels@raspberrypi:~/
python3 hailo_inference.py --hef act.hef
```

**Result:** 10-15ms latency, 100Hz control! 🚀

---

**Questions?** Check the troubleshooting section above or the documentation!
