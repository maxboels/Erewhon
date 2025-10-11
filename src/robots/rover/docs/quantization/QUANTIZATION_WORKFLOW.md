# ACT Model Quantization Workflow

## Complete Pipeline: Training → Quantization → Deployment

```
┌─────────────────────────────────────────────────────────────────┐
│                    1️⃣ TRAIN ACT MODEL                           │
│                                                                 │
│  python official_lerobot_trainer.py \                          │
│      --data_dir src/robots/rover/episodes \                    │
│      --output_dir ./outputs/lerobot_act \                      │
│      --epochs 100 --batch_size 8 --device cuda                 │
│                                                                 │
│  Output: best_model.pth (54 MB, FP32, ~100ms latency)          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                 2️⃣ QUANTIZE MODEL (INT8)                        │
│                                                                 │
│  ./deploy_quantized_model.sh \                                 │
│      outputs/lerobot_act/best_model.pth \                      │
│      src/robots/rover/episodes \                               │
│      pi@raspberrypi                                            │
│                                                                 │
│  Steps:                                                         │
│  ├─ Dynamic quantization (quick test)                          │
│  ├─ Static quantization (production)                           │
│  ├─ Accuracy validation (<2% loss)                             │
│  ├─ Performance benchmark (3-5x speedup)                       │
│  ├─ Create deployment package                                  │
│  └─ Transfer to Raspberry Pi                                   │
│                                                                 │
│  Output: model_quantized.pth (14 MB, INT8, ~20-30ms latency)   │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              3️⃣ BENCHMARK ON RASPBERRY PI 5                     │
│                                                                 │
│  ssh pi@raspberrypi                                            │
│  cd ~/act_model                                                │
│                                                                 │
│  python3 lerobot_act_inference_rpi5.py \                       │
│      --checkpoint model.pth \                                  │
│      --benchmark --iterations 1000                             │
│                                                                 │
│  Expected Results:                                              │
│  ├─ Mean latency: 30-40 ms                                     │
│  ├─ P95 latency:  <50 ms ✅                                    │
│  ├─ Throughput:   25-35 FPS                                    │
│  └─ Control rate: 22-30 Hz capable                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│            4️⃣ DEPLOY TO RC CAR (AUTONOMOUS CONTROL)             │
│                                                                 │
│  python3 lerobot_act_inference_rpi5.py \                       │
│      --checkpoint model.pth \                                  │
│      --camera_id 0 \                                           │
│      --arduino_port /dev/ttyUSB0 \                             │
│      --control_freq 30                                         │
│                                                                 │
│  Real-time control loop:                                        │
│  ├─ Capture camera frame (30 FPS)                              │
│  ├─ Preprocess image (resize, normalize)                       │
│  ├─ Model inference (~30ms)                                    │
│  ├─ Action chunking (smooth control)                           │
│  ├─ Convert to PWM values                                      │
│  └─ Send to Arduino → RC car moves! 🚗                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Performance Comparison

### Before Quantization (FP32)

```
Model Size:        54 MB
Inference Time:    ~100 ms (dev machine)
                   ~600 ms (Raspberry Pi) ❌
Control Rate:      <2 Hz (unusable)
Accuracy:          100% (baseline)
```

### After Quantization (INT8 Static)

```
Model Size:        14 MB ↓ 74%
Inference Time:    ~20 ms (dev machine) ↓ 5x
                   ~40 ms (Raspberry Pi) ↓ 15x ✅
Control Rate:      25-30 Hz (smooth control)
Accuracy:          98-99% (minimal loss)
```

---

## Quantization Methods Comparison

| Method | Speedup | Size | Accuracy | Setup | Use Case |
|--------|---------|------|----------|-------|----------|
| **Dynamic** | 2-4x | ↓75% | 99%+ | Easy | Quick test |
| **Static** | 3-5x | ↓75% | 98%+ | Medium | Production ⭐ |
| **Mixed** | 2-3x | ↓50% | 99.5%+ | Medium | Max accuracy |

---

## File Structure

```
Erewhon/
├── deploy_quantized_model.sh              # 🔧 Automated pipeline
│
├── src/policies/ACT/
│   ├── official_lerobot_trainer.py       # 1️⃣ Training
│   ├── quantize_act_model.py             # 2️⃣ Quantization tool
│   ├── lerobot_act_inference_quantized.py # Testing
│   ├── lerobot_act_inference_rpi5.py     # 3️⃣ 4️⃣ Pi deployment
│   │
│   ├── QUANTIZATION_GUIDE.md             # 📖 Complete guide
│   ├── QUANTIZATION_SUMMARY.md           # 📋 Quick reference
│   └── QUANTIZATION_WORKFLOW.md          # 📊 This file
│
└── outputs/lerobot_act/
    ├── best_model.pth                     # Original FP32 model
    └── quantized/
        ├── model_dynamic_quant.pth       # Dynamic INT8
        ├── model_static_quant.pth        # Static INT8
        ├── model_final.pth               # Production model
        └── deploy/                        # 📦 Pi deployment package
            ├── model.pth
            ├── lerobot_act_inference_rpi5.py
            ├── lerobot/
            └── README.md
```

---

## Quick Commands Reference

### 1. Train Model
```bash
python src/policies/ACT/official_lerobot_trainer.py \
    --data_dir src/robots/rover/episodes \
    --output_dir ./outputs/lerobot_act \
    --epochs 100 --batch_size 8 --device cuda
```

### 2. Quantize (Automated)
```bash
./deploy_quantized_model.sh \
    outputs/lerobot_act/best_model.pth \
    src/robots/rover/episodes \
    pi@raspberrypi
```

### 3. Quantize (Manual)
```bash
# Dynamic (quick)
python src/policies/ACT/quantize_act_model.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --mode dynamic \
    --output outputs/lerobot_act/model_quant.pth \
    --benchmark

# Static (production)
python src/policies/ACT/quantize_act_model.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --mode static \
    --calibration_data src/robots/rover/episodes \
    --num_calibration_batches 200 \
    --output outputs/lerobot_act/model_quant.pth \
    --benchmark --compare --test_data src/robots/rover/episodes
```

### 4. Test Locally
```bash
# Benchmark
python src/policies/ACT/lerobot_act_inference_quantized.py \
    --checkpoint outputs/lerobot_act/model_quant.pth \
    --benchmark --num_iterations 1000

# Test image
python src/policies/ACT/lerobot_act_inference_quantized.py \
    --checkpoint outputs/lerobot_act/model_quant.pth \
    --test_image test.jpg
```

### 5. Deploy to Pi
```bash
# Transfer
scp -r outputs/lerobot_act/quantized/deploy/ pi@raspberrypi:~/act_model/

# Install (on Pi)
ssh pi@raspberrypi
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy opencv-python pillow pyserial tqdm

# Benchmark (on Pi)
cd ~/act_model
python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --benchmark

# Deploy (on Pi)
python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0 \
    --control_freq 30
```

---

## Raspberry Pi 5 Optimizations

### Hardware Configuration
- **CPU:** 4x Cortex-A76 @ 2.4 GHz
- **RAM:** 8GB LPDDR4X
- **Architecture:** ARM v8-A

### Software Optimizations
```python
# 1. ARM-optimized quantization backend
torch.backends.quantized.engine = 'qnnpack'

# 2. Use all 4 cores
torch.set_num_threads(4)

# 3. Fast preprocessing (cv2 on ARM)
frame = cv2.resize(frame, (640, 360))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# 4. Disable gradients
torch.set_grad_enabled(False)

# 5. Batch normalization fusion
torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])
```

### Expected Performance
- **Latency:** 30-45ms (P95 <50ms)
- **Control Rate:** 22-30 Hz
- **Power:** ~11W (vs 13W for FP32)
- **Model Size:** 14MB (fits in memory easily)

---

## Validation Checklist

### Before Deployment
- [ ] Model trained successfully
- [ ] Training loss converged (<0.5)
- [ ] Validation accuracy good
- [ ] Best model checkpoint saved

### Quantization
- [ ] Quantization completed without errors
- [ ] Model size reduced (54MB → 14MB)
- [ ] Accuracy preserved (MSE < 0.01)
- [ ] Latency improved (3-5x faster)

### Raspberry Pi Testing
- [ ] Dependencies installed
- [ ] Model transferred successfully
- [ ] Benchmark shows <50ms P95
- [ ] Camera working
- [ ] Arduino connected

### Autonomous Driving
- [ ] Real-time control at 30Hz
- [ ] Smooth steering and throttle
- [ ] Latency stable
- [ ] No crashes or errors

---

## Troubleshooting

### Issue: Slow inference on Pi (>100ms)

**Check:**
1. QNNPACK enabled: `torch.backends.quantized.engine`
2. Thread count: `torch.get_num_threads()` should be 4
3. Model is quantized: file size should be ~14MB, not 54MB

**Fix:**
```python
import torch.backends.quantized
torch.backends.quantized.engine = 'qnnpack'
torch.set_num_threads(4)
```

### Issue: Poor accuracy after quantization

**Check:**
1. MSE should be < 0.01
2. Max error should be < 0.1
3. Visual predictions should look reasonable

**Fix:**
1. Increase calibration samples: `--num_calibration_batches 500`
2. Use more diverse calibration data
3. Try mixed precision: `--mode mixed`

### Issue: Model won't load on Pi

**Check:**
1. PyTorch version compatible
2. Model file not corrupted
3. LeRobot source included

**Fix:**
```bash
# Reinstall PyTorch
pip3 install torch torchvision --force-reinstall

# Verify model
python3 -c "import torch; m=torch.load('model.pth'); print(m.keys())"
```

---

## Performance Monitoring

### Real-time Latency Tracking
```python
import numpy as np
from collections import deque

latencies = deque(maxlen=1000)

for frame in camera:
    start = time.perf_counter()
    steering, throttle = controller.predict(frame)
    latencies.append((time.perf_counter() - start) * 1000)
    
    if len(latencies) % 100 == 0:
        print(f"Avg: {np.mean(latencies):.1f}ms, "
              f"P95: {np.percentile(latencies, 95):.1f}ms")
```

### CPU/Memory Monitoring (on Pi)
```bash
# CPU usage
top -b -n 1 | grep python

# Memory usage
free -h

# Temperature
vcgencmd measure_temp

# Throttling status
vcgencmd get_throttled
```

---

## Next Steps

1. **Train your model** (if not done)
   ```bash
   python src/policies/ACT/official_lerobot_trainer.py ...
   ```

2. **Run quantization pipeline**
   ```bash
   ./deploy_quantized_model.sh outputs/lerobot_act/best_model.pth ...
   ```

3. **Deploy to Raspberry Pi**
   ```bash
   scp -r deploy/ pi@raspberrypi:~/act_model/
   ```

4. **Test and iterate**
   - Benchmark performance
   - Tune control frequency
   - Monitor stability

5. **Future optimizations**
   - Hailo NPU acceleration (AI HAT+)
   - ONNX export for cross-platform
   - TensorRT for Jetson (if switching platform)

---

**Ready to deploy your quantized ACT model! 🚀**

See detailed guides:
- `QUANTIZATION_GUIDE.md` - Complete documentation
- `QUANTIZATION_SUMMARY.md` - Implementation summary
