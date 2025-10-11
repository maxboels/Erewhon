# Quantization Documentation for Raspberry Pi Deployment

This folder contains comprehensive documentation about the ACT model quantization process and deployment strategies for the Raspberry Pi 5.

## 📚 Documentation Index

### Core Guides

1. **[QUANTIZATION_GUIDE.md](./QUANTIZATION_GUIDE.md)** - Complete quantization guide
   - Dynamic, static, and mixed precision quantization methods
   - Performance benchmarks and expected results
   - Step-by-step quantization workflow
   - Accuracy validation techniques
   - Raspberry Pi 5 specific optimizations

2. **[QUANTIZATION_WORKFLOW.md](./QUANTIZATION_WORKFLOW.md)** - Quick reference workflow
   - Fast deployment paths
   - Command cheat sheet
   - Common troubleshooting

3. **[ONNX_AND_FORMATS_EXPLAINED.md](./ONNX_AND_FORMATS_EXPLAINED.md)** - Format explanations
   - What is ONNX and when to use it
   - Different quantization formats (PyTorch vs Hailo)
   - Model conversion pipelines
   - Format comparison tables

4. **[HAILO_DEPLOYMENT_GUIDE.md](./HAILO_DEPLOYMENT_GUIDE.md)** - NPU acceleration guide
   - CPU-only vs Hailo NPU deployment
   - Performance comparisons
   - Future upgrade path when Hailo HAT arrives
   - Compilation pipeline for Hailo-8L

## 🎯 Quick Start

If you're on the Raspberry Pi and want to run inference immediately:

### 1. Check Your Model
```bash
ls -lh ~/src/robots/rover/models/
# You should see best_model_static_quantized.pth (~85 MB)
```

### 2. Install Dependencies (if not done)
```bash
pip3 install -r ~/src/robots/rover/requirements_pi.txt --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run Benchmark
```bash
cd ~/src/robots/rover
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_static_quantized.pth \
    --benchmark \
    --iterations 1000
```

### 4. Expected Performance
- **Latency:** ~40ms (P95: ~50ms)
- **Control Rate:** 25-30 Hz
- **Model Size:** 85 MB (12x compression from original 1024 MB)

## 📊 Quantization Results Summary

Our ACT model (13.5M parameters) was successfully quantized:

| Method | Size | Compression | Latency (Dev) | Latency (Pi) | Accuracy |
|--------|------|-------------|---------------|--------------|----------|
| Original FP32 | 1024 MB | 1x | ~100ms | ~600ms | 100% |
| Dynamic INT8 | 171 MB | 6x | ~25ms | ~60ms | >99% |
| **Static INT8** | **85 MB** | **12x** | **~20ms** | **~40ms** | **>98%** |

**Deployed Version:** Static INT8 (best balance of size, speed, and accuracy)

## 🔧 Understanding Quantization Modes

### Dynamic Quantization
- **What:** Only weights are INT8, activations stay FP32
- **When:** Quick testing, no calibration data needed
- **Pro:** Easy to deploy, minimal accuracy loss
- **Con:** Larger file size, moderate speedup

### Static Quantization (Current Deployment)
- **What:** Both weights AND activations are INT8
- **When:** Production deployment, maximum optimization
- **Pro:** Maximum compression and speedup
- **Con:** Requires calibration data

### Mixed Precision
- **What:** Transformers INT8, vision encoder FP32
- **When:** Maximum accuracy preservation needed
- **Pro:** Best accuracy-speed tradeoff
- **Con:** Larger than static, more complex

## 🚀 Performance Optimization Tips

### For Raspberry Pi 5

1. **Use QNNPACK Backend** (ARM-optimized)
   ```python
   import torch
   torch.backends.quantized.engine = 'qnnpack'
   ```

2. **Set Thread Count** (4 cores on Pi 5)
   ```python
   torch.set_num_threads(4)
   ```

3. **Disable Gradients** (inference only)
   ```python
   torch.set_grad_enabled(False)
   ```

4. **Memory Management**
   ```python
   import gc
   gc.collect()  # Between inferences
   ```

## 🎮 Real-Time Inference

For autonomous RC car control:

```bash
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_static_quantized.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0 \
    --control_freq 30
```

**Expected Results:**
- Control loop: 25-30 Hz (target: 30 Hz)
- Latency: ~40ms average, ~50ms P95
- Camera: 640x360 @ 30 FPS
- Arduino: PWM commands at 30 Hz

## 📖 Additional Resources

### On Development Machine

See `src/policies/ACT/` for:
- `quantize_act_model.py` - Quantization tool
- `lerobot_act_inference_quantized.py` - Inference testing
- `export_act_to_onnx.py` - ONNX export for Hailo

### Deployment Info

See parent directory:
- `../PI_DEPLOYMENT_COMMANDS.md` - Installation commands
- `../QUANTIZATION_DEPLOYMENT_SUMMARY.md` - Deployment summary
- `../requirements_pi.txt` - Exact dependency versions

## ⚠️ Important Notes

### Version Compatibility
The quantized model MUST use the same PyTorch version as training:
- **Required:** PyTorch 2.8.0
- **Required:** NumPy 2.2.6
- **Required:** OpenCV 4.12.0.88

Using different versions may cause:
- Model loading failures
- Incorrect predictions
- QNNPACK backend errors

### Future Upgrades

When the Hailo AI HAT arrives:
1. Export model to ONNX (see `ONNX_AND_FORMATS_EXPLAINED.md`)
2. Compile for Hailo-8L NPU (see `HAILO_DEPLOYMENT_GUIDE.md`)
3. Expect 3-4x latency improvement (~10-15ms)
4. Upgrade to 60-100 Hz control rate

## 🐛 Troubleshooting

### Model Won't Load
```bash
# Check PyTorch version
python3 -c "import torch; print(torch.__version__)"
# Should be: 2.8.0

# Check quantization backend
python3 -c "import torch; print(torch.backends.quantized.engine)"
# Should be: qnnpack
```

### Poor Performance
```bash
# Run benchmark to measure latency
python3 src/inference/act_inference_quantized.py \
    --checkpoint models/best_model_static_quantized.pth \
    --benchmark \
    --iterations 1000

# Check CPU usage
htop  # Should use 4 cores at ~100% during inference
```

### Accuracy Issues
```bash
# Compare with development machine results
# Expected: MSE < 0.001, Max Error < 0.05

# If accuracy is poor:
# 1. Verify model transfer (checksums)
# 2. Check Python version matches (3.10)
# 3. Verify NumPy version (2.2.6)
```

## 📞 Support

For questions about quantization or deployment:
1. Check the guides in this folder
2. Review `PI_DEPLOYMENT_COMMANDS.md` for quick commands
3. See training logs in development machine: `outputs/lerobot_act/*/logs/`

---

**Last Updated:** October 11, 2025  
**Model Version:** best_model_static_quantized.pth (85 MB)  
**Quantization:** Static INT8 with 200 calibration samples  
**Target Hardware:** Raspberry Pi 5 (4x Cortex-A76, 8GB RAM)
