# Post-Training Quantization Guide for ACT Model

**Last Updated:** October 10, 2025

## 🎯 Goal: Deploy 13.5M Parameter ACT Model on Raspberry Pi 5

Reduce inference latency from ~100ms to <20ms while preserving learned policy accuracy.

---

## 📊 Quantization Strategy Overview

| Method | Latency Improvement | Accuracy | Effort | Recommended For |
|--------|---------------------|----------|--------|-----------------|
| **Dynamic** | 2-4x faster | >99% preserved | Low | Quick deployment |
| **Static** | 3-5x faster | >98% preserved | Medium | Production |
| **Mixed** | 2-3x faster | >99.5% preserved | Medium | Best balance |

---

## 🚀 Quick Start: 3-Step Deployment

### Step 1: Quantize Your Model

```bash
# Navigate to ACT policy directory
cd /home/maxboels/projects/Erewhon/src/policies/ACT

# Dynamic quantization (RECOMMENDED for first deployment)
python quantize_act_model.py \
    --checkpoint ../../outputs/lerobot_act/best_model.pth \
    --mode dynamic \
    --output ../../outputs/lerobot_act/best_model_quantized.pth \
    --benchmark
```

**Expected Output:**
```
✅ Quantized model saved to ../../outputs/lerobot_act/best_model_quantized.pth
   Mode: dynamic
   Original size: 54.32 MB
   Quantized size: 13.81 MB
   Compression: 3.93x
   Size reduction: 74.6%

📊 Inference Latency Benchmark:
   Average: 12.45 ± 1.23 ms
   P95: 15.67 ms
   Throughput: 80.3 FPS
```

### Step 2: Test Quantized Model

```bash
# Test with a sample image
python lerobot_act_inference_quantized.py \
    --checkpoint ../../outputs/lerobot_act/best_model_quantized.pth \
    --test_image ../../src/robots/rover/episodes/episode_*/frames/frame_0000.jpg \
    --benchmark
```

### Step 3: Compare Accuracy

```bash
# Compare quantized vs original model
python quantize_act_model.py \
    --checkpoint ../../outputs/lerobot_act/best_model.pth \
    --mode dynamic \
    --output ../../outputs/lerobot_act/best_model_quantized.pth \
    --compare \
    --test_data ../../src/robots/rover/episodes
```

**Expected Output:**
```
📊 Accuracy Comparison:
   Average MSE: 0.000234
   Average Max Error: 0.012
   ✅ Excellent: Negligible accuracy loss
```

---

## 🔧 Advanced Quantization Methods

### Method 1: Dynamic Quantization (Easiest)

**Best for:** Quick deployment, minimal setup  
**Quantizes:** Weights only (INT8), activations stay FP32  
**Benefits:** 2-4x speedup, 4x smaller, <1% accuracy loss

```bash
python quantize_act_model.py \
    --checkpoint ../../outputs/lerobot_act/best_model.pth \
    --mode dynamic \
    --output ../../outputs/lerobot_act/best_model_dynamic.pth \
    --benchmark
```

**How it works:**
- Converts Linear and MultiheadAttention layers to INT8
- Keeps activations in FP32 (computed on-the-fly)
- No calibration needed
- Works immediately after training

### Method 2: Static Quantization (Best Accuracy)

**Best for:** Production deployment  
**Quantizes:** Weights AND activations (both INT8)  
**Benefits:** 3-5x speedup, 4x smaller, ~2% accuracy loss  
**Requires:** Calibration data (100-500 samples)

```bash
python quantize_act_model.py \
    --checkpoint ../../outputs/lerobot_act/best_model.pth \
    --mode static \
    --calibration_data ../../src/robots/rover/episodes \
    --num_calibration_batches 200 \
    --output ../../outputs/lerobot_act/best_model_static.pth \
    --benchmark \
    --compare \
    --test_data ../../src/robots/rover/episodes
```

**How it works:**
1. Inserts observers into model to track activation ranges
2. Runs inference on calibration data (100-500 samples)
3. Calculates optimal quantization scales
4. Converts weights and activations to INT8
5. Fuses operations (Conv+BN+ReLU) for efficiency

**Calibration tips:**
- Use diverse episodes (different scenarios)
- 100-200 samples usually sufficient
- More calibration ≠ always better (diminishing returns)

### Method 3: Mixed Precision (Optimal Balance)

**Best for:** Maximum accuracy preservation  
**Quantizes:** Transformers to INT8, Vision encoder to FP32  
**Benefits:** 2-3x speedup, ~0.5% accuracy loss  

```bash
python quantize_act_model.py \
    --checkpoint ../../outputs/lerobot_act/best_model.pth \
    --mode mixed \
    --calibration_data ../../src/robots/rover/episodes \
    --num_calibration_batches 100 \
    --output ../../outputs/lerobot_act/best_model_mixed.pth \
    --benchmark \
    --compare \
    --test_data ../../src/robots/rover/episodes
```

**How it works:**
- Quantizes transformer layers (attention, FFN) to INT8
- Keeps ResNet18 vision encoder in FP32
- Best accuracy-speed tradeoff
- Vision encoder is pre-trained, sensitive to quantization

---

## 📈 Expected Performance Gains

### On Development Machine (Reference)

| Model | Latency (avg) | P95 | Throughput | Size |
|-------|---------------|-----|------------|------|
| Original FP32 | ~100ms | ~120ms | 10 FPS | 54 MB |
| Dynamic INT8 | ~25ms | ~35ms | 40 FPS | 14 MB |
| Static INT8 | ~20ms | ~28ms | 50 FPS | 14 MB |
| Mixed | ~30ms | ~40ms | 33 FPS | 28 MB |

### On Raspberry Pi 5 (Expected)

| Model | Latency (avg) | P95 | Control Rate | Power |
|-------|---------------|-----|--------------|-------|
| Original FP32 | ~500ms | ~600ms | 2 Hz ❌ | 13W |
| Dynamic INT8 | ~50ms | ~70ms | 15-20 Hz ⚠️ | 11W |
| Static INT8 | ~30ms | ~45ms | 22-30 Hz ✅ | 11W |
| Mixed | ~40ms | ~55ms | 18-25 Hz ✅ | 11.5W |

**Target:** <33ms P95 latency for 30Hz control ✅

---

## 🔍 Accuracy Validation

### Understanding Error Metrics

**MSE (Mean Squared Error):**
- Measures average prediction difference²
- Good: <0.001 (negligible)
- Acceptable: 0.001-0.01 (minor)
- Poor: >0.01 (significant)

**Max Error:**
- Worst-case prediction difference
- Good: <0.05 (5% of range)
- Acceptable: 0.05-0.1 (10% of range)
- Poor: >0.1 (>10% of range)

### Example Validation Run

```bash
python quantize_act_model.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --mode static \
    --calibration_data src/robots/rover/episodes \
    --output outputs/lerobot_act/best_model_quantized.pth \
    --compare \
    --test_data src/robots/rover/episodes
```

**Good Result:**
```
📊 Accuracy Comparison:
   Average MSE: 0.000234
   Average Max Error: 0.012
   Max Error Range: [0.001, 0.034]
   ✅ Excellent: Negligible accuracy loss
```

**If accuracy is poor:**
```
📊 Accuracy Comparison:
   Average MSE: 0.0234
   Average Max Error: 0.156
   ⚠️ Warning: Significant accuracy loss, consider recalibration
```

**Solutions:**
1. Increase calibration samples (--num_calibration_batches 500)
2. Use more diverse calibration data
3. Try mixed precision instead
4. Check if model was properly trained

---

## 🎯 Deployment Workflow

### Complete Pipeline

```bash
#!/bin/bash
# Complete quantization and deployment pipeline

# 1. Quantize model (dynamic for quick test)
python quantize_act_model.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --mode dynamic \
    --output outputs/lerobot_act/best_model_quantized.pth

# 2. Benchmark performance
python lerobot_act_inference_quantized.py \
    --checkpoint outputs/lerobot_act/best_model_quantized.pth \
    --benchmark \
    --num_iterations 1000

# 3. Test with sample images
python lerobot_act_inference_quantized.py \
    --checkpoint outputs/lerobot_act/best_model_quantized.pth \
    --test_image src/robots/rover/episodes/episode_*/frames/frame_0000.jpg

# 4. If satisfied, create optimized static version for production
python quantize_act_model.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --mode static \
    --calibration_data src/robots/rover/episodes \
    --num_calibration_batches 200 \
    --output outputs/lerobot_act/best_model_production.pth \
    --benchmark \
    --compare \
    --test_data src/robots/rover/episodes

# 5. Transfer to Raspberry Pi
scp outputs/lerobot_act/best_model_production.pth pi@raspberrypi:~/models/
```

---

## 🔧 Raspberry Pi 5 Specific Optimizations

### 1. Use QNNPACK Backend (ARM Optimization)

Add to your inference script:

```python
import torch.backends.quantized

# Set ARM-optimized backend
torch.backends.quantized.engine = 'qnnpack'
```

### 2. Thread Optimization

```python
import torch

# Set number of threads (Pi 5 has 4 cores)
torch.set_num_threads(4)

# Enable optimizations
torch.set_flush_denormal(True)
```

### 3. Memory Management

```python
import gc

# Clear cache between inferences
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

### 4. Complete Optimized Inference Script

See: `lerobot_act_inference_rpi5.py` (created in next section)

---

## 📊 Monitoring & Profiling

### Profile Inference Latency

```bash
# Detailed profiling
python -m torch.utils.bottleneck lerobot_act_inference_quantized.py \
    --checkpoint outputs/lerobot_act/best_model_quantized.pth \
    --test_image test.jpg
```

### Real-time Monitoring

```python
import time
import numpy as np

latencies = []

for i in range(1000):
    start = time.perf_counter()
    steering, throttle = inference.predict(frame)
    end = time.perf_counter()
    latencies.append((end - start) * 1000)
    
    if i % 100 == 0:
        print(f"Current avg: {np.mean(latencies[-100:]):.2f} ms")
```

---

## ⚠️ Troubleshooting

### Issue 1: Quantization Fails

**Error:** `RuntimeError: Could not run 'quantized::...' with arguments`

**Solution:**
```python
# Ensure model is in eval mode
model.eval()

# Use CPU for quantization
model = model.to('cpu')

# Check PyTorch version (need >=1.13)
print(torch.__version__)
```

### Issue 2: Poor Accuracy After Quantization

**Symptoms:** MSE > 0.01, Max Error > 0.1

**Solutions:**
1. Increase calibration samples
2. Use diverse calibration data (different scenarios)
3. Try mixed precision
4. Check original model quality

### Issue 3: Slow Inference on Pi 5

**Expected:** <30ms, **Actual:** >100ms

**Solutions:**
1. Verify QNNPACK backend: `torch.backends.quantized.engine = 'qnnpack'`
2. Check thread count: `torch.set_num_threads(4)`
3. Ensure model is quantized (check file size)
4. Profile to find bottleneck

### Issue 4: Model Not Loading

**Error:** `KeyError: 'model_state_dict'`

**Solution:**
```python
# Check checkpoint contents
checkpoint = torch.load('model.pth')
print(checkpoint.keys())

# Load correctly
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)  # Direct state dict
```

---

## 📚 Technical Background

### What is Quantization?

**Full Precision (FP32):**
- 32 bits per number
- Range: ±3.4 × 10³⁸
- Precision: ~7 decimal digits

**Quantized (INT8):**
- 8 bits per number
- Range: -128 to 127
- Precision: Integer only

**Conversion:**
```
quantized_value = round((fp32_value - zero_point) / scale)
fp32_value = quantized_value * scale + zero_point
```

### Why It Works for Neural Networks

1. **Redundancy:** Neural networks are over-parameterized
2. **Robustness:** Small weight changes don't affect output much
3. **Statistics:** Activations often have limited range
4. **Inference Only:** No gradients needed (simpler ops)

### Layer-wise Considerations

**Good for INT8:**
- ✅ Linear layers (fully connected)
- ✅ Convolutions
- ✅ Attention mechanisms
- ✅ Batch normalization (fused)

**Keep in FP32:**
- ⚠️ Layer normalization (sensitive)
- ⚠️ First/last layers (critical)
- ⚠️ Small layers (<1000 params)

---

## 🎯 Next Steps

### After Successful Quantization:

1. **Deploy to Raspberry Pi 5**
   ```bash
   scp best_model_quantized.pth pi@raspberrypi:~/models/
   ```

2. **Test on real hardware**
   ```bash
   # On Pi
   python lerobot_act_inference_quantized.py \
       --checkpoint models/best_model_quantized.pth \
       --benchmark
   ```

3. **Integrate with RC car control**
   - Update `episode_recorder.py` to use quantized model
   - Test autonomous driving
   - Monitor latency in real-time

4. **Further optimizations** (if needed)
   - ONNX conversion for cross-platform
   - TensorRT for NVIDIA Jetson (future)
   - Hailo NPU deployment (Pi AI HAT)

---

## 📖 Additional Resources

- **PyTorch Quantization Docs:** https://pytorch.org/docs/stable/quantization.html
- **ARM QNNPACK:** https://github.com/pytorch/pytorch/tree/master/aten/src/ATen/native/quantized/cpu/qnnpack
- **LeRobot ACT Paper:** https://huggingface.co/papers/2304.13705
- **Raspberry Pi 5 Specs:** 4x Cortex-A76 @ 2.4GHz, 8GB RAM

---

## ✅ Checklist

Before deploying to Raspberry Pi 5:

- [ ] Model trained and validated
- [ ] Quantization applied and tested
- [ ] Accuracy comparison shows <1% MSE
- [ ] Benchmark shows <30ms P95 latency
- [ ] Tested with sample images
- [ ] Model file transferred to Pi
- [ ] Dependencies installed on Pi
- [ ] QNNPACK backend configured
- [ ] Real-time control tested
- [ ] Latency monitored during operation

---

**Ready to deploy! 🚀**
