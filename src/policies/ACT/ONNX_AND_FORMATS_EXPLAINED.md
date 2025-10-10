# ONNX & Quantization Formats Explained

**Your Questions Answered:**
1. What does "model parsing ONNX" mean?
2. Are there different quantization formats for Pi 5 with/without Hailo AI HAT?

---

## 📚 Q1: What is ONNX and Model Parsing?

### ONNX (Open Neural Network Exchange)

**Think of ONNX as a "universal translator" for neural networks.**

```
┌─────────────┐
│  PyTorch    │ ─┐
│   Model     │  │
└─────────────┘  │
                 │    ┌──────────┐    ┌─────────────────┐
┌─────────────┐  │    │          │    │  - ONNX Runtime │
│ TensorFlow  │ ─┼──→ │   ONNX   │ ──→│  - TensorRT     │
│   Model     │  │    │  Format  │    │  - Hailo NPU    │
└─────────────┘  │    │          │    │  - CoreML       │
                 │    └──────────┘    │  - OpenVINO     │
┌─────────────┐  │                    └─────────────────┘
│   Other     │ ─┘
│ Frameworks  │
└─────────────┘
```

### What "Parsing" Means

**Parsing ONNX** = Converting your model from one format to ONNX's standardized format

**Example: PyTorch → ONNX**

```python
# PyTorch operations (framework-specific)
class MyModel(nn.Module):
    def forward(self, x):
        x = self.linear(x)      # PyTorch Linear
        x = F.relu(x)           # PyTorch ReLU
        return x

# ↓ ONNX Export "parses" this into...

# ONNX operators (standardized)
graph {
  node {
    op_type: "MatMul"         # Linear → MatMul + Add
    op_type: "Add"
  }
  node {
    op_type: "Relu"           # ReLU → Relu
  }
}
```

**Why this matters:**
- ✅ Your model can run on **any hardware** that supports ONNX
- ✅ Framework-agnostic (PyTorch, TensorFlow, etc.)
- ✅ Optimized for specific hardware (Hailo, TensorRT, etc.)

### When You Need ONNX

**You DON'T need ONNX if:**
- Running on CPU with PyTorch ✅ (your current setup)
- Using standard PyTorch inference

**You DO need ONNX for:**
- Hailo AI HAT deployment 🚀
- Cross-platform deployment
- Hardware accelerators (NPU, TPU)
- Optimized inference engines

---

## 🔄 Q2: Different Quantization Formats

**YES! The quantization format is COMPLETELY DIFFERENT with vs without the Hailo HAT.**

### Scenario 1: Pi 5 WITHOUT Hailo HAT (Current)

**Format:** PyTorch `.pth` (INT8 quantized)

```python
# Quantization method
torch.backends.quantized.engine = 'qnnpack'  # ARM CPU backend
model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# Save
torch.save(model.state_dict(), 'model_quantized.pth')
```

**Properties:**
- File: `.pth` (PyTorch checkpoint)
- Backend: QNNPACK (ARM CPU optimized)
- Runtime: PyTorch inference on CPU
- Latency: ~40ms
- Power: 11W

**Deployment:**
```bash
./deploy_quantized_model.sh best_model.pth episodes
# Output: model_quantized.pth (14 MB)
```

---

### Scenario 2: Pi 5 WITH Hailo AI HAT (Future)

**Format:** Hailo `.hef` (Hailo Executable Format)

```bash
# Complete different pipeline!
PyTorch Model (.pth)
    ↓
ONNX Model (.onnx)
    ↓
Hailo Parser → Hailo Optimized (.har)
    ↓
Hailo Compiler → Hailo Executable (.hef)
```

**Properties:**
- File: `.hef` (Hailo-specific binary)
- Backend: Hailo NPU (13 TOPS)
- Runtime: Hailo Runtime (not PyTorch!)
- Latency: ~10-15ms
- Power: 8-9W

**Deployment:**
```bash
# Step 1: Export to ONNX
python export_act_to_onnx.py --checkpoint best_model.pth --output act.onnx

# Step 2: Compile for Hailo (on x86_64 dev machine)
hailo parser onnx act.onnx --output act.har
hailo optimize act.har --hw-arch hailo8l --output act_opt.har
hailo compiler act_opt.har --output act.hef

# Step 3: Deploy HEF to Pi
scp act.hef pi@raspberrypi:~/
python3 hailo_inference.py --hef act.hef
```

---

## 📊 Comparison Table

| Aspect | **CPU (No HAT)** | **Hailo NPU (With HAT)** |
|--------|------------------|--------------------------|
| **Model Format** | `.pth` (PyTorch) | `.hef` (Hailo) |
| **Intermediate** | Not needed | `.onnx` required |
| **Quantization Type** | PyTorch INT8 (symmetric) | Hailo INT8 (asymmetric, optimized) |
| **Quantization Tool** | PyTorch quantization API | Hailo Dataflow Compiler |
| **Backend** | QNNPACK (ARM CPU) | Hailo NPU (13 TOPS) |
| **Runtime** | PyTorch | Hailo Runtime |
| **Compilation** | Not needed | Required (on x86_64) |
| **Latency** | ~40ms | ~10-15ms ⚡ |
| **Power** | 11W | 8-9W |
| **Setup Complexity** | Easy ✅ | Medium 🔧 |
| **Use Now?** | **YES** ✅ | Wait for HAT 📦 |

---

## 🛠️ Key Differences Explained

### 1. Quantization Method

**CPU (QNNPACK):**
```python
# Symmetric INT8 quantization
# Range: -128 to 127 (centered at 0)
quantized_value = round(float_value / scale)
```

**Hailo NPU:**
```python
# Asymmetric INT8 quantization
# Range: 0 to 255 or custom range
quantized_value = round((float_value - zero_point) / scale)
```

Hailo's method better preserves accuracy for activations with non-zero mean.

### 2. Compilation Pipeline

**CPU:** No compilation needed
```
PyTorch Model → Quantize → Save .pth → Run on Pi ✅
```

**Hailo:** Multi-step compilation
```
PyTorch → ONNX → Parse → Optimize → Compile → .hef → Run on Pi NPU 🚀
```

### 3. Runtime Environment

**CPU:**
```python
import torch
model = torch.load('model.pth')
output = model(input)  # PyTorch inference
```

**Hailo:**
```python
from hailo_platform import VDevice
device = VDevice()
network = device.configure('model.hef')[0]
output = network.infer(input)  # Hailo NPU inference
```

Completely different APIs!

---

## 🎯 What This Means for You

### **RIGHT NOW (No HAT):**

1. ✅ **Use PyTorch INT8 quantization I implemented**
   ```bash
   ./deploy_quantized_model.sh best_model.pth episodes
   ```

2. ✅ **Deploy to Pi with CPU inference**
   ```bash
   python3 lerobot_act_inference_rpi5.py --checkpoint model.pth
   ```

3. ✅ **You'll get ~40ms latency, 25-30 Hz control**
   - Totally usable!
   - No special hardware needed
   - Works immediately

**You do NOT need ONNX for this.** PyTorch → Quantized PyTorch → Done.

---

### **WHEN HAT ARRIVES:**

1. 🔄 **Export to ONNX** (I created the script for you)
   ```bash
   python export_act_to_onnx.py --checkpoint best_model.pth --output act.onnx
   ```

2. 🔄 **Compile for Hailo** (need x86_64 machine)
   ```bash
   # Install Hailo Dataflow Compiler
   # Then:
   hailo parser onnx act.onnx
   hailo optimize --hw-arch hailo8l --calib-dataset calibration/
   hailo compiler --output act.hef
   ```

3. 🚀 **Deploy HEF to Pi**
   ```bash
   scp act.hef pi@raspberrypi:~/
   python3 hailo_inference.py --hef act.hef
   ```

4. 🚀 **Enjoy 10-15ms latency, 100 Hz control!**

---

## 🤔 Common Questions

### Q: "Can I use the same .pth file for both CPU and Hailo?"

**A:** No! They need different formats:
- CPU: `.pth` (PyTorch quantized)
- Hailo: `.hef` (Hailo compiled binary)

But you can convert: `.pth` → ONNX → `.hef`

### Q: "Do I need to retrain for Hailo?"

**A:** No! Use the same trained model:
1. Train once: `best_model.pth`
2. For CPU: Quantize with PyTorch
3. For Hailo: Export → ONNX → Compile to HEF

Same model, different deployment formats.

### Q: "Why not use ONNX Runtime on CPU?"

**A:** You could, but PyTorch is simpler for CPU-only:
- PyTorch: Native, works out of box ✅
- ONNX Runtime: Extra dependency, same performance

ONNX shines for **hardware accelerators** (Hailo, TensorRT).

### Q: "Should I wait for HAT before deploying?"

**A:** NO! Deploy on CPU now:
- Works immediately
- Good enough for testing
- Easy upgrade path when HAT arrives

---

## 📋 Summary

### ONNX Parsing:
- **What it is:** Converting model operations to standardized ONNX format
- **When you need it:** For Hailo HAT, cross-platform deployment
- **When you don't:** CPU-only PyTorch inference (your current case)

### Quantization Formats:

**Without HAT:**
- Format: PyTorch `.pth` (INT8)
- Use: `deploy_quantized_model.sh` ✅
- Result: ~40ms, works now

**With HAT:**
- Format: Hailo `.hef` (custom INT8)
- Use: ONNX → Hailo compiler 🚀
- Result: ~10-15ms, when HAT arrives

### Recommendation:

1. **NOW:** Use CPU quantization (already implemented)
2. **Test:** Validate 30Hz control works
3. **LATER:** When HAT arrives, I'll help you compile for Hailo
4. **Upgrade:** ~40ms → ~10ms, easy transition!

---

## 📁 Files Ready for You

### Available NOW (CPU deployment):
- ✅ `quantize_act_model.py` - PyTorch INT8 quantization
- ✅ `lerobot_act_inference_rpi5.py` - CPU inference
- ✅ `deploy_quantized_model.sh` - Automated pipeline

### Ready for Hailo HAT:
- ✅ `export_act_to_onnx.py` - ONNX export (just created)
- 📦 `hailo_compile.sh` - Compilation script (I can create when needed)
- 📦 `hailo_inference.py` - NPU inference (I can create when needed)

---

**Bottom Line:**

- **ONNX = Universal model format** (like PDF for neural networks)
- **Different hardware = Different quantization formats**
- **CPU (now) = PyTorch .pth** ✅
- **Hailo (later) = Hailo .hef** 🚀
- **You're ready to deploy on CPU immediately!**

Questions? Let me know! 🚀
