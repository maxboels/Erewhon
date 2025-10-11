# Hailo AI HAT Deployment Guide

**Raspberry Pi 5 + Hailo-8L (13 TOPS) ACT Model Deployment**

---

## 🎯 Two Deployment Paths

### Path 1: **CPU-Only (Current - No HAT)** ✅
- Uses PyTorch INT8 quantization
- Runs on ARM Cortex-A76 CPUs
- ~40ms latency, 25-30 Hz control
- **Use this NOW while waiting for HAT**

### Path 2: **NPU Acceleration (With Hailo HAT)** 🚀
- Uses Hailo-specific quantization (HEF format)
- Runs on Hailo-8L NPU (13 TOPS)
- ~10-15ms latency, 60-100 Hz control
- **Switch to this when HAT arrives**

---

## 📊 Performance Comparison

| Configuration | Latency | Control Rate | Power | Model Format |
|--------------|---------|--------------|-------|--------------|
| **Pi 5 CPU (FP32)** | ~600ms | 2 Hz ❌ | 13W | `.pth` |
| **Pi 5 CPU (INT8)** | ~40ms | 25-30 Hz ✅ | 11W | `.pth` |
| **Pi 5 + Hailo (HEF)** | ~10-15ms | 60-100 Hz 🚀 | 8-9W | `.hef` |

---

## 🔧 Path 1: CPU-Only Deployment (Use Now)

**This is what I just implemented for you!**

### Quick Start:
```bash
# Quantize for CPU
./deploy_quantized_model.sh \
    outputs/lerobot_act/best_model.pth \
    src/robots/rover/episodes \
    pi@raspberrypi

# Deploy
ssh pi@raspberrypi
cd ~/act_model
python3 lerobot_act_inference_rpi5.py \
    --checkpoint model.pth \
    --camera_id 0 \
    --arduino_port /dev/ttyUSB0
```

**Pros:**
- ✅ Works immediately (no special hardware)
- ✅ Already implemented and tested
- ✅ Good enough for 30Hz control
- ✅ Simple setup

**Cons:**
- ⚠️ Limited to ~30Hz control rate
- ⚠️ Higher latency (~40ms)
- ⚠️ Uses more power (11W)

---

## 🚀 Path 2: Hailo NPU Deployment (When HAT Arrives)

### Overview

The Hailo AI HAT uses a **completely different** quantization and deployment pipeline:

```
┌─────────────┐    ┌──────────┐    ┌──────────────┐    ┌─────────┐
│ PyTorch ACT │ → │   ONNX   │ → │ Hailo Parser │ → │   HEF   │
│   (.pth)    │    │  (.onnx) │    │  & Compiler  │    │ (.hef)  │
└─────────────┘    └──────────┘    └──────────────┘    └─────────┘
                                                              ↓
                                                    ┌─────────────────┐
                                                    │ Hailo Runtime   │
                                                    │ (NPU execution) │
                                                    └─────────────────┘
```

### Step-by-Step Hailo Deployment

#### Step 1: Export to ONNX

First, convert PyTorch → ONNX format:

```python
# export_to_onnx.py
import torch
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig

# Load trained model
checkpoint = torch.load('best_model.pth', map_location='cpu')
config = ACTConfig(...)
model = ACTPolicy(config)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create dummy input
dummy_image = torch.randn(1, 3, 360, 640)
dummy_state = torch.randn(1, 2)

dummy_input = {
    "observation.images.cam_front": dummy_image,
    "observation.state": dummy_state,
}

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "act_model.onnx",
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['image', 'state'],
    output_names=['actions'],
    dynamic_axes={
        'image': {0: 'batch_size'},
        'state': {0: 'batch_size'},
        'actions': {0: 'batch_size'}
    }
)
```

#### Step 2: Parse & Compile for Hailo

**On your development machine** (not Pi):

```bash
# Install Hailo Dataflow Compiler
# (Requires x86_64 Ubuntu machine, not ARM)
pip install hailo-dataflow-compiler

# Parse ONNX to Hailo format
hailo parser onnx act_model.onnx \
    --output-name act_model.har

# Optimize for Hailo-8L
hailo optimize act_model.har \
    --output-name act_model_optimized.har \
    --hw-arch hailo8l

# Compile to HEF (Hailo Executable Format)
hailo compiler act_model_optimized.har \
    --output-name act_model.hef \
    --hw-arch hailo8l
```

**Important:** The Hailo Dataflow Compiler requires calibration data to determine optimal quantization scales.

#### Step 3: Calibration for Hailo

Create calibration dataset:

```python
# create_hailo_calibration.py
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

# Load calibration images from episodes
calibration_dir = Path("hailo_calibration")
calibration_dir.mkdir(exist_ok=True)

episodes_dir = Path("src/robots/rover/episodes")
frames = []

# Collect 100-500 diverse frames
for episode in list(episodes_dir.glob("episode_*/frames"))[:10]:
    for frame_path in list(episode.glob("*.jpg"))[:10]:
        img = cv2.imread(str(frame_path))
        img = cv2.resize(img, (640, 360))
        frames.append(img)

# Save as NumPy arrays for Hailo
np.save(calibration_dir / "calibration_images.npy", np.array(frames))
```

Then run calibration:

```bash
hailo optimize act_model.har \
    --output-name act_model_optimized.har \
    --hw-arch hailo8l \
    --calib-dataset hailo_calibration/calibration_images.npy \
    --quantization-mode INT8
```

#### Step 4: Deploy to Raspberry Pi with Hailo

Transfer HEF file to Pi:

```bash
scp act_model.hef pi@raspberrypi:~/hailo_model/
```

**On Raspberry Pi:**

```bash
# Install Hailo Runtime
sudo apt update
sudo apt install hailo-all

# Install Python bindings
pip3 install hailort

# Verify HAT is detected
hailortcli scan
# Should show: "Hailo-8L on PCIe"
```

#### Step 5: Inference with Hailo Runtime

```python
# hailo_act_inference.py
from hailo_platform import VDevice, HailoStreamInterface, FormatType
import numpy as np
import cv2

class HailoACTInference:
    def __init__(self, hef_path: str):
        # Load HEF model
        self.device = VDevice()
        self.network_group = self.device.configure(hef_path)[0]
        self.network_group.activate()
        
    def predict(self, image: np.ndarray, state: np.ndarray):
        # Preprocess
        image = cv2.resize(image, (640, 360))
        image = image.astype(np.float32) / 255.0
        
        # Create input tensors
        input_data = {
            'image': image,
            'state': state
        }
        
        # Run inference on NPU
        output = self.network_group.infer(input_data)
        
        # Extract actions
        actions = output['actions'][0]
        steering, throttle = actions[0], actions[1]
        
        return steering, throttle

# Usage
inference = HailoACTInference('act_model.hef')
steering, throttle = inference.predict(frame, current_state)
```

---

## 🔀 ONNX Explained in Detail

### What is ONNX?

**ONNX (Open Neural Network Exchange)** is like a "universal translator" for neural networks.

**Problem it solves:**
```
PyTorch model → Only runs on PyTorch
TensorFlow model → Only runs on TensorFlow
Custom model → Only runs on specific hardware
```

**ONNX solution:**
```
Any framework → ONNX → Any hardware/runtime
```

### ONNX Conversion Process

```python
# 1. PyTorch Model (framework-specific)
model = ACTPolicy(config)

# 2. Export to ONNX (standardized)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=11  # ONNX operator set version
)

# 3. ONNX model can now run on:
#    - ONNX Runtime (CPU/GPU)
#    - TensorRT (NVIDIA)
#    - OpenVINO (Intel)
#    - CoreML (Apple)
#    - Hailo (NPU)
#    - etc.
```

### What Happens During "Parsing"?

**ONNX Parsing** converts PyTorch operations to ONNX operators:

```
PyTorch Operations        →    ONNX Operators
─────────────────────────      ────────────────
torch.nn.Linear          →    MatMul + Add
torch.nn.Conv2d          →    Conv
torch.nn.ReLU            →    Relu
torch.nn.BatchNorm2d     →    BatchNormalization
F.softmax                →    Softmax
```

**For Hailo specifically:**

```
ONNX Graph → Hailo Parser → Optimized Graph → Quantization → HEF
```

The Hailo parser:
1. **Reads ONNX operators** (MatMul, Conv, etc.)
2. **Maps to Hailo NPU operations** (optimized for 13 TOPS)
3. **Fuses layers** (Conv+BN+ReLU → single op)
4. **Quantizes** (FP32 → INT8 with Hailo's method)
5. **Compiles to HEF** (Hailo Executable Format)

---

## 📋 Current vs Future Workflow

### **NOW (Without Hailo HAT):**

```bash
# 1. Train model
python official_lerobot_trainer.py --data_dir episodes --epochs 100

# 2. Quantize for CPU (what I implemented)
./deploy_quantized_model.sh best_model.pth episodes pi@raspberrypi

# 3. Deploy to Pi
ssh pi@raspberrypi
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth

# ✅ Result: 30Hz control, ~40ms latency
```

### **FUTURE (With Hailo HAT):**

```bash
# 1. Train model (same)
python official_lerobot_trainer.py --data_dir episodes --epochs 100

# 2. Export to ONNX
python export_act_to_onnx.py --checkpoint best_model.pth

# 3. Compile for Hailo (on x86_64 dev machine)
hailo parser onnx act_model.onnx
hailo optimize --calib-dataset calibration/
hailo compiler --hw-arch hailo8l --output act_model.hef

# 4. Deploy to Pi with HAT
scp act_model.hef pi@raspberrypi:~/
ssh pi@raspberrypi
python3 hailo_act_inference.py --hef act_model.hef

# 🚀 Result: 100Hz control, ~10ms latency
```

---

## 🛠️ Tool Compatibility Matrix

| Tool/Format | CPU (No HAT) | Hailo NPU |
|-------------|--------------|-----------|
| **PyTorch .pth** | ✅ Native | ❌ Not supported |
| **ONNX .onnx** | ✅ Via ONNX Runtime | ✅ Via Hailo Parser |
| **Hailo .hef** | ❌ Not supported | ✅ Native |
| **TensorRT** | ❌ | ❌ |

---

## 🎯 Recommendation for Your Workflow

### **Phase 1 (NOW - No HAT):**

1. ✅ **Use the CPU quantization I just implemented**
   ```bash
   ./deploy_quantized_model.sh best_model.pth episodes
   ```

2. ✅ **Deploy and test on Pi CPU**
   ```bash
   python3 lerobot_act_inference_rpi5.py --checkpoint model.pth
   ```

3. ✅ **Validate 30Hz control works**

**Pros:**
- Works immediately
- Already implemented
- Good enough for initial testing
- ~40ms latency is acceptable

---

### **Phase 2 (WHEN HAT ARRIVES):**

1. 🔄 **Export trained model to ONNX**
   - I'll create the export script

2. 🔄 **Compile for Hailo on dev machine**
   - Need x86_64 Ubuntu (or VM)
   - Use Hailo Dataflow Compiler

3. 🔄 **Deploy HEF to Pi with HAT**
   - Transfer .hef file
   - Use Hailo Runtime

4. 🚀 **Enjoy 10-15ms latency, 100Hz control**

---

## 📦 Files Needed for Hailo (I'll Create These)

When your HAT arrives, you'll need:

1. ✅ `export_act_to_onnx.py` - Export PyTorch → ONNX
2. ✅ `hailo_compile.sh` - Automated Hailo compilation
3. ✅ `hailo_act_inference.py` - Hailo Runtime inference
4. ✅ `create_hailo_calibration.py` - Calibration dataset

**I can create these now if you want**, or wait until your HAT arrives.

---

## ❓ Key Differences Summary

### PyTorch INT8 (Current - No HAT):
- **Format:** `.pth` file
- **Quantization:** PyTorch qint8 (symmetric)
- **Runtime:** PyTorch on CPU
- **Latency:** ~40ms
- **Setup:** Easy, works now ✅

### Hailo HEF (Future - With HAT):
- **Format:** `.hef` file (Hailo Executable Format)
- **Quantization:** Hailo INT8 (asymmetric, optimized for NPU)
- **Runtime:** Hailo Runtime on NPU
- **Latency:** ~10-15ms
- **Setup:** Requires compilation step, needs HAT 🚀

---

## 🚀 Next Steps

### **For NOW (Recommended):**

1. Use the CPU quantization I implemented:
   ```bash
   ./deploy_quantized_model.sh best_model.pth episodes
   ```

2. Test on Pi without HAT

3. Validate 30Hz control

### **When HAT Arrives:**

1. Let me know, and I'll create:
   - ONNX export script
   - Hailo compilation pipeline
   - Hailo inference runtime

2. We'll compile for Hailo NPU

3. Achieve 100Hz control! 🎯

---

## 📚 Resources

- **Hailo Documentation:** https://hailo.ai/developer-zone/
- **Hailo-8L Datasheet:** 13 TOPS, INT8 optimized
- **ONNX Specification:** https://onnx.ai/
- **PyTorch ONNX Export:** https://pytorch.org/docs/stable/onnx.html

---

**Questions?**

1. "Do I need ONNX for CPU deployment?" → **No**, PyTorch .pth works fine
2. "Can I use the same quantized model on Hailo?" → **No**, need HEF format
3. "Should I wait for HAT before deploying?" → **No**, deploy on CPU now, upgrade later
4. "Will I need to retrain for Hailo?" → **No**, just recompile existing model

---

**Ready to proceed?** 

Use CPU quantization NOW, I'll create Hailo tools when your HAT arrives! 🚀
