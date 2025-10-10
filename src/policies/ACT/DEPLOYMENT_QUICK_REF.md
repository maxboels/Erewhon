# Quick Reference: CPU vs Hailo Deployment

## 🎯 Current Setup (No Hailo HAT)

### Format: PyTorch INT8
```bash
# Quantize
./deploy_quantized_model.sh best_model.pth episodes

# Deploy
ssh pi@raspberrypi
python3 lerobot_act_inference_rpi5.py --checkpoint model.pth
```

**Result:** ~40ms, 30Hz control ✅

---

## 🚀 Future Setup (With Hailo HAT)

### Format: Hailo HEF
```bash
# 1. Export to ONNX
python export_act_to_onnx.py --checkpoint best_model.pth --output act.onnx

# 2. Compile for Hailo (on x86_64 machine)
hailo parser onnx act.onnx
hailo optimize --hw-arch hailo8l
hailo compiler --output act.hef

# 3. Deploy
scp act.hef pi@raspberrypi:~/
python3 hailo_inference.py --hef act.hef
```

**Result:** ~10-15ms, 100Hz control 🚀

---

## 📊 Comparison

| | CPU | Hailo |
|---|-----|-------|
| **Format** | `.pth` | `.hef` |
| **ONNX?** | No | Yes |
| **Latency** | 40ms | 10-15ms |
| **Power** | 11W | 8-9W |
| **Use now?** | ✅ YES | 📦 Wait for HAT |

---

## 💡 Key Takeaways

1. **ONNX = Universal format** for neural networks
2. **Different hardware = Different formats**
3. **Deploy on CPU now**, upgrade to Hailo later
4. **Same trained model**, just different compilation

---

**Questions?** See `ONNX_AND_FORMATS_EXPLAINED.md`
