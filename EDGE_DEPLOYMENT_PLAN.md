# Edge Deployment Hardware Plan

## Two-Platform Strategy

**Platform A**: Raspberry Pi 5 + Hailo AI HAT - Optimized for ACT  
**Platform B**: Jetson Orin Nano + STM32 - Optimized for VLA models

---

## Platform A: Raspberry Pi 5 + Hailo AI HAT + Arduino UNO

**Best for: ACT Policy (13.5M parameters)**

### Hardware Overview
- **Compute**: Raspberry Pi 5 8GB + Hailo-8L NPU (13 TOPS)
- **Motor Control**: Arduino UNO R3 (ATmega328P @ 16 MHz)
- **Camera**: Pi Camera Module 3 (1080p60)
- **Total Cost**: $230 (or $280 with battery)

### Performance - ACT Model
| Metric | Performance |
|--------|-------------|
| **Inference Latency** | 10-20ms |
| **Control Frequency** | 50-100 Hz |
| **Total End-to-End** | ~50ms |
| **Memory Usage** | ~1.5GB |
| **Power Consumption** | 11-13W |
| **Battery Life** | 5-8 hours (5000mAh LiPo) |

### Bill of Materials
| Component | Price |
|-----------|-------|
| Raspberry Pi 5 8GB | $80 |
| AI HAT+ 13 TOPS (Hailo-8L) | $70 |
| Pi Camera Module 3 | $25 |
| Arduino UNO R3 | $25 |
| microSD 64GB + cables + PSU | $30 |
| **SUBTOTAL** | **$230** |
| *Optional: Battery + regulator* | *+$50* |

### Model Compatibility

✅ **ACT (13.5M params)**: Excellent  
- 10-20ms inference is ideal for real-time control
- 13 TOPS sufficient for efficient deployment
- Proven, reliable performance

❌ **SmolVLA (2B params)**: Not Recommended  
- Would require 1-2 seconds inference (too slow)
- 13 TOPS insufficient for transformer models
- Poor user experience for VLA demos

❌ **Larger VLAs**: Impossible  
- Insufficient compute and memory

### Pros & Cons

**Advantages:**
- ✅ Most affordable option ($230)
- ✅ Power efficient (11W = longest battery life)
- ✅ Perfect ACT performance (10-20ms)
- ✅ Simple setup and deployment
- ✅ Compact and easy to mount

**Limitations:**
- ❌ ACT-only platform (no VLA capability)
- ❌ Cannot run language models
- ❌ Arduino 2KB RAM limits control complexity
- ❌ No room for future model expansion

**Use Cases:**
- Production ACT deployment
- Cost-optimized autonomous systems
- Long battery life applications
- Educational demonstrations
- ACT baseline for comparison

---

## Platform B: Jetson Orin Nano 8GB + STM32

**Best for: VLA Models (SmolVLA 2B, hybrid architectures)**

### Hardware Overview
- **Compute**: Jetson Orin Nano 8GB (67 TOPS, 1024 CUDA cores)
- **Motor Control**: STM32 Nucleo-F446RE (Cortex-M4 @ 180 MHz)
- **Camera**: IMX219 CSI Camera (8MP)
- **Total Cost**: $499 (or $572 with battery)

### Performance by Model

#### ACT (13.5M params)
| Metric | Performance |
|--------|-------------|
| **Inference Latency** | 3-5ms |
| **Control Frequency** | 200-300 Hz |
| **Memory Usage** | ~600MB |
| **Power** | ~10W |
| **Verdict** | ✅ Excellent (overkill) |

#### SmolVLA (2B params, INT8)
| Metric | Performance |
|--------|-------------|
| **Inference Latency** | 200-400ms |
| **Control Frequency** | 2-5 Hz (high-level) |
| **Memory Usage** | ~4GB |
| **Power** | 15-20W |
| **Verdict** | ✅ Good demo quality |

#### Hybrid (ACT + SmolVLA)
- **SmolVLA**: Language understanding + waypoint planning @ 2-5 Hz
- **ACT**: Real-time trajectory execution @ 50-100 Hz
- **Memory**: ~5GB total
- **Power**: ~22W
- **Verdict** | ✅ Impressive capabilities |

### Bill of Materials
| Component | Price |
|-----------|-------|
| Jetson Orin Nano 8GB Dev Kit | $399 |
| STM32 Nucleo-F446RE | $15 |
| IMX219 CSI Camera (8MP) | $25 |
| M.2 NVMe SSD 256GB | $40 |
| Cables + mounting hardware | $20 |
| **SUBTOTAL** | **$499** |
| *Optional: Battery + regulator* | *+$73* |

### Model Compatibility Matrix

| Model | Latency | Usability | Recommended |
|-------|---------|-----------|-------------|
| **ACT (13.5M)** | 3-5ms | Excellent | ✅ Yes (baseline) |
| **SmolVLA (2B)** | 200-400ms | Good | ✅ Yes (primary use) |
| **Pi0 (4-5B)** | 5-10s | Poor | ❌ Need Orin NX 16GB |
| **Larger VLAs (7B+)** | Too slow | N/A | ❌ Need AGX Orin |

### Pros & Cons

**Advantages:**
- ✅ VLA capabilities with natural language
- ✅ 5x faster than Pi5 (67 vs 13 TOPS)
- ✅ Can run ACT, SmolVLA, or hybrid
- ✅ Professional STM32 MCU (180 MHz)
- ✅ Multi-camera support (8-lane CSI-2)
- ✅ Future-proof CUDA/TensorRT ecosystem
- ✅ Industry-standard skills

**Limitations:**
- ❌ 2.2x more expensive ($499 vs $230)
- ❌ Higher power (15-20W vs 11-13W)
- ❌ More complex setup
- ❌ VLA latency (300-500ms) slower than ACT
- ⚠️ 8GB RAM limits to 2B models

**Use Cases:**
- Vision-language-action demonstrations
- Natural language robot control
- VLA research platform
- Multi-sensor autonomous systems
- Portfolio-quality robotics projects
- CUDA/TensorRT learning

---

## Platform Comparison

### Performance by Model

| Model | Platform A (Pi5 + Hailo) | Platform B (Jetson Orin) |
|-------|--------------------------|--------------------------|
| **ACT (13.5M)** | ✅ 10-20ms (excellent) | ✅ 3-5ms (overkill) |
| **SmolVLA (2B)** | ❌ 1-2s (too slow) | ✅ 200-400ms (good) |
| **Pi0 (4-5B)** | ❌ Impossible | ⚠️ 5-10s (barely usable) |
| **Larger VLAs** | ❌ Impossible | ❌ Needs AGX Orin |

### Cost-Performance Summary

| Metric | Platform A | Platform B | Winner |
|--------|-----------|-----------|--------|
| **Cost** | $230 | $499 | A (2.2x cheaper) |
| **AI Compute** | 13 TOPS | 67 TOPS | B (5x faster) |
| **ACT Latency** | 10-20ms | 3-5ms | B (2-4x faster) |
| **VLA Capable** | ❌ No | ✅ Yes | B |
| **Power Draw** | 11-13W | 15-20W | A (lower) |
| **Battery Life** | 5-8 hours | 4-6 hours | A (longer) |
| **MCU** | Arduino (16 MHz) | STM32 (180 MHz) | B (11x faster) |

### Decision Guide

**Choose Platform A (Pi5 + Hailo) if:**
- ✅ Only deploying ACT model
- ✅ Budget constrained ($230 max)
- ✅ Need longest battery life
- ✅ Want simple, proven setup
- ✅ Learning embedded basics

**Choose Platform B (Jetson Orin) if:**
- ✅ Want VLA capabilities (SmolVLA)
- ✅ Need language understanding
- ✅ Building portfolio project
- ✅ Budget allows $499
- ✅ Want professional-grade platform
- ✅ Plan future experiments

**Choose BOTH if:**
- ✅ Direct comparison needed
- ✅ Budget allows $729 total
- ✅ Research/teaching project

---

## Recommended Strategy

### For ACT Deployment Only
**→ Platform A (Pi5 + Hailo): $230**
- Perfect ACT performance at lowest cost
- Simple setup, long battery life
- Production-ready solution

### For VLA Experimentation
**→ Platform B (Jetson Orin): $499**
- SmolVLA at demo-quality speeds
- Can run ACT as baseline
- Professional platform for portfolio

### For Research/Comparison
**→ Both Platforms: $729 total**
- Direct ACT comparison (Pi5 vs Jetson)
- VLA capabilities on Jetson
- Complete research story

---

## Next Steps

### Platform A Quick Start (4 weeks to demo)
1. **Week 1**: Order hardware, setup Pi5 + Hailo
2. **Week 2**: Export ACT to Hailo format, flash Arduino
3. **Week 3**: Integration testing, calibration
4. **Week 4**: Validation and benchmarking

### Platform B Quick Start (8-10 weeks to demo)
1. **Weeks 1-2**: Setup Jetson, install JetPack + TensorRT
2. **Weeks 3-4**: Deploy SmolVLA, quantize, optimize
3. **Weeks 5-6**: STM32 integration, safety features
4. **Weeks 7-8**: Hybrid architecture, demo scenarios
5. **Weeks 9-10**: Optimization and polish

---

## Additional Resources

### Motor Control Details
See full STM32 setup guide and communication protocol in the original detailed plan.

### Model Deployment Workflows
- **ACT on Pi5**: PyTorch → ONNX → Hailo HEF
- **SmolVLA on Jetson**: PyTorch → ONNX → TensorRT engine (INT8)

### Safety Implementation
Both platforms require:
- Command timeout detection (200ms)
- Emergency stop capability
- PWM output limits
- Heartbeat monitoring

---

## Appendix: Hardware Upgrades

### If SmolVLA Too Slow on Orin Nano
**Jetson Orin NX 16GB**: $799 (100 TOPS, 16GB RAM)
- SmolVLA: 150-300ms (1.5x faster)
- Pi0 (4-5B): 2-5s (becomes usable)
- **Worth it?** Only if testing larger VLAs

### If Need Maximum VLA Performance
**AGX Orin 64GB**: $1,870 (275 TOPS, 64GB RAM)
- SmolVLA: 80-150ms (3x faster than Nano)
- Pi0: 500ms-1s (production quality)
- **Worth it?** Only for multi-robot or research lab setups
- **Better value**: 2x Orin Nano for different projects

**Recommendation**: Start with Orin Nano 8GB. Upgrade only if you hit clear limitations.
1. **Hardware setup**:
   - Flash JetPack 6.0 SDK to Orin Nano
   - Connect CSI camera (640x360 @ 30fps)
   - Connect to Arduino via USB/UART for PWM control

2. **Software deployment**:
   - Export trained ACT model to ONNX
   - Optimize with TensorRT (FP16 precision)
   - Deploy inference script with camera → model → PWM pipeline
   - Validate <5ms latency achieved

3. **Testing**:
   - Bench test inference speed
   - Test real-time control loop at 50-100 Hz
   - Verify Arduino communication latency

### Phase 2: VLA Integration (Week 3-4)
1. **Model preparation**:
   - Download SmolVLA checkpoint (recommend 2B variant)
   - Quantize to INT8 with NVIDIA's quantization toolkit
   - Optimize with TensorRT

2. **Hierarchical control**:
   - SmolVLA runs at 1-5 Hz for waypoint planning
   - ACT runs at 50-100 Hz for trajectory tracking
   - Implement asynchronous pipeline (VLA in separate thread)

3. **Integration testing**:
   - Test combined memory usage (<6GB target)
   - Monitor power consumption (<20W target)
   - Validate hierarchical control performance

### Phase 3: Production (Week 5+)
1. **Optimization**:
   - Fine-tune VLA on your specific environment
   - Profile and optimize inference pipeline
   - Implement model switching based on task

2. **Deployment**:
   - Design compact mount for Orin on RC car
   - Battery sizing for 15-25W continuous draw
   - Thermal management (small heatsink + fan)

## Power & Thermal Considerations

### Battery Sizing
- **Average power**: 15W (ACT) + 5W (sensors) = 20W
- **Peak power**: 25W (ACT + VLA) + 5W = 30W
- **Recommended battery**: 4S LiPo (14.8V) 5000mAh = 74Wh
- **Runtime**: 74Wh / 20W = 3.7 hours typical, 2.5 hours with VLA active

### Cooling
- **Passive**: Included heatsink sufficient for 15W
- **Active**: Add 40mm 5V fan for continuous 25W operation
- **Ambient**: Designed for 0-50°C operation

## Development Workflow

```bash
# On Orin Nano (JetPack 6.0)

# 1. Setup TensorRT environment
sudo apt-get install nvidia-tensorrt python3-libnvinfer

# 2. Export ACT to ONNX (on training PC)
python export_act_to_onnx.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --output act_model.onnx

# 3. Convert to TensorRT on Jetson
trtexec --onnx=act_model.onnx \
    --saveEngine=act_fp16.trt \
    --fp16 \
    --workspace=2048

# 4. Run inference
python jetson_act_inference.py \
    --model act_fp16.trt \
    --camera /dev/video0 \
    --arduino /dev/ttyUSB0 \
    --frequency 100
```

## Summary

**Purchase: Jetson Orin Nano 8GB Developer Kit ($399)**

This single platform gives you:
- ✅ 3-5ms ACT latency (faster than Pi5 HAT)
- ✅ SmolVLA capability (200-400ms)
- ✅ Simpler integration (no dual-system complexity)
- ✅ Better value ($399 vs $574)
- ✅ Room for future expansion (multi-camera, sensor fusion)
- ✅ Strong ecosystem (JetPack SDK, TensorRT optimization)

**Skip**: Pi5 + AI HAT entirely. The Orin Nano does everything better for just $224 more than the Pi5 setup alone.

**Future upgrade path**: If you later need Pi0 or larger VLAs, upgrade to Orin NX 16GB ($599). But start with Nano 8GB + SmolVLA to validate the concept first.
