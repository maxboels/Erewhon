# RC Car Imitation Learning - Erewhon

End-to-end autonomous RC car system using vision-based imitation learning on Raspberry Pi 5.

## Project Location

**Main project:** `src/robots/rover/`

See `src/robots/rover/README.md` for complete documentation.

## Quick Links

- **Getting Started:** `src/robots/rover/README.md`
- **Training Guide:** `TRAINING_GUIDE.md` ⭐ **NEW!**
- **System Architecture:** `src/robots/rover/docs/system_architecture.md`
- **Validation Tool:** `src/robots/rover/src/debug/pwm_calibration_validator.py` ⭐

## System Overview

```
RC Receiver → Arduino UNO → Raspberry Pi 5 → Autonomous Control
    (PWM)        (30Hz)      (Camera + ML)
```

**Hardware:**
- RC car with 2-channel receiver  
- Arduino UNO R3 (PWM reader)
- Raspberry Pi 5 (data + inference)
- Camera (30fps)

**Wiring:**
- Brown wire → GND
- Purple wire → Arduino Pin 2 (Steering)
- Black wire → Arduino Pin 3 (Throttle)

## Workflow

1. **Validate** - Run calibration tool to verify signals
2. **Collect** - Record driving episodes (camera + PWM)
3. **Train** - Learn from demonstrations (see below ⭐)
4. **Deploy** - Autonomous control on-device

## 🚀 Training with tmux (Recommended)

**Best way to train ACT policy:** Use tmux in a separate terminal for monitored, persistent training:

```bash
# In system terminal (NOT VS Code)
tmux new -s training

# Split vertically: Ctrl+B then %
# LEFT PANE - training:
conda activate lerobot
cd /home/maxboels/projects/Erewhon
python src/policies/ACT/official_lerobot_trainer.py \
    --data_dir src/robots/rover/episodes \
    --output_dir ./outputs/lerobot_act \
    --epochs 100 \
    --batch_size 8 \
    --device cuda

# RIGHT PANE (Ctrl+B then →) - GPU monitor:
watch -n 1 nvidia-smi

# Split right horizontally (Ctrl+B then ") - logs:
tail -f outputs/lerobot_act/lerobot_act_*/logs/batch_metrics.csv

# Detach: Ctrl+B then D | Reattach: tmux attach -t training
```

**Benefits:**
- ✅ Monitor GPU + training + logs simultaneously
- ✅ Survives SSH disconnections
- ✅ Can close laptop, training continues

---

📁 **Navigate to `src/robots/rover/` for full documentation**

