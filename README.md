# RC Car Imitation Learning - Erewhon

End-to-end autonomous RC car system using vision-based imitation learning on Raspberry Pi 5.

## Project Location

**Main project:** `src/robots/rover/`

See `src/robots/rover/README.md` for complete documentation.

## Quick Links

- **Getting Started:** `src/robots/rover/README.md`
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
3. **Train** - Learn from demonstrations
4. **Deploy** - Autonomous control on-device

---

📁 **Navigate to `src/robots/rover/` for full documentation**
