# Quick Reference Card

## Wiring
```
RC Receiver → Arduino UNO:
  Brown  → GND
  Purple → Pin 2 (Steering)
  Black  → Pin 3 (Throttle)

Arduino → Raspberry Pi:
  USB cable
```

## Essential Commands

### 1. Validate System (Run First!)
```bash
python3 src/debug/pwm_calibration_validator.py \
  --arduino-port /dev/ttyACM0 --camera-id 0
```

### 2. Collect Episodes
```bash
python3 src/record/integrated_data_collector.py \
  --episode-duration 60 \
  --arduino-port /dev/ttyACM0
```

### 3. Analyze Quality
```bash
python3 src/record/episode_analyzer.py \
  --episode-dir ./episodes/episode_* \
  --plots
```

## PWM Specifications (Measured)

| Channel | Pin | Wire | Pulse Range | Normalized |
|---------|-----|------|-------------|------------|
| Steering | 2 | Purple | 1008-1948 us | -1.0 to +1.0 |
| Throttle | 3 | Black | 0-948 us | 0.0 to 1.0 |

**Data Rate**: 30 Hz (synchronized)

## Troubleshooting

### No Arduino
```bash
ls /dev/ttyACM*  # Check if connected
```

### No PWM Signal
1. Turn on RC transmitter
2. Check wires (Purple→2, Black→3, Brown→GND)
3. Run diagnostics (Option 1 in validator)

### Camera Not Found
```bash
ls /dev/video*  # List cameras
```

### Low Throttle
- Use fully charged battery
- Check ESC calibration

## File Locations

| Purpose | File |
|---------|------|
| Arduino Firmware | `src/arduino/rc_car_pwm_recording.ino` |
| Validation Tool | `src/debug/pwm_calibration_validator.py` |
| Data Collection | `src/record/integrated_data_collector.py` |
| Data Analysis | `src/record/episode_analyzer.py` |
| Documentation | `docs/system_architecture.md` |

## Workflow

1. ✅ Validate → Run calibration tool
2. ✅ Collect → Record 10-20 episodes  
3. ✅ Analyze → Check data quality
4. ⏳ Train → ML model (future)
5. ⏳ Deploy → Autonomous control (future)

---

**Current Status**: System validated and working ✓
