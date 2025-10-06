# RC Car Imitation Learning System

End-to-end system for training an RC car to drive autonomously using imitation learning on a Raspberry Pi 5.

## System Overview

```
┌─────────────┐      ┌──────────────┐      ┌────────────────────┐
│ RC Receiver │─PWM─→│ Arduino UNO  │─USB─→│  Raspberry Pi 5    │
│             │      │              │      │  - Camera          │
└─────────────┘      └──────────────┘      │  - Data Collection │
                                            │  - ML Inference    │
                                            └────────────────────┘
```

## Hardware Wiring

**RC Receiver → Arduino:**
- **Brown wire**: GND (ground)
- **Purple wire**: Pin 2 → Steering PWM (~50Hz, 1000-2000us)
- **Black wire**: Pin 3 → Throttle PWM (~900Hz, 0-950us)

**Arduino → Raspberry Pi:**
- USB cable (power + serial @115200 baud)

**Camera → Raspberry Pi:**
- USB or CSI camera connection

## Quick Start

### 1. Setup Arduino
```bash
# Upload firmware using Arduino IDE
# File: src/arduino/rc_car_pwm_recording.ino
# Board: Arduino UNO
# Port: /dev/ttyACM0
```

### 2. Validate System ⚠️ **DO THIS FIRST!**
```bash
# Run comprehensive validation tool
python3 src/debug/pwm_calibration_validator.py \
  --arduino-port /dev/ttyACM0 \
  --camera-id 0

# Test checklist:
# ✓ Signal Diagnostics - verify PWM signals
# ✓ Real-time Monitor - check synchronization  
# ✓ Interactive Calibration - record min/max values
# ✓ Stability Test - ensure signal quality
```

### 3. Collect Training Data
```bash
python3 src/robots/rover/src/record/episode_recorder.py \
  --episode-duration 6 \
  --output-dir ./src/robots/rover/episodes/ \
  --arduino-port /dev/ttyACM0
```

### 4. Analyze Data Quality
```bash
python3 src/record/episode_analyzer.py \
  --episode-dir ./episodes/episode_YYYYMMDD_HHMMSS \
  --plots
```

## Project Structure

```
src/
├── arduino/
│   └── rc_car_pwm_recording.ino    # Arduino firmware (PWM reader)
├── debug/
│   └── pwm_calibration_validator.py # Pre-training validation tool ⭐
├── record/
│   ├── integrated_data_collector.py # Main data collection
│   └── episode_analyzer.py          # Data quality analysis
├── inference/
│   └── autonomous_control.py        # ML inference (future)
└── docs/
    └── system_architecture.md       # Detailed documentation
```

## Key Files

| File | Purpose |
|------|---------|
| `rc_car_pwm_recording.ino` | Arduino firmware - reads PWM signals |
| `pwm_calibration_validator.py` | **START HERE** - validates everything works |
| `integrated_data_collector.py` | Records synchronized episodes |
| `episode_analyzer.py` | Visualizes and validates collected data |
| `system_architecture.md` | Complete system documentation |

## Data Flow

1. **RC Transmitter** → sends control signals
2. **RC Receiver** → outputs PWM signals (steering + throttle)
3. **Arduino** → reads PWM with interrupts, sends to Pi @ 30Hz
4. **Raspberry Pi** → synchronizes PWM + camera frames
5. **Training** → learns from human demonstrations
6. **Inference** → autonomous control on-device

## PWM Signal Specifications

**Measured from actual hardware:**

- **Steering (Pin 2)**: 1008-1948 us pulses @ ~47Hz
- **Throttle (Pin 3)**: 0-948 us pulses @ ~985Hz
- **Sample Rate**: 30Hz (synchronized with camera)
- **Normalization**:
  - Steering: -1.0 (left) to +1.0 (right)
  - Throttle: 0.0 (idle) to 1.0 (full)

## Troubleshooting

### No PWM signal detected
1. Check Arduino is powered via USB
2. Verify RC transmitter is on and paired
3. Check wire connections (purple→Pin2, black→Pin3, brown→GND)
4. Run signal diagnostics tool

### Camera not found
```bash
# List available cameras
ls /dev/video*

# Test camera
python3 -c "import cv2; print('OK' if cv2.VideoCapture(0).read()[0] else 'FAIL')"
```

### Arduino not connecting
```bash
# Check USB connection
ls /dev/ttyACM* /dev/ttyUSB*

# Check permissions
sudo usermod -a -G dialout $USER
# Then logout/login
```

## Dependencies

```bash
# Install Python packages
pip install opencv-python pyserial numpy matplotlib

# System packages (if needed)
sudo apt-get install python3-opencv
```

## Development Workflow

1. **Validate** → Run `pwm_calibration_validator.py`
2. **Calibrate** → Record actual min/max PWM values
3. **Collect** → Record 10-20 episodes of good driving
4. **Analyze** → Check data quality with `episode_analyzer.py`
5. **Train** → Train ML model on collected data
6. **Deploy** → Run inference on Raspberry Pi

## Notes

- **Battery**: Use fully charged battery for consistent throttle range
- **Lighting**: Collect data in consistent lighting conditions
- **Driving**: Drive smoothly - the model learns from your style
- **Storage**: Each 60s episode ≈ 200MB (1800 frames + control data)

## Documentation

See `docs/system_architecture.md` for complete technical details.
