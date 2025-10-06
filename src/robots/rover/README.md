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

**Basic recording (with defaults):**
```bash
python3 src/record/episode_recorder.py
```

**Full command with all options:**
```bash
python3 src/record/episode_recorder.py \
  --episode-duration 15 \
  --output-dir ./episodes \
  --arduino-port /dev/ttyACM0 \
  --camera-id 0 \
  --action-label "hit red balloon"
```

**Parameters:**
- `--episode-duration`: Recording length in seconds (default: 15)
- `--output-dir`: Where to save episodes (default: ./episodes)
- `--arduino-port`: Arduino serial port (default: /dev/ttyACM0)
- `--camera-id`: Camera device ID (default: 0)
- `--action-label`: Task description for VLA training (default: "hit red balloon")

**Examples:**
```bash
# Quick 6-second recording for testing
python3 src/record/episode_recorder.py --episode-duration 6

# Custom action label for different tasks
python3 src/record/episode_recorder.py --action-label "follow the line"

# Different camera
python3 src/record/episode_recorder.py --camera-id 1
```

### 4. Validate & Analyze Episode Data

**Validate all episodes:**
```bash
python3 src/eval/validate_episode_data.py
```

**Validate specific episode:**
```bash
python3 src/eval/validate_episode_data.py episodes/episode_20251006_220059
```

**Clean anomalous data (optional):**
```bash
python3 src/eval/clean_episode_data.py
```

**View PWM statistics:**
```bash
python3 src/eval/pwm_statistics.py
```

**Analyze single episode:**
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
├── eval/                           # Evaluation & validation tools
│   ├── validate_episode_data.py    # Check data quality & PWM ranges
│   ├── clean_episode_data.py       # Remove anomalous samples
│   └── pwm_statistics.py           # Analyze PWM distributions
├── record/
│   ├── episode_recorder.py         # Main data collection script
│   └── episode_analyzer.py         # Visualize single episode
├── inference/
│   └── autonomous_control.py       # ML inference (future)
└── docs/
    ├── system_architecture.md      # Detailed documentation
    ├── VALIDATION_SUMMARY.md       # Data quality report
    └── episode_data_validation_report.md  # Full validation details
```

## Key Files

| File | Purpose |
|------|---------|
| `rc_car_pwm_recording.ino` | Arduino firmware - reads PWM signals |
| `pwm_calibration_validator.py` | **START HERE** - validates everything works |
| `episode_recorder.py` | Records synchronized episodes with action labels |
| `validate_episode_data.py` | Validates recorded data quality & PWM ranges |
| `clean_episode_data.py` | Removes anomalous data samples (optional) |
| `pwm_statistics.py` | Analyzes PWM value distributions |
| `episode_analyzer.py` | Visualizes and analyzes single episodes |
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

1. **Validate** → Run `pwm_calibration_validator.py` to ensure hardware works
2. **Calibrate** → Record actual min/max PWM values
3. **Collect** → Record episodes with `episode_recorder.py`
   - Set appropriate `--action-label` for your task
   - Collect 10-20 episodes of good driving
4. **Validate Data** → Run `validate_episode_data.py` to check quality
   - Reviews all episodes for logging errors
   - Validates PWM ranges
   - Generates quality report
5. **Clean (optional)** → Run `clean_episode_data.py` to remove anomalies
6. **Analyze** → Use `pwm_statistics.py` and `episode_analyzer.py` for insights
7. **Train** → Train ML model on collected data
8. **Deploy** → Run inference on Raspberry Pi

## Episode Data Structure

Each recorded episode contains:
```json
{
  "episode_id": "episode_20251006_220059",
  "action_label": "hit red balloon",
  "start_time": 1759784466.376049,
  "end_time": 1759784472.4773798,
  "duration": 6.101329803466797,
  "metadata": {
    "camera_fps": 30,
    "camera_resolution": [640, 480],
    "arduino_port": "/dev/ttyACM0",
    "total_control_samples": 288,
    "total_frames": 176
  },
  "control_samples": [...],
  "frame_samples": [...]
}
```

**Files in each episode directory:**
- `episode_data.json` - Complete episode metadata with action label
- `control_data.csv` - PWM control samples
- `frame_data.csv` - Frame metadata
- `frames/` - Directory with all captured images

## Notes

- **Battery**: Use fully charged battery for consistent throttle range
- **Lighting**: Collect data in consistent lighting conditions
- **Driving**: Drive smoothly - the model learns from your style
- **Storage**: Each 60s episode ≈ 200MB (1800 frames + control data)
- **Action Labels**: Use descriptive action labels for VLA training
- **Data Quality**: Always run validation after collecting episodes

## Evaluation Tools

### validate_episode_data.py
Comprehensive validation of recorded episodes:
- Checks file structure integrity
- Validates PWM value ranges
- Detects timestamp ordering issues
- Identifies logging errors
- Generates detailed quality report

### clean_episode_data.py
Cleans anomalous data from episodes:
- Removes PWM outliers
- Filters sensor glitches
- Creates automatic backups
- Updates metadata

### pwm_statistics.py
Analyzes PWM distributions:
- Shows steering/throttle ranges used
- Calculates coverage of available range
- Identifies control biases
- Provides quality metrics

## Documentation

- `README.md` - This file (quick start & commands)
- `docs/system_architecture.md` - Complete technical details
- `docs/VALIDATION_SUMMARY.md` - Quick data quality overview
- `docs/episode_data_validation_report.md` - Detailed validation analysis

````
