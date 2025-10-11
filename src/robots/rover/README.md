# RC Car Imitation Learning System

End-to-end system for training an RC car to drive autonomously using imitation learning on a Raspberry Pi 5.

## 📚 Documentation

- **[SPARSE_CHECKOUT_SETUP.md](./SPARSE_CHECKOUT_SETUP.md)** - Git sparse checkout configuration for Pi
- **[PI_DEPLOYMENT_COMMANDS.md](./PI_DEPLOYMENT_COMMANDS.md)** - Quick command reference for deployment
- **[QUANTIZATION_DEPLOYMENT_SUMMARY.md](./QUANTIZATION_DEPLOYMENT_SUMMARY.md)** - Deployment summary
- **[docs/quantization/](./docs/quantization/)** - Complete quantization guides and explanations
  - Quantization methods and workflows
  - ONNX format explanations
  - Hailo NPU deployment guide
  - Performance benchmarks

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

**Option A: Real-Time PWM Monitor (Recommended for Testing)**
```bash
# Live visualization of PWM signals with graphs
python3 src/eval/realtime_pwm_monitor.py --arduino-port /dev/ttyACM0

# Shows:
# - Raw PWM values (microseconds)
# - Normalized values (-1.0 to 1.0)
# - Live graphs with 10-second history
# - Statistics and calibration info
# - Duty cycle calculations

# Controls:
# [Q] Quit  |  [R] Reset  |  [S] Screenshot  |  [C] Calibration Info
```

**Option B: Comprehensive Validation Tool**
```bash
# Run full validation suite
python3 src/eval/pwm_calibration_validator.py \
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
  --episode-duration 6 \
  --output-dir ./episodes \
  --arduino-port /dev/ttyACM0 \
  --camera-id 0 \
  --action-label "hit red balloon" \
  --resolution 640x360
```

**Parameters:**
- `--episode-duration`: Recording length in seconds (default: 6)
- `--output-dir`: Where to save episodes (default: ./episodes)
- `--arduino-port`: Arduino serial port (default: /dev/ttyACM0)
- `--camera-id`: Camera device ID (default: 0)
- `--action-label`: Task description for VLA training (default: "hit red balloon")
- `--resolution`: Camera resolution WxH (default: 640x360, options: 640x360, 640x480, 1280x720)

**Examples:**
```bash
# Quick 6-second recording for testing
python3 src/record/episode_recorder.py --episode-duration 6

# Custom action label for different tasks
python3 src/record/episode_recorder.py --action-label "follow the line"

# Different camera
python3 src/record/episode_recorder.py --camera-id 1

# Use higher resolution (larger file sizes)
python3 src/record/episode_recorder.py --resolution 640x480
```

### 4. Transfer Episode Data to Your Laptop ⚡ **RECOMMENDED**

**Important**: Episode data (images) should **NOT** be pushed to GitHub. Instead, transfer them directly from the Raspberry Pi to your laptop using one of these methods:

**Option 1: SCP (Simple, one-time copy)**
```bash
# From your laptop terminal, copy episodes from Pi to local clone
scp -C -r mboels@raspberrypi:~/EDTH2025/Erewhon/src/robots/rover/episodes ./src/robots/rover/

# Copy specific episode only
scp -C -r mboels@raspberrypi:~/EDTH2025/Erewhon/src/robots/rover/episodes/episode_20251007_144013 ./src/robots/rover/episodes/
```

**Option 2: rsync (Smart, incremental sync - recommended for multiple transfers)**
```bash
# From your laptop, sync only new/changed episodes
rsync -avz --progress mboels@raspberrypi:~/EDTH2025/Erewhon/src/robots/rover/episodes/ ./src/robots/rover/episodes/

# This only copies new files, making subsequent syncs very fast
```

**Option 3: VS Code Remote (GUI - easiest)**
- In VS Code's Remote Explorer, navigate to `src/robots/rover/episodes/`
- Right-click the episode folder → **"Download..."**
- Choose your local workspace location
- Transfer happens instantly! ✨

**Why transfer directly instead of using Git?**
- ⚡ **Instant transfer** - Direct download is much faster than push/pull
- 🚀 **Faster git operations** - Keep your repository lightweight (code only)
- 💾 **Better practice** - Git is for code, not training data
- 🎯 **Selective sync** - Only download episodes you need for training
- 📦 **No GitHub limits** - Avoid large file warnings and storage issues

**Workflow:**
1. Record episodes on Raspberry Pi
2. Transfer data directly to laptop (instant!)
3. Train ML models locally with the data
4. Only push/pull code changes via Git

### 5. Validate & Analyze Episode Data

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

**Create episode animations (video/gif with live control plots):**
```bash
# Create MP4 animation for all episodes (default, skips existing)
python3 src/eval/create_episode_animation.py --data_dir ./episodes --fps 10

# Force recreation even if animations already exist
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --force

# Create animation for specific episodes
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --episodes episode_20251006_220059 episode_20251006_220145 \
  --format mp4

# Create GIF animation (smaller file size, good for sharing)
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --format gif \
  --fps 8

# Create combined multi-episode animation (side-by-side comparison)
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --combined \
  --format mp4 \
  --output_dir ./animations

# Full command with all options
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --output_dir ./episode_animations \
  --fps 10 \
  --episodes episode_20251006_220059 \
  --format mp4 \
  --combined \
  --force
```

**Animation features:**
- 📹 Camera view with synchronized frames
- 📊 Live steering and throttle plots
- 📈 Episode statistics and metadata
- 🎬 MP4 or GIF output formats
- 🔄 Single or multi-episode comparisons
- ⚡ Smart caching - skips existing animations (use `--force` to override)

**Interactive frame viewer (browse frames with keyboard):**
```bash
python3 src/eval/episode_frame_viewer.py \
  --episode-dir ./episodes/episode_20251006_220059

# Use arrow keys to navigate frames
# Press 'q' to quit
```

## Project Structure

```
src/
├── arduino/
│   └── rc_car_pwm_recording.ino    # Arduino firmware (PWM reader)
├── eval/                           # Evaluation & validation tools
│   ├── validate_episode_data.py    # Check data quality & PWM ranges
│   ├── clean_episode_data.py       # Remove anomalous samples
│   ├── pwm_statistics.py           # Analyze PWM distributions
│   ├── create_episode_animation.py # Create video/gif animations with plots
│   └── episode_frame_viewer.py     # Interactive frame browser
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
| `create_episode_animation.py` | Creates animated videos/gifs with control plots |
| `episode_frame_viewer.py` | Interactive frame-by-frame browser |
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
7. **Train** → Train state-aware ACT model on collected data
   ```bash
   # Train using BOTH camera frames AND current state
   cd ../../policies/ACT
   python3 state_aware_act_trainer.py \
     --data_dir ../../robots/rover/episodes \
     --max_epochs 50 \
     --device cuda
   
   # See QUICK_COMMANDS.md for full guide
   ```
8. **Test Inference** → Validate model on episode data
   ```bash
   python3 state_aware_inference.py \
     --model outputs/state_aware_act_XXX/best_model.pth \
     --episode ../../robots/rover/episodes/episode_XXX \
     --device cuda
   ```
9. **Deploy** → Run autonomous control on Raspberry Pi

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
    "camera_resolution": [640, 360],
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

## Data Management

**Episode data is excluded from Git:**
- The `episodes/` and `episode_animations/` folders are in `.gitignore`
- This keeps the repository lightweight and fast
- Episode data should be transferred directly (see section 4 above)
- Only code and documentation are version controlled

**Storage considerations:**
- Each 6-second episode at 640x360 ≈ 4.3 MB
- At 640x480 ≈ 13 MB (not recommended)
- 10 episodes at 640x360 ≈ 43 MB
- 50 episodes at 640x360 ≈ 215 MB

## Notes

- **Battery**: Use fully charged battery for consistent throttle range
- **Lighting**: Collect data in consistent lighting conditions
- **Driving**: Drive smoothly - the model learns from your style
- **Storage**: Each 6s episode ≈ 4.3MB at 640x360 (or ~13MB at 640x480)
- **Action Labels**: Use descriptive action labels for VLA training
- **Data Quality**: Always run validation after collecting episodes
- **Resolution**: 640x360 (default) offers 66% smaller files vs 640x480 with good quality

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
