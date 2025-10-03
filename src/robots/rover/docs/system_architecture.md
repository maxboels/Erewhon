# System Architecture - RC Car Imitation Learning

## Current Architecture

### Data Flow Overview
```
RC Transmitter → ESC/Receiver → Arduino UNO R3 → USB Serial → Raspberry Pi 5
                                                            ↓
                                Camera → Raspberry Pi 5 → Synchronized Episodes
                                                            ↓
                                                      ML Inference
```

### Hardware Components

**RC Car ESC/Receiver:**
- Outputs 2 PWM channels to Arduino
- **Physical Wiring** (from receiver to Arduino):
  - **Brown wire**: GND (ground)
  - **Black wire**: PWM signal → Arduino Pin 3 (Throttle)
  - **Purple wire**: PWM signal → Arduino Pin 2 (Steering)

**Arduino UNO R3:**
- **Purpose**: Real-time PWM signal acquisition using hardware interrupts
- **Wiring**:
  - Pin 2: **Steering** PWM (purple wire, ~50Hz, 1000-2000us pulses)
  - Pin 3: **Throttle** PWM (black wire, ~900Hz, 0-950us pulses)
  - GND: Brown wire (ground)
- **Output**: USB Serial at 115200 baud
- **Firmware**: `rc_car_pwm_recording.ino`

**Raspberry Pi 5:**
- **Purpose**: 
  - Data collection: camera capture + PWM data synchronization
  - ML inference: run trained model for autonomous control
  - Episode management and storage
- **Inputs**: 
  - USB Serial from Arduino (control data at ~30Hz)
  - USB/CSI Camera (visual data at 30fps)
- **Output**: Synchronized training episodes
- **Software**: 
  - `integrated_data_collector.py` - Data collection
  - `pwm_calibration_validator.py` - Pre-training validation
  - ML inference (on-device)

### Key Advantages

1. **Reliable PWM Reading**: Arduino's hardware interrupts provide precise, jitter-free PWM timing
2. **5V Compatibility**: Arduino handles 5V PWM signals directly (no level converter needed)
3. **Real-time Synchronization**: Raspberry Pi coordinates timestamps between control and camera data at 30Hz
4. **On-device Inference**: All processing happens on the RC car (no network latency)
5. **Simple Wiring**: Only 3 wires from receiver to Arduino (2 signals + ground)

## Software Architecture

### Arduino Firmware (`rc_car_pwm_recording.ino`)
- **Interrupt-based PWM measurement**: Hardware interrupts on Pin 2 and Pin 3
- **30Hz synchronized output**: Matches camera frame rate
- **CSV-format serial**: Easy parsing by Raspberry Pi
- **Built-in normalization**: 
  - Steering: -1.0 (full left) to +1.0 (full right)
  - Throttle: 0.0 (idle) to 1.0 (full speed)

### Raspberry Pi Scripts

**Pre-Training Validation:**
- `pwm_calibration_validator.py` - **PRIMARY TOOL**
  - Signal diagnostics (verify PWM signals are present)
  - Real-time monitor (visualize PWM + camera sync)
  - Interactive calibration (find actual min/max values)
  - Stability testing (detect signal issues)
  - Latency testing (measure sync delay)

**Data Collection:**
- `integrated_data_collector.py` - Main episode recording system
- `episode_analyzer.py` - Data quality analysis and visualization

**Inference (Future):**
- Autonomous control using trained model
- On-device inference on Raspberry Pi 5


## Data Format

### Arduino Serial Output
```
DATA,timestamp,steering_normalized,throttle_normalized,steering_raw_us,throttle_raw_us,steering_period_us,throttle_period_us
```

### Episode Structure
```
episodes/episode_YYYYMMDD_HHMMSS/
├── episode_data.json          # Metadata and configuration
├── control_data.csv           # Arduino PWM data with timestamps
├── frame_data.csv             # Camera frame metadata
├── frames/                    # Individual frame images
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
└── training_data.npz          # Synchronized training format
```

## Quick Start

### 1. Hardware Setup
```
RC Receiver → Arduino UNO R3:
  - Brown wire  → GND
  - Purple wire → Pin 2 (Steering)
  - Black wire  → Pin 3 (Throttle)

Arduino → Raspberry Pi 5:
  - USB cable (provides power + serial communication)

Camera → Raspberry Pi 5:
  - USB or CSI connection
```

### 2. Upload Arduino Firmware
```bash
# Open rc_car_pwm_recording.ino in Arduino IDE
# Select: Tools → Board → Arduino UNO
# Select: Tools → Port → /dev/ttyACM0 (or similar)
# Click Upload
```

### 3. Validate System (IMPORTANT - Run Before Training!)
```bash
cd /home/mboels/EDTH2025/Erewhon/src/robots/rover
python3 src/debug/pwm_calibration_validator.py --arduino-port /dev/ttyACM0 --camera-id 0

# Run these tests in order:
# 1. Signal Diagnostics - verify both PWM channels work
# 2. Real-time Monitor - check camera/PWM synchronization
# 3. Interactive Calibration - record actual min/max values
# 4. Stability Test - ensure no signal bursts/jitter
```

### 4. Collect Training Data
```bash
python3 src/record/integrated_data_collector.py \
  --episode-duration 60 \
  --output-dir ./episodes \
  --arduino-port /dev/ttyACM0 \
  --camera-id 0
```

### 5. Analyze Episodes
```bash
python3 src/record/episode_analyzer.py \
  --episode-dir ./episodes/episode_YYYYMMDD_HHMMSS \
  --plots
```

## Measured PWM Specifications

**Steering Channel (Pin 2, Purple Wire):**
- Frequency: ~47 Hz (21,300 us period)
- Pulse Range: 1008-1948 microseconds
- Full Left: 1008 us → normalized -1.0
- Center: 1468 us → normalized 0.0
- Full Right: 1948 us → normalized +1.0

**Throttle Channel (Pin 3, Black Wire):**
- Frequency: ~985 Hz (1016 us period)
- Pulse Range: 0-948 microseconds  
- Idle: 0-140 us → normalized 0.0-0.2
- Full Throttle: 948 us → normalized 1.0
- Note: Requires good battery for full range