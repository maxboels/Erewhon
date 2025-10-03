# RC Car Vision-Based Control - Inference Setup Guide

## Overview

This guide covers the hardware and software setup for deploying vision-based policies (ACT and VLA models) on an RC car. The system processes camera frames and outputs steering/throttle commands in real-time.

## System Architecture

### Current Training Setup
- **Arduino**: Reads PWM signals from RC receiver, sends to Pi via serial
- **Raspberry Pi 5**: Records episodes (timestamps, frame IDs, throttle, steering)
- **Camera**: Captures frames synchronized with control data
- **Output**: CSV files per episode with training data

### Inference Setup Options

The key challenge is running model inference fast enough for real-time control while maintaining low latency.

## Hardware Recommendations by Phase

### Phase 1: ACT Model (Recommended Starting Point)

**Components:**
- Raspberry Pi 5 (main compute)
- ESP32 or Arduino (PWM generation)
- Camera module (Pi Camera or USB camera)

**Performance:**
- Inference: 50-100ms per frame
- Control frequency: 10-20 Hz
- Sufficient for moderate-speed RC car control

**Why start here:**
- Uses existing hardware
- Pi 5 is 2-3x faster than Pi 4
- No network latency issues
- Simpler system to debug

### Phase 2: VLA Model (Future Upgrade)

**Components:**
- Jetson Orin Nano (main compute, $200-500)
- ESP32 (PWM generation + telemetry)
- Camera module (CSI or USB)

**Performance:**
- Inference: 30-100ms per frame
- Control frequency: 30-60 Hz
- GPU-accelerated inference with TensorRT

**Why upgrade:**
- VLA models are larger and need GPU acceleration
- Better for complex vision tasks
- Higher control frequencies = smoother driving

## Why NOT to Use Remote Server

**Avoid WiFi/Bluetooth to remote server for control:**

❌ **Cons:**
- Network latency: 20-100ms round trip (even on local WiFi)
- Packet loss and jitter
- Unreliable - car could crash during network drops
- Defeats real-time control purpose

✅ **Exception:** 
Remote server is fine for monitoring, telemetry, or offline analysis - just not for the control loop.

## Microcontroller Comparison

| Feature | Arduino | ESP32 | STM32 | Jetson |
|---------|---------|-------|-------|---------|
| Real-time | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Linux (not RT) |
| PWM precision | Excellent | Excellent | Excellent | Poor (use MCU) |
| WiFi/BT | ❌ No | ✅ Built-in | ❌ No* | ✅ Yes |
| Dual-core | ❌ No | ✅ Yes | ⚠️ Some | ✅ Yes |
| Cost | ~$5-25 | ~$5-15 | ~$10-30 | ~$200-500 |
| Best for | Simple PWM | PWM + WiFi | Advanced RT | ML inference |

*Some STM32 variants have WiFi/BT, but less common

**Recommendation:** Use **ESP32** - it combines real-time PWM with built-in WiFi for telemetry/debugging.

## Wiring Guide

### Option 1: ESP32 to Raspberry Pi 5

**Connection Type: UART (Serial)**

```
ESP32                  Raspberry Pi 5
-----------------      ------------------
GPIO1 (TX)      -----> GPIO15 (RX) [Pin 10]
GPIO3 (RX)      <----- GPIO14 (TX) [Pin 8]
GND             -----> GND [Pin 6]
```

**Power Options:**
1. **Separate power**: Use USB cable to power ESP32 independently (recommended)
2. **Pi powers ESP32**: Connect 3.3V from Pi (Pin 1) to ESP32 VIN/3V3 (only if ESP32 current draw < 500mA)

**Steps:**
1. Solder header pins to ESP32 if not pre-installed
2. Use female-to-female jumper wires for connections
3. No soldering needed if both boards have headers

**Pi Configuration:**
```bash
# Enable serial port, disable console
sudo raspi-config
# Interface Options > Serial Port
# - Login shell over serial: No
# - Serial port hardware: Yes
```

### Option 2: ESP32 to Jetson Nano/Orin

**Connection Type: UART (Serial)**

```
ESP32                  Jetson Nano
-----------------      ------------------
GPIO1 (TX)      -----> UART RX [Pin 10]
GPIO3 (RX)      <----- UART TX [Pin 8]
GND             -----> GND [Pin 6]
```

**Power:**
- Power ESP32 via USB (recommended)
- Or use Jetson 5V/3.3V pins with appropriate regulation

**Jetson Configuration:**
```bash
# Check available serial ports
ls /dev/ttyTHS*
# Usually /dev/ttyTHS1 on Jetson Nano
```

### Option 3: STM32 to Raspberry Pi/Jetson

**Connection Type: UART (Serial) or USB**

**UART Connection:**
```
STM32 (e.g., Blue Pill)    Raspberry Pi 5
------------------------   ------------------
PA9 (TX1)          ------> GPIO15 (RX) [Pin 10]
PA10 (RX1)         <------ GPIO14 (TX) [Pin 8]
GND                ------> GND [Pin 6]
```

**USB Connection (if STM32 has USB):**
- Simply connect STM32 via USB cable to Pi/Jetson
- Appears as `/dev/ttyACM0` or `/dev/ttyUSB0`
- Easier but requires USB-capable STM32 variant

**Power:**
- STM32 can be powered via USB or 3.3V pin
- Check your specific board's requirements

### PWM Output from Microcontroller to ESC/Servo

All microcontrollers connect the same way to RC hardware:

```
Microcontroller          RC Car
-----------------        ------------------
PWM Pin 1        ------> Steering Servo Signal
PWM Pin 2        ------> ESC (Throttle) Signal
GND              ------> Servo/ESC GND (common ground)
```

**Important:**
- RC servos typically use 50 Hz PWM (20ms period)
- Pulse width: 1000-2000 μs (1ms = full left, 1.5ms = center, 2ms = full right)
- Ensure common ground between microcontroller and RC electronics

## Software Setup

### Raspberry Pi 5 - Serial Communication

```python
import serial
import time

# Open serial connection
ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=1)

# Send steering and throttle values
def send_control(steering, throttle):
    """
    steering: -1.0 to 1.0
    throttle: -1.0 to 1.0
    """
    # Convert to PWM values (1000-2000 microseconds)
    steering_pwm = int(1500 + steering * 500)
    throttle_pwm = int(1500 + throttle * 500)
    
    # Send as comma-separated values
    command = f"{steering_pwm},{throttle_pwm}\n"
    ser.write(command.encode())

# Example usage
while True:
    send_control(0.5, 0.3)  # Turn right, go forward
    time.sleep(0.05)  # 20 Hz control loop
```

### ESP32 - Receiving and Outputting PWM

```cpp
#include <ESP32Servo.h>

Servo steeringServo;
Servo throttleESC;

const int STEERING_PIN = 18;
const int THROTTLE_PIN = 19;

void setup() {
  Serial.begin(115200);
  
  // Attach servos
  steeringServo.attach(STEERING_PIN, 1000, 2000);
  throttleESC.attach(THROTTLE_PIN, 1000, 2000);
  
  // Center position
  steeringServo.writeMicroseconds(1500);
  throttleESC.writeMicroseconds(1500);
}

void loop() {
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    
    // Parse "steering_pwm,throttle_pwm"
    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      int steering = data.substring(0, commaIndex).toInt();
      int throttle = data.substring(commaIndex + 1).toInt();
      
      // Constrain values
      steering = constrain(steering, 1000, 2000);
      throttle = constrain(throttle, 1000, 2000);
      
      // Output PWM
      steeringServo.writeMicroseconds(steering);
      throttleESC.writeMicroseconds(throttle);
    }
  }
}
```

### Arduino - Same Code (Compatible)

The ESP32 code above works on Arduino with minor modifications:
```cpp
#include <Servo.h>  // Use Arduino Servo library instead

// Rest of the code is identical
```

### STM32 - PWM Output

```cpp
// Using STM32duino framework
#include <Servo.h>

Servo steeringServo;
Servo throttleESC;

const int STEERING_PIN = PA6;  // TIM3_CH1
const int THROTTLE_PIN = PA7;  // TIM3_CH2

void setup() {
  Serial.begin(115200);
  
  steeringServo.attach(STEERING_PIN);
  throttleESC.attach(THROTTLE_PIN);
  
  steeringServo.writeMicroseconds(1500);
  throttleESC.writeMicroseconds(1500);
}

void loop() {
  // Same parsing logic as ESP32
  if (Serial.available()) {
    String data = Serial.readStringUntil('\n');
    int commaIndex = data.indexOf(',');
    if (commaIndex > 0) {
      int steering = data.substring(0, commaIndex).toInt();
      int throttle = data.substring(commaIndex + 1).toInt();
      
      steering = constrain(steering, 1000, 2000);
      throttle = constrain(throttle, 1000, 2000);
      
      steeringServo.writeMicroseconds(steering);
      throttleESC.writeMicroseconds(throttle);
    }
  }
}
```

## Performance Benchmarks

| Setup | ACT Inference | VLA Inference | Control Frequency |
|-------|---------------|---------------|-------------------|
| Raspberry Pi 5 | 50-100ms | 200-500ms | 10-20 Hz |
| Jetson Nano | 20-30ms | 100-200ms | 20-30 Hz |
| Jetson Orin Nano | 10-20ms | 30-100ms | 30-60 Hz |

## Optimization Tips

### For Raspberry Pi 5
- Use ONNX Runtime for inference
- Enable quantization (INT8) for faster inference
- Use lightweight vision encoders
- Consider model pruning

### For Jetson
- Use TensorRT for GPU acceleration
- Optimize model with TensorRT engine
- Use CUDA for preprocessing
- Enable NVDLA (Deep Learning Accelerator) if available

## Recommended Setup

**Start:** Raspberry Pi 5 + ESP32
- Total cost: ~$60-80
- Good for learning and ACT models
- Easy to debug

**Upgrade:** Jetson Orin Nano + ESP32
- Total cost: ~$220-520
- Required for VLA models
- Production-ready performance

## Safety Considerations

1. **Failsafe:** Implement timeout on microcontroller - if no commands received for 500ms, stop motors
2. **Testing:** Test in controlled environment before full-speed runs
3. **Kill switch:** Keep RC transmitter active as manual override
4. **Power:** Use separate battery for motors (ESC) and electronics to avoid brownouts

## Troubleshooting

**Serial communication not working:**
- Check ground connection
- Verify TX connects to RX and vice versa
- Confirm baud rates match (115200)
- Use `sudo dmesg | grep tty` to check if device detected

**Jerky PWM output:**
- Ensure common ground between all components
- Check power supply stability
- Verify control loop frequency is consistent

**High latency:**
- Profile inference time with `time.time()`
- Check camera frame rate
- Monitor CPU usage with `htop`

## Next Steps

1. Flash ESP32 with provided code
2. Wire ESP32 to Pi/Jetson following guide
3. Test serial communication with simple script
4. Deploy your trained ACT model
5. Benchmark inference speed and control frequency
6. Optimize model if needed
7. Plan Jetson upgrade when moving to VLA

## Resources

- [ESP32 Arduino Core](https://github.com/espressif/arduino-esp32)
- [PySerial Documentation](https://pyserial.readthedocs.io/)
- [Jetson GPIO Library](https://github.com/NVIDIA/jetson-gpio)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)

Good luck with your RC car project! 🚗💨