# Calibration Issues Analysis

**Date:** October 7, 2025

## Issues Identified

### Issue 1: Steering Center Offset (-0.07)
**Symptom:** When the steering is centered, normalized value shows -0.07 instead of 0.0

**Root Cause:** RC transmitter steering trim is slightly off-center
- **Measured center pulse:** 1456 μs (most common centered value)
- **Measured duty cycle:** 6.83% 
- **Expected center duty:** 7.0% (from calibration)
- **Offset:** -0.17% duty cycle difference
- **Result:** -0.07 normalized value

### Issue 2: Throttle Stays High When Stopped (0.1687)
**Symptom:** At the end of the episode when car is stopped, throttle reads 0.1687 instead of 0.0

**Root Cause:** ESC has a non-zero "neutral/stopped" signal
- **Measured stopped pulse:** 120 μs
- **Measured duty cycle:** 11.81%
- **Expected stopped duty:** 0.0% (from calibration)
- **Offset:** 11.81% duty cycle
- **Result:** 0.1687 normalized value (11.81% / 70% max = 16.87%)

## Data Validation

Both issues are **NOT plotting errors** - they are in the **RAW DATA** from the Arduino/receiver system.

Evidence from `control_data.csv`:
```csv
# Centered steering (most of episode):
steering_raw_us: 1456, 1460, 1468  → normalized: -0.07 to -0.04

# Stopped throttle (end of episode):
throttle_raw_us: 120, period: 1016  → normalized: 0.1687
```

## Solutions

### Quick Fix: Update Arduino Calibration Constants

Update the Arduino code to match your actual hardware:

```cpp
// Current (incorrect) calibration:
const float STEERING_NEUTRAL_DUTY = 7.0;    // Off by -0.17%
const float THROTTLE_NEUTRAL_DUTY = 0.0;    // Off by +11.81%

// Corrected calibration based on measurements:
const float STEERING_NEUTRAL_DUTY = 6.83;   // Matches your 1456 μs center
const float THROTTLE_NEUTRAL_DUTY = 11.81;  // Matches your 120 μs stopped
```

### Proper Fix: Recalibrate with Actual Min/Max Values

Run the calibration tool and record the actual values:

```bash
python3 src/eval/pwm_calibration_validator.py \
  --arduino-port /dev/ttyACM0 \
  --camera-id 0
```

Then use **Option 3: Interactive Calibration** to find:
1. Steering full left, center, full right
2. Throttle stopped, full throttle

Update these measured values in the Arduino code.

### Alternative: Post-Processing Correction

If you don't want to update Arduino firmware, you can apply corrections during training data preprocessing:

```python
# Correct steering offset
steering_corrected = steering_normalized + 0.07

# Correct throttle offset
throttle_corrected = max(0.0, (throttle_normalized - 0.1687) / (1.0 - 0.1687))
```

## Recommendations

### For Current Training Data
**Option A:** Use data as-is
- The offsets are consistent across all episodes
- Neural networks can learn to compensate for constant biases
- No action needed

**Option B:** Post-process existing data
- Write a script to apply corrections to CSV files
- Update normalized values in place
- Keeps raw data intact for reference

### For Future Data Collection
**Option C:** Fix Arduino calibration (recommended)
- Most accurate solution
- All future data will be properly centered
- Requires re-uploading firmware

**Option D:** Physical calibration
- Adjust RC transmitter steering trim
- Might not be possible to change ESC neutral point (hardware limitation)
- Some ESCs have a "dead zone" around neutral that causes this

## Impact on Training

### Low Impact (Can Proceed)
- Steering offset of -0.07 is minor (3.5% of full range)
- Throttle offset is consistent (always 16.87% at stop)
- Modern neural networks are robust to these calibration offsets
- The relative changes in steering/throttle are still captured correctly

### Benefits of Fixing
- Cleaner data interpretation
- More intuitive debugging
- Better transfer if you change hardware
- Proper zero-centered steering is better for symmetric learning

## Conclusion

**Both issues are REAL hardware calibration offsets, not software bugs.**

The data is still usable for training, but updating the Arduino calibration constants will produce cleaner data for future episodes.
