#!/usr/bin/env python3
"""
PWM Calibration and Real-Time Validation Tool
==============================================

This tool validates PWM signal quality and camera synchronization before training.

Features:
1. Interactive calibration to find actual min/max values for throttle and steering
2. Real-time visualization of PWM signals with camera feed
3. Stability testing to detect PWM bursts or signal issues
4. Latency measurement between PWM changes and frame capture
5. Signal quality metrics (jitter, dropout detection)

Usage:
    python3 pwm_calibration_validator.py --arduino-port /dev/ttyACM0 --camera-id 0

Requirements:
    - Arduino running rc_car_pwm_recording.ino
    - Camera connected to Raspberry Pi
    - pip install opencv-python pyserial matplotlib numpy
"""

import serial
import time
import cv2
import numpy as np
import argparse
import threading
import queue
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys
import termios
import tty
import select

@dataclass
class PWMReading:
    """Single PWM measurement"""
    timestamp: float
    steering_norm: float
    throttle_norm: float
    steering_raw_us: int
    throttle_raw_us: int
    steering_period_us: int
    throttle_period_us: int
    steering_duty: float
    throttle_duty: float

@dataclass
class CalibrationLimits:
    """Calibration limits for PWM channels"""
    steering_min_duty: float = 5.0
    steering_max_duty: float = 9.2
    steering_neutral_duty: float = 7.0
    throttle_min_duty: float = 0.0
    throttle_max_duty: float = 70.0
    throttle_neutral_duty: float = 0.0

class ArduinoReader:
    """Handles Arduino serial communication"""
    
    def __init__(self, port: str, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.data_queue = queue.Queue()
        self.latest_reading: Optional[PWMReading] = None
        self.read_thread = None
        
    def connect(self) -> bool:
        """Connect to Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            
            # Flush any garbage from reset
            self.serial_conn.reset_input_buffer()
            time.sleep(0.5)
            
            # Wait for ready signal
            start_time = time.time()
            while time.time() - start_time < 5:
                if self.serial_conn.in_waiting:
                    try:
                        line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                        if line == "ARDUINO_READY":
                            print("✓ Arduino connected")
                            return True
                    except Exception:
                        continue  # Ignore decode errors during initial connection
            
            print("✗ Arduino did not send READY signal")
            return False
            
        except serial.SerialException as e:
            print(f"✗ Failed to connect: {e}")
            return False
    
    def start(self):
        """Start reading in background"""
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
    
    def stop(self):
        """Stop reading"""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
    
    def _read_loop(self):
        """Background reading thread"""
        while self.running and self.serial_conn:
            try:
                if self.serial_conn.in_waiting:
                    line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if line.startswith('DATA,'):
                        reading = self._parse_line(line)
                        if reading:
                            self.latest_reading = reading
                            self.data_queue.put(reading)
            except Exception as e:
                # Silently ignore read errors to avoid spam
                time.sleep(0.01)
    
    def _parse_line(self, line: str) -> Optional[PWMReading]:
        """Parse Arduino data line"""
        try:
            parts = line.split(',')
            if len(parts) >= 8:
                # Robust parsing - handle floats that should be ints
                steering_raw = int(float(parts[4]))
                throttle_raw = int(float(parts[5]))
                steering_period = int(float(parts[6]))
                throttle_period = int(float(parts[7]))
                
                # Calculate duty cycles
                steering_duty = (steering_raw / steering_period * 100.0) if steering_period > 0 else 0.0
                throttle_duty = (throttle_raw / throttle_period * 100.0) if throttle_period > 0 else 0.0
                
                return PWMReading(
                    timestamp=time.time(),
                    steering_norm=float(parts[2]),
                    throttle_norm=float(parts[3]),
                    steering_raw_us=steering_raw,
                    throttle_raw_us=throttle_raw,
                    steering_period_us=steering_period,
                    throttle_period_us=throttle_period,
                    steering_duty=steering_duty,
                    throttle_duty=throttle_duty
                )
        except (ValueError, IndexError) as e:
            # Only print first few errors to avoid spam
            if not hasattr(self, '_error_count'):
                self._error_count = 0
            if self._error_count < 3:
                print(f"Parse error: {e} - Line: {line}")
                self._error_count += 1
        return None

class CameraReader:
    """Handles camera capture"""
    
    def __init__(self, camera_id: int = 0, fps: int = 30):
        self.camera_id = camera_id
        self.fps = fps
        self.cap = None
        self.running = False
        self.latest_frame = None
        self.frame_timestamp = 0.0
        self.capture_thread = None
        
    def initialize(self) -> bool:
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            ret, frame = self.cap.read()
            if ret:
                print("✓ Camera initialized")
                return True
            return False
        except Exception as e:
            print(f"✗ Camera error: {e}")
            return False
    
    def start(self):
        """Start capturing"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop(self):
        """Stop capturing"""
        self.running = False
        if self.cap:
            self.cap.release()
    
    def _capture_loop(self):
        """Background capture thread"""
        while self.running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.latest_frame = cv2.flip(frame, 0)  # Flip vertically
                self.frame_timestamp = time.time()
            time.sleep(1.0 / self.fps)

class CalibrationValidator:
    """Main calibration and validation tool"""
    
    def __init__(self, arduino: ArduinoReader, camera: CameraReader):
        self.arduino = arduino
        self.camera = camera
        self.calibration = CalibrationLimits()
        
        # Statistics tracking
        self.steering_history = deque(maxlen=300)  # 10 seconds at 30Hz
        self.throttle_history = deque(maxlen=300)
        self.steering_duty_history = deque(maxlen=300)
        self.throttle_duty_history = deque(maxlen=300)
        
        # Calibration values
        self.recorded_steering_min = None
        self.recorded_steering_max = None
        self.recorded_throttle_min = None
        self.recorded_throttle_max = None
    
    def run_menu(self):
        """Main interactive menu"""
        while True:
            self.clear_screen()
            print("\n" + "="*60)
            print("  PWM CALIBRATION & VALIDATION TOOL")
            print("="*60)
            print("\nDiagnostics:")
            print("  1. Signal Diagnostics (check if signals are present)")
            print("  2. Real-time Monitor (PWM + Camera sync)")
            print("\nCalibration & Testing:")
            print("  3. Interactive Calibration (find min/max values)")
            print("  4. Stability Test (detect bursts/jitter)")
            print("  5. Latency Test (PWM to camera delay)")
            print("\nCalibration Status:")
            print(f"  Steering: {self.recorded_steering_min or 'N/A'} - {self.recorded_steering_max or 'N/A'} duty%")
            print(f"  Throttle: {self.recorded_throttle_min or 'N/A'} - {self.recorded_throttle_max or 'N/A'} duty%")
            print("\nOptions:")
            print("  6. Export calibration to Arduino code")
            print("  0. Exit")
            print("\n" + "="*60)
            
            choice = input("\nSelect option: ").strip()
            
            if choice == '1':
                self.signal_diagnostics()
            elif choice == '2':
                self.real_time_monitor()
            elif choice == '3':
                self.interactive_calibration()
            elif choice == '4':
                self.stability_test()
            elif choice == '5':
                self.latency_test()
            elif choice == '6':
                self.export_calibration()
            elif choice == '0':
                print("\nExiting...")
                break
            else:
                print("Invalid option")
                time.sleep(1)
    
    def signal_diagnostics(self):
        """Diagnose PWM signal connectivity and quality"""
        self.clear_screen()
        print("\n" + "="*60)
        print("  SIGNAL DIAGNOSTICS")
        print("="*60)
        print("\nChecking PWM signal connectivity and quality...")
        print("Duration: 10 seconds")
        print("\nMove both steering and throttle controls during test...")
        input("Press ENTER to start...")
        
        # Collect data
        print("\nRecording... (10s)")
        readings = self._collect_samples(duration=10.0, show_progress=True)
        
        if len(readings) < 10:
            print("✗ Not enough data collected - Arduino connection issue")
            input("\nPress ENTER to continue...")
            return
        
        # Analyze each channel
        print("\n" + "="*60)
        print("  DIAGNOSTIC RESULTS")
        print("="*60)
        
        # Steering analysis
        steering_pulses = [r.steering_raw_us for r in readings]
        steering_periods = [r.steering_period_us for r in readings]
        steering_valid = [p for p in steering_pulses if 100 < p < 5000]
        steering_periods_valid = [p for p in steering_periods if 10000 < p < 30000]
        
        print("\n--- STEERING CHANNEL (Pin 3) ---")
        print(f"  Expected: 50Hz (~20,000 us period), 1000-2000 us pulse")
        print(f"  Samples received: {len(readings)}")
        print(f"  Valid pulses: {len(steering_valid)}/{len(readings)} ({len(steering_valid)/len(readings)*100:.1f}%)")
        print(f"  Valid periods: {len(steering_periods_valid)}/{len(readings)} ({len(steering_periods_valid)/len(readings)*100:.1f}%)")
        
        if steering_valid:
            print(f"  Pulse range: {min(steering_valid)}-{max(steering_valid)} us")
            print(f"  Pulse average: {np.mean(steering_valid):.0f} us")
        else:
            print(f"  Pulse range: NO VALID DATA")
            
        if steering_periods_valid:
            print(f"  Period average: {np.mean(steering_periods_valid):.0f} us")
            freq = 1000000.0 / np.mean(steering_periods_valid)
            print(f"  Calculated frequency: {freq:.1f} Hz")
        else:
            print(f"  Period: NO VALID DATA")
        
        if len(steering_valid) > len(readings) * 0.8:
            print("  Status: ✓ GOOD - Signal detected and stable")
        elif len(steering_valid) > 0:
            print("  Status: ⚠ WARNING - Intermittent signal")
        else:
            print("  Status: ✗ FAIL - No signal detected")
            print("  → Check: Is steering PWM wire connected to Arduino Pin 3?")
            print("  → Check: Is RC receiver powered on?")
            print("  → Check: Is transmitter paired with receiver?")
        
        # Throttle analysis
        throttle_pulses = [r.throttle_raw_us for r in readings]
        throttle_periods = [r.throttle_period_us for r in readings]
        throttle_valid = [p for p in throttle_pulses if 10 < p < 2000]
        throttle_periods_valid = [p for p in throttle_periods if 500 < p < 2000]
        
        print("\n--- THROTTLE CHANNEL (Pin 2) ---")
        print(f"  Expected: 900Hz (~1111 us period), 0-777 us pulse")
        print(f"  Samples received: {len(readings)}")
        print(f"  Valid pulses: {len(throttle_valid)}/{len(readings)} ({len(throttle_valid)/len(readings)*100:.1f}%)")
        print(f"  Valid periods: {len(throttle_periods_valid)}/{len(readings)} ({len(throttle_periods_valid)/len(readings)*100:.1f}%)")
        
        if throttle_valid:
            print(f"  Pulse range: {min(throttle_valid)}-{max(throttle_valid)} us")
            print(f"  Pulse average: {np.mean(throttle_valid):.0f} us")
        else:
            print(f"  Pulse range: NO VALID DATA")
            print(f"  Current raw values: {throttle_pulses[-5:] if len(throttle_pulses) >= 5 else throttle_pulses}")
            
        if throttle_periods_valid:
            print(f"  Period average: {np.mean(throttle_periods_valid):.0f} us")
            freq = 1000000.0 / np.mean(throttle_periods_valid)
            print(f"  Calculated frequency: {freq:.1f} Hz")
        else:
            print(f"  Period: NO VALID DATA")
            print(f"  Current raw periods: {throttle_periods[-5:] if len(throttle_periods) >= 5 else throttle_periods}")
        
        if len(throttle_valid) > len(readings) * 0.8:
            print("  Status: ✓ GOOD - Signal detected and stable")
        elif len(throttle_valid) > 0:
            print("  Status: ⚠ WARNING - Intermittent signal")
        else:
            print("  Status: ✗ FAIL - No signal detected")
            print("  → Check: Is throttle PWM wire connected to Arduino Pin 2?")
            print("  → Check: Are you moving the throttle trigger?")
            print("  → Check: ESC/Receiver wiring - throttle output correct?")
            print("  → Troubleshoot: Try swapping steering and throttle wires to test")
        
        # Overall assessment
        print("\n--- OVERALL ASSESSMENT ---")
        if len(steering_valid) > 0 and len(throttle_valid) == 0:
            print("  ⚠ STEERING OK, THROTTLE FAILED")
            print("  → Most likely: Throttle wire not connected to Pin 2")
            print("  → Or: Wrong wire from receiver (using wrong channel)")
            print("  → Action: Double-check physical wiring")
        elif len(throttle_valid) > 0 and len(steering_valid) == 0:
            print("  ⚠ THROTTLE OK, STEERING FAILED")
            print("  → Most likely: Steering wire not connected to Pin 3")
        elif len(steering_valid) > 0 and len(throttle_valid) > 0:
            print("  ✓ BOTH CHANNELS WORKING")
            print("  → Ready for calibration!")
        else:
            print("  ✗ NO SIGNALS DETECTED")
            print("  → Check: Arduino USB connection")
            print("  → Check: RC receiver is powered")
            print("  → Check: RC transmitter is on and paired")
        
        print("\n" + "="*60)
        input("\nPress ENTER to continue...")
    
    def interactive_calibration(self):
        """Interactive calibration to find actual PWM extremes"""
        self.clear_screen()
        print("\n" + "="*60)
        print("  INTERACTIVE CALIBRATION")
        print("="*60)
        print("\nThis will help you find the actual min/max PWM values.")
        print("You'll be prompted to move the controls to specific positions.")
        print("\nPress ENTER to continue or 'q' to cancel...")
        
        if input().strip().lower() == 'q':
            return
        
        # Calibrate steering
        print("\n--- STEERING CALIBRATION ---\n")
        
        input("1. Turn steering wheel FULL LEFT and press ENTER...")
        time.sleep(0.5)
        left_readings = self._collect_samples(duration=2.0)
        left_duty = np.mean([r.steering_duty for r in left_readings])
        left_pulse = np.mean([r.steering_raw_us for r in left_readings])
        print(f"   → Full Left: {left_duty:.2f}% duty cycle ({left_pulse:.0f} us pulse)")
        
        input("2. Center the steering wheel and press ENTER...")
        time.sleep(0.5)
        center_readings = self._collect_samples(duration=2.0)
        center_duty = np.mean([r.steering_duty for r in center_readings])
        center_pulse = np.mean([r.steering_raw_us for r in center_readings])
        print(f"   → Center: {center_duty:.2f}% duty cycle ({center_pulse:.0f} us pulse)")
        
        input("3. Turn steering wheel FULL RIGHT and press ENTER...")
        time.sleep(0.5)
        right_readings = self._collect_samples(duration=2.0)
        right_duty = np.mean([r.steering_duty for r in right_readings])
        right_pulse = np.mean([r.steering_raw_us for r in right_readings])
        print(f"   → Full Right: {right_duty:.2f}% duty cycle ({right_pulse:.0f} us pulse)")
        
        # Update steering calibration
        self.recorded_steering_min = left_duty
        self.recorded_steering_max = right_duty
        self.calibration.steering_min_duty = left_duty
        self.calibration.steering_neutral_duty = center_duty
        self.calibration.steering_max_duty = right_duty
        
        # Calibrate throttle
        print("\n--- THROTTLE CALIBRATION ---\n")
        
        input("1. Set throttle to ZERO (stopped) and press ENTER...")
        time.sleep(0.5)
        zero_readings = self._collect_samples(duration=2.0)
        zero_duty = np.mean([r.throttle_duty for r in zero_readings])
        zero_pulse = np.mean([r.throttle_raw_us for r in zero_readings])
        print(f"   → Zero throttle: {zero_duty:.2f}% duty cycle ({zero_pulse:.0f} us pulse)")
        
        input("2. Set throttle to MAXIMUM and press ENTER...")
        time.sleep(0.5)
        max_readings = self._collect_samples(duration=2.0)
        max_duty = np.mean([r.throttle_duty for r in max_readings])
        max_pulse = np.mean([r.throttle_raw_us for r in max_readings])
        print(f"   → Max throttle: {max_duty:.2f}% duty cycle ({max_pulse:.0f} us pulse)")
        
        # Update throttle calibration
        self.recorded_throttle_min = zero_duty
        self.recorded_throttle_max = max_duty
        self.calibration.throttle_min_duty = zero_duty
        self.calibration.throttle_neutral_duty = zero_duty
        self.calibration.throttle_max_duty = max_duty
        
        # Summary
        print("\n" + "="*60)
        print("  CALIBRATION COMPLETE")
        print("="*60)
        print("\nSteering:")
        print(f"  Min (Left):   {self.recorded_steering_min:.2f}%")
        print(f"  Center:       {center_duty:.2f}%")
        print(f"  Max (Right):  {self.recorded_steering_max:.2f}%")
        print(f"  Range:        {self.recorded_steering_max - self.recorded_steering_min:.2f}%")
        print("\nThrottle:")
        print(f"  Min (Zero):   {self.recorded_throttle_min:.2f}%")
        print(f"  Max (Full):   {self.recorded_throttle_max:.2f}%")
        print(f"  Range:        {self.recorded_throttle_max - self.recorded_throttle_min:.2f}%")
        print("\n" + "="*60)
        
        input("\nPress ENTER to return to menu...")
    
    def real_time_monitor(self):
        """Real-time visualization of PWM signals and camera"""
        self.clear_screen()
        print("\n" + "="*60)
        print("  REAL-TIME MONITOR")
        print("="*60)
        print("\nShowing live PWM signals synchronized with camera feed.")
        print("Press 'q' in the video window to exit")
        print("Press 'p' to toggle console printout\n")
        
        # Create window
        window_name = "PWM + Camera Monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        
        print_to_console = False
        last_print_time = 0
        
        try:
            while True:
                # Get latest data
                reading = self.arduino.latest_reading
                frame = self.camera.latest_frame
                
                if reading:
                    self.steering_history.append(reading.steering_norm)
                    self.throttle_history.append(reading.throttle_norm)
                    self.steering_duty_history.append(reading.steering_duty)
                    self.throttle_duty_history.append(reading.throttle_duty)
                    
                    # Console printout every 0.5 seconds if enabled
                    if print_to_console and time.time() - last_print_time > 0.5:
                        print(f"\r[{time.time():.1f}] " +
                              f"STEER: {reading.steering_raw_us:5d}us ({reading.steering_duty:5.2f}%) " +
                              f"| THROTTLE: {reading.throttle_raw_us:5d}us ({reading.throttle_duty:6.3f}%) " +
                              f"| Periods: S={reading.steering_period_us}us T={reading.throttle_period_us}us", 
                              end='', flush=True)
                        last_print_time = time.time()
                
                # Create visualization
                if frame is not None:
                    display_frame = self._create_monitor_frame(frame, reading)
                    cv2.imshow(window_name, display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    print_to_console = not print_to_console
                    if print_to_console:
                        print("\n✓ Console printout enabled")
                    else:
                        print("\n✗ Console printout disabled")
                
                time.sleep(0.01)
        
        finally:
            cv2.destroyAllWindows()
            if print_to_console:
                print()  # New line after printout
    
    def stability_test(self):
        """Test signal stability and detect bursts"""
        self.clear_screen()
        print("\n" + "="*60)
        print("  STABILITY TEST")
        print("="*60)
        print("\nTesting for signal stability, jitter, and bursts.")
        print("Duration: 30 seconds")
        print("\nKeep controls STEADY at center position...")
        input("Press ENTER to start...")
        
        # Collect data
        print("\nRecording... (30s)")
        readings = self._collect_samples(duration=30.0, show_progress=True)
        
        if len(readings) < 10:
            print("✗ Not enough data collected")
            input("\nPress ENTER to continue...")
            return
        
        # Analyze stability
        steering_duties = [r.steering_duty for r in readings]
        throttle_duties = [r.throttle_duty for r in readings]
        
        steering_mean = np.mean(steering_duties)
        steering_std = np.std(steering_duties)
        steering_min = np.min(steering_duties)
        steering_max = np.max(steering_duties)
        
        throttle_mean = np.mean(throttle_duties)
        throttle_std = np.std(throttle_duties)
        throttle_min = np.min(throttle_duties)
        throttle_max = np.max(throttle_duties)
        
        # Detect bursts (outliers > 3 std devs)
        steering_bursts = sum(1 for d in steering_duties if abs(d - steering_mean) > 3 * steering_std)
        throttle_bursts = sum(1 for d in throttle_duties if abs(d - throttle_mean) > 3 * throttle_std)
        
        # Calculate sample rate
        timestamps = [r.timestamp for r in readings]
        time_diffs = np.diff(timestamps)
        avg_sample_rate = 1.0 / np.mean(time_diffs) if len(time_diffs) > 0 else 0
        
        # Report
        print("\n" + "="*60)
        print("  STABILITY TEST RESULTS")
        print("="*60)
        print(f"\nSample Rate: {avg_sample_rate:.1f} Hz (expected ~30 Hz)")
        print(f"Total Samples: {len(readings)}")
        
        print("\n--- STEERING ---")
        print(f"  Mean:     {steering_mean:.3f}%")
        print(f"  Std Dev:  {steering_std:.3f}%")
        print(f"  Range:    {steering_min:.3f}% - {steering_max:.3f}%")
        print(f"  Jitter:   {steering_max - steering_min:.3f}%")
        print(f"  Bursts:   {steering_bursts} ({steering_bursts/len(readings)*100:.1f}%)")
        
        print("\n--- THROTTLE ---")
        print(f"  Mean:     {throttle_mean:.3f}%")
        print(f"  Std Dev:  {throttle_std:.3f}%")
        print(f"  Range:    {throttle_min:.3f}% - {throttle_max:.3f}%")
        print(f"  Jitter:   {throttle_max - throttle_min:.3f}%")
        print(f"  Bursts:   {throttle_bursts} ({throttle_bursts/len(readings)*100:.1f}%)")
        
        print("\n--- ASSESSMENT ---")
        if steering_std < 0.5 and throttle_std < 0.5:
            print("  ✓ EXCELLENT: Very stable signals")
        elif steering_std < 1.0 and throttle_std < 1.0:
            print("  ✓ GOOD: Acceptable stability for training")
        else:
            print("  ⚠ WARNING: High jitter detected - check connections")
        
        if steering_bursts > len(readings) * 0.01 or throttle_bursts > len(readings) * 0.01:
            print("  ⚠ WARNING: Bursts detected - check for interference")
        else:
            print("  ✓ No significant bursts detected")
        
        print("\n" + "="*60)
        input("\nPress ENTER to continue...")
    
    def latency_test(self):
        """Test latency between PWM changes and camera frames"""
        self.clear_screen()
        print("\n" + "="*60)
        print("  LATENCY TEST")
        print("="*60)
        print("\nThis test measures the delay between PWM signal changes")
        print("and when they appear in camera frames.")
        print("\nYou'll be asked to make quick control changes.")
        input("\nPress ENTER to start...")
        
        latencies = []
        
        for i in range(5):
            print(f"\n--- Test {i+1}/5 ---")
            input("Prepare to make a QUICK steering change... (press ENTER)")
            
            # Record baseline
            baseline_reading = self.arduino.latest_reading
            baseline_time = time.time()
            
            print("GO! Move steering NOW!")
            
            # Wait for change
            change_detected = False
            change_time = 0.0
            start_wait = time.time()
            
            while time.time() - start_wait < 3.0:
                current = self.arduino.latest_reading
                if current and baseline_reading:
                    if abs(current.steering_norm - baseline_reading.steering_norm) > 0.2:
                        change_time = time.time()
                        change_detected = True
                        break
                time.sleep(0.01)
            
            if not change_detected:
                print("  ✗ No significant change detected")
                continue
            
            # Calculate latency
            latency = (self.camera.frame_timestamp - change_time) * 1000  # ms
            latencies.append(latency)
            
            print(f"  → Latency: {latency:.1f} ms")
            time.sleep(1)
        
        if latencies:
            avg_latency = np.mean(latencies)
            print("\n" + "="*60)
            print("  LATENCY TEST RESULTS")
            print("="*60)
            print(f"\nAverage Latency: {avg_latency:.1f} ms")
            print(f"Min: {np.min(latencies):.1f} ms")
            print(f"Max: {np.max(latencies):.1f} ms")
            print(f"Std: {np.std(latencies):.1f} ms")
            
            if avg_latency < 50:
                print("\n✓ EXCELLENT: Low latency, good for real-time control")
            elif avg_latency < 100:
                print("\n✓ GOOD: Acceptable latency for training")
            else:
                print("\n⚠ WARNING: High latency detected")
            print("="*60)
        
        input("\nPress ENTER to continue...")
    
    def export_calibration(self):
        """Export calibration values as Arduino code"""
        if not all([self.recorded_steering_min, self.recorded_steering_max,
                   self.recorded_throttle_min, self.recorded_throttle_max]):
            print("\n✗ Please run calibration first (Option 1)")
            input("Press ENTER to continue...")
            return
        
        self.clear_screen()
        print("\n" + "="*60)
        print("  EXPORT CALIBRATION")
        print("="*60)
        print("\nArduino code snippet to update rc_car_pwm_recording.ino:\n")
        print("-" * 60)
        print(f"// Throttle: 900Hz PWM, {self.recorded_throttle_min:.1f}-{self.recorded_throttle_max:.1f}% duty cycle")
        print(f"const float THROTTLE_MIN_DUTY = {self.recorded_throttle_min:.1f};")
        print(f"const float THROTTLE_MAX_DUTY = {self.recorded_throttle_max:.1f};")
        print(f"const float THROTTLE_NEUTRAL_DUTY = {self.calibration.throttle_neutral_duty:.1f};")
        print()
        print(f"// Steering: 50Hz PWM, {self.recorded_steering_min:.1f}-{self.recorded_steering_max:.1f}% duty cycle")
        print(f"const float STEERING_MIN_DUTY = {self.recorded_steering_min:.1f};")
        print(f"const float STEERING_NEUTRAL_DUTY = {self.calibration.steering_neutral_duty:.1f};")
        print(f"const float STEERING_MAX_DUTY = {self.recorded_steering_max:.1f};")
        print("-" * 60)
        print("\nReplace the calibration constants in your Arduino code with these values.")
        
        # Save to file
        output_file = "/home/mboels/EDTH2025/Erewhon/src/robots/rover/calibration_values.txt"
        try:
            with open(output_file, 'w') as f:
                f.write("PWM Calibration Values\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Throttle Min: {self.recorded_throttle_min:.2f}%\n")
                f.write(f"Throttle Max: {self.recorded_throttle_max:.2f}%\n")
                f.write(f"Throttle Neutral: {self.calibration.throttle_neutral_duty:.2f}%\n\n")
                f.write(f"Steering Min: {self.recorded_steering_min:.2f}%\n")
                f.write(f"Steering Max: {self.recorded_steering_max:.2f}%\n")
                f.write(f"Steering Neutral: {self.calibration.steering_neutral_duty:.2f}%\n")
            print(f"\n✓ Calibration saved to: {output_file}")
        except Exception as e:
            print(f"\n✗ Could not save file: {e}")
        
        input("\nPress ENTER to continue...")
    
    def _collect_samples(self, duration: float, show_progress: bool = False) -> List[PWMReading]:
        """Collect PWM samples for a duration"""
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                reading = self.arduino.data_queue.get(timeout=0.1)
                samples.append(reading)
                
                if show_progress and len(samples) % 30 == 0:
                    elapsed = time.time() - start_time
                    print(f"  {elapsed:.1f}s / {duration:.1f}s - {len(samples)} samples", end='\r')
            except queue.Empty:
                pass
        
        if show_progress:
            print()  # New line after progress
        
        return samples
    
    def _create_monitor_frame(self, camera_frame, reading: Optional[PWMReading]) -> np.ndarray:
        """Create visualization frame with camera + graphs"""
        # Create canvas
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        
        # Resize camera frame to fit
        cam_resized = cv2.resize(camera_frame, (640, 480))
        canvas[0:480, 0:640] = cam_resized
        
        # Draw graphs on the right side
        graph_x = 650
        graph_y_steering = 50
        graph_y_throttle = 300
        graph_width = 600
        graph_height = 200
        
        # Steering graph
        self._draw_graph(canvas, self.steering_history, 
                        "Steering", graph_x, graph_y_steering, 
                        graph_width, graph_height, -1.0, 1.0, (0, 255, 0))
        
        # Throttle graph
        self._draw_graph(canvas, self.throttle_history,
                        "Throttle", graph_x, graph_y_throttle,
                        graph_width, graph_height, 0.0, 1.0, (255, 0, 0))
        
        # Current values overlay
        if reading:
            y_pos = 550
            cv2.putText(canvas, "CURRENT VALUES", (graph_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            y_pos += 35
            cv2.putText(canvas, "STEERING:", (graph_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_pos += 25
            cv2.putText(canvas, f"  Normalized: {reading.steering_norm:+.3f}", 
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25
            cv2.putText(canvas, f"  Duty Cycle: {reading.steering_duty:.2f}%",
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25
            cv2.putText(canvas, f"  Pulse: {reading.steering_raw_us} us",
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            y_pos += 25
            cv2.putText(canvas, f"  Period: {reading.steering_period_us} us",
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
            
            y_pos += 35
            cv2.putText(canvas, "THROTTLE:", (graph_x, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
            y_pos += 25
            cv2.putText(canvas, f"  Normalized: {reading.throttle_norm:.3f}",
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            y_pos += 25
            cv2.putText(canvas, f"  Duty Cycle: {reading.throttle_duty:.2f}%",
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            y_pos += 25
            cv2.putText(canvas, f"  Pulse: {reading.throttle_raw_us} us",
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
            y_pos += 25
            cv2.putText(canvas, f"  Period: {reading.throttle_period_us} us",
                       (graph_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
        
        # Instructions
        cv2.putText(canvas, "Press 'q' to exit", (20, 700),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return canvas
    
    def _draw_graph(self, canvas, data, title, x, y, width, height, y_min, y_max, color):
        """Draw a time-series graph"""
        # Background
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        # Title
        cv2.putText(canvas, title, (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Zero line
        zero_y = int(y + height - (0 - y_min) / (y_max - y_min) * height)
        cv2.line(canvas, (x, zero_y), (x + width, zero_y), (100, 100, 100), 1)
        
        # Plot data
        if len(data) > 1:
            points = []
            for i, value in enumerate(data):
                px = int(x + (i / len(data)) * width)
                py = int(y + height - ((value - y_min) / (y_max - y_min)) * height)
                py = max(y, min(y + height, py))  # Clamp
                points.append((px, py))
            
            for i in range(len(points) - 1):
                cv2.line(canvas, points[i], points[i + 1], color, 2)
        
        # Y-axis labels
        cv2.putText(canvas, f"{y_max:.1f}", (x + 5, y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(canvas, f"{y_min:.1f}", (x + 5, y + height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    @staticmethod
    def clear_screen():
        """Clear terminal screen"""
        print("\033[2J\033[H", end='')

def main():
    parser = argparse.ArgumentParser(description='PWM Calibration and Validation Tool')
    parser.add_argument('--arduino-port', type=str, default='/dev/ttyACM0',
                       help='Arduino serial port')
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  PWM CALIBRATION & VALIDATION TOOL")
    print("="*60)
    print(f"\nArduino Port: {args.arduino_port}")
    print(f"Camera ID: {args.camera_id}")
    print("\nInitializing...")
    
    # Initialize components
    arduino = ArduinoReader(args.arduino_port)
    camera = CameraReader(args.camera_id)
    
    if not arduino.connect():
        print("\n✗ Failed to connect to Arduino")
        print("Check that:")
        print("  1. Arduino is connected")
        print("  2. rc_car_pwm_recording.ino is uploaded")
        print(f"  3. Port {args.arduino_port} is correct")
        return 1
    
    if not camera.initialize():
        print("\n✗ Failed to initialize camera")
        return 1
    
    # Start background threads
    arduino.start()
    camera.start()
    
    # Wait for first data
    print("\nWaiting for data...")
    timeout = time.time() + 5
    while not arduino.latest_reading and time.time() < timeout:
        time.sleep(0.1)
    
    if not arduino.latest_reading:
        print("✗ No data received from Arduino")
        return 1
    
    print("✓ System ready!\n")
    time.sleep(1)
    
    # Run validator
    validator = CalibrationValidator(arduino, camera)
    
    try:
        validator.run_menu()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        arduino.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print("\nCleanup complete. Goodbye!")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
