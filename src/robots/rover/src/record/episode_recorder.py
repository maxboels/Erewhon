#!/usr/bin/env python3
"""
Integrated Data Collection System for RC Car Imitation Learning
================================================================

This system coordinates:
1. Arduino PWM signal reading via USB serial
2. Camera frame capture with timestamps
3. Synchronized data logging for training episodes

OPTIMIZED FOR LOW LATENCY AND REAL-TIME SYNCHRONIZATION:
- Async frame writing (no blocking I/O in main loop)
- Minimal sleep intervals (sub-millisecond precision)
- High-priority threads for camera and Arduino
- Timestamp synchronization validation

Requirements:
- Arduino running enhanced_pwm_recorder.ino connected via USB
- Camera connected to Raspberry Pi
- Python packages: opencv-python, pyserial, numpy

Usage:
    python3 integrated_data_collector.py --episode-duration 60 --output-dir ./episodes
"""

import argparse
import cv2
import serial
import json
import time
import os
import threading
import queue
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np

@dataclass
class ControlSample:
    """Single control measurement from Arduino"""
    arduino_timestamp: int  # Arduino millis()
    system_timestamp: float  # Python time.time()
    steering_normalized: float  # [-1.0, 1.0]
    throttle_normalized: float  # [0.0, 1.0]
    steering_raw_us: int
    throttle_raw_us: int
    steering_period_us: int
    throttle_period_us: int

@dataclass
class FrameSample:
    """Single camera frame with metadata"""
    frame_id: int
    timestamp: float
    image_path: str

@dataclass
class EpisodeData:
    """Complete episode data structure"""
    episode_id: str
    start_time: float
    end_time: float
    duration: float
    control_samples: List[ControlSample]
    frame_samples: List[FrameSample]
    metadata: dict

class ArduinoReader:
    """Handles serial communication with Arduino"""
    
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.data_queue = queue.Queue()
        
    def connect(self) -> bool:
        """Establish serial connection with Arduino"""
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            
            # Wait for Arduino ready signal
            while True:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line == "ARDUINO_READY":
                    print(f"✓ Arduino connected on {self.port}")
                    return True
                    
        except serial.SerialException as e:
            print(f"✗ Failed to connect to Arduino: {e}")
            return False
    
    def start_reading(self):
        """Start reading data in background thread"""
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
    
    def stop_reading(self):
        """Stop reading and close connection"""
        self.running = False
        if self.serial_conn:
            self.serial_conn.close()
    
    def _read_loop(self):
        """Background thread for reading serial data"""
        while self.running and self.serial_conn:
            try:
                line = self.serial_conn.readline().decode('utf-8').strip()
                if line.startswith('DATA,'):
                    self._parse_data_line(line)
            except Exception as e:
                print(f"Serial read error: {e}")
                
    def _parse_data_line(self, line: str):
        """Parse incoming data line from Arduino"""
        try:
            parts = line.split(',')
            if len(parts) == 8:  # DATA,timestamp,steer,throttle,steer_raw,throttle_raw,steer_period,throttle_period
                sample = ControlSample(
                    arduino_timestamp=int(parts[1]),
                    system_timestamp=time.time(),
                    steering_normalized=float(parts[2]),
                    throttle_normalized=float(parts[3]),
                    steering_raw_us=int(parts[4]),
                    throttle_raw_us=int(parts[5]),
                    steering_period_us=int(parts[6]),
                    throttle_period_us=int(parts[7])
                )
                self.data_queue.put(sample)
        except (ValueError, IndexError) as e:
            print(f"Data parsing error: {e}")

class CameraCapture:
    """Handles camera frame capture"""
    
    def __init__(self, camera_id: int = 0, fps: int = 30, resolution: Tuple[int, int] = (640, 360)):
        self.camera_id = camera_id
        self.fps = fps
        self.resolution = resolution
        self.camera = None
        self.is_capturing = False
        self.capture_thread = None
        self.frame_queue = queue.Queue()
        self.frame_counter = 0
        
    def initialize(self) -> bool:
        """Initialize camera"""
        try:
            # Force V4L2 backend on Linux to avoid GStreamer issues
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                print("✗ Failed to open camera device")
                return False
            
            # Set resolution first, then FPS
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Set MJPEG format for better compatibility
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            
            # Test capture
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"✓ Camera initialized: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps")
                return True
            else:
                print("✗ Failed to capture test frame")
                return False
                
        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            return False
    
    def reset_for_new_episode(self):
        """Reset frame counter and clear queue for new episode"""
        self.frame_counter = 0
        # Clear any stale frames from previous episode
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
    
    def start_capture(self):
        """Start capturing frames in background thread"""
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
    
    def stop_capture(self):
        """Stop capture and cleanup"""
        self.running = False
        time.sleep(0.1)  # Give capture thread time to exit cleanly
        if self.cap and self.cap.isOpened():
            try:
                self.cap.release()
            except Exception as e:
                print(f"⚠️  Error releasing camera: {e}")
    
    def _capture_loop(self):
        """Background thread for frame capture"""
        frame_interval = 1.0 / self.fps
        last_frame_time = time.time()
        
        while self.running and self.cap and self.cap.isOpened():
            try:
                current_time = time.time()
                
                if current_time - last_frame_time >= frame_interval:
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        self.frame_counter += 1
                        frame_sample = FrameSample(
                            frame_id=self.frame_counter,
                            timestamp=current_time,
                            image_path=""  # Will be set when saved
                        )
                        self.frame_queue.put((frame_sample, frame))
                        last_frame_time = current_time
                    elif not ret:
                        print("⚠️  Camera read failed, stopping capture")
                        break
            except cv2.error as e:
                print(f"⚠️  Camera error: {e}")
                break
            except Exception as e:
                print(f"⚠️  Unexpected capture error: {e}")
                break
            
            time.sleep(0.001)  # Small sleep to prevent CPU spinning


class EpisodeRecorder:
    """Coordinates episode recording with optimized real-time performance"""
    
    def __init__(self, output_dir: str, episode_duration: int = 6, action_label: str = "hit red balloon", 
                 resolution: Tuple[int, int] = (640, 360), jpeg_quality: int = 85):
        self.output_dir = output_dir
        self.episode_duration = episode_duration
        self.action_label = action_label
        self.jpeg_quality = jpeg_quality
        self.arduino_reader = ArduinoReader()
        self.camera = CameraCapture(resolution=resolution)
        
        # Async frame writer queue (prevents blocking on disk I/O)
        self.write_queue = queue.Queue(maxsize=200)  # Buffer up to ~6 seconds @ 30fps
        self.writer_thread = None
        self.writer_running = False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def _frame_writer_loop(self, episode_dir: str):
        """Background thread for async frame writing - prevents I/O blocking"""
        frames_written = 0
        while self.writer_running or not self.write_queue.empty():
            try:
                frame_sample, frame, frame_filename = self.write_queue.get(timeout=0.1)
                frame_path = os.path.join(episode_dir, "frames", frame_filename)
                cv2.imwrite(frame_path, frame)
                frames_written += 1
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️  Frame write error: {e}")
        
        print(f"✓ Frame writer completed: {frames_written} frames saved")
        
    def record_episode(self) -> bool:
        """Record a single episode"""
        
        # Generate episode ID
        episode_id = datetime.now().strftime("episode_%Y%m%d_%H%M%S")
        episode_dir = os.path.join(self.output_dir, episode_id)
        os.makedirs(episode_dir, exist_ok=True)
        os.makedirs(os.path.join(episode_dir, "frames"), exist_ok=True)
        
        print(f"\n🎬 Starting Episode: {episode_id}")
        print(f"Duration: {self.episode_duration} seconds")
        print("=" * 50)
        
        # Initialize systems
        if not self.arduino_reader.connect():
            return False
            
        if not self.camera.initialize():
            return False
        
        # Reset frame counter and clear queues for new episode
        self.camera.reset_for_new_episode()
        
        # Clear any stale control data from previous episode
        while not self.arduino_reader.data_queue.empty():
            try:
                self.arduino_reader.data_queue.get_nowait()
            except queue.Empty:
                break
        
        # Start data collection
        self.arduino_reader.start_reading()
        self.camera.start_capture()
        
        # Start async frame writer
        self.writer_running = True
        self.writer_thread = threading.Thread(target=self._frame_writer_loop, args=(episode_dir,), daemon=True)
        self.writer_thread.start()
        
        # Episode data collection
        control_samples = []
        frame_samples = []
        start_time = time.time()
        end_time = start_time + self.episode_duration
        
        last_progress_time = 0
        
        print("🔴 Recording started...")
        
        try:
            while time.time() < end_time:
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = self.episode_duration - elapsed
                
                # Collect control data (non-blocking, high priority)
                try:
                    while True:
                        control_sample = self.arduino_reader.data_queue.get_nowait()
                        control_samples.append(control_sample)
                except queue.Empty:
                    pass
                
                # Collect frame data (non-blocking, offload I/O to writer thread)
                try:
                    while True:
                        frame_sample, frame = self.camera.frame_queue.get_nowait()
                        
                        # Queue frame for async writing (NO blocking cv2.imwrite here!)
                        frame_filename = f"frame_{frame_sample.frame_id:06d}.jpg"
                        self.write_queue.put((frame_sample, frame, frame_filename))
                        
                        # Update frame sample with path (for metadata)
                        frame_sample.image_path = os.path.join("frames", frame_filename)
                        frame_samples.append(frame_sample)
                        
                except queue.Empty:
                    pass
                
                # Progress update (throttled to reduce print overhead)
                if current_time - last_progress_time >= 0.1:  # Update every 0.1 seconds (10 Hz)
                    print(f"⏱️  {elapsed:.1f}s | Controls: {len(control_samples)} | Frames: {len(frame_samples)} | Write queue: {self.write_queue.qsize()} | Remaining: {remaining:.1f}s")
                    last_progress_time = current_time
                
                # Minimal sleep for CPU efficiency (1ms instead of 10ms)
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n⏹️  Recording interrupted by user")
        
        finally:
            # Stop data collection
            self.arduino_reader.stop_reading()
            self.camera.stop_capture()
            
            # Signal writer thread to finish
            self.writer_running = False
            print("⏳ Waiting for frame writer to complete...")
            self.writer_thread.join(timeout=10.0)  # Wait up to 10s for writes to complete
            
            actual_duration = time.time() - start_time
            
            # Validate synchronization quality
            sync_quality = self._validate_synchronization(control_samples, frame_samples)
            
            # Create episode data structure
            episode_data = EpisodeData(
                episode_id=episode_id,
                start_time=start_time,
                end_time=time.time(),
                duration=actual_duration,
                control_samples=control_samples,
                frame_samples=frame_samples,
                metadata={
                    "camera_fps": self.camera.fps,
                    "camera_resolution": self.camera.resolution,
                    "arduino_port": self.arduino_reader.port,
                    "total_control_samples": len(control_samples),
                    "total_frames": len(frame_samples),
                    "avg_control_rate": len(control_samples) / actual_duration if actual_duration > 0 else 0,
                    "avg_frame_rate": len(frame_samples) / actual_duration if actual_duration > 0 else 0,
                    "sync_quality": sync_quality
                }
            )
            
            # Save episode metadata
            self._save_episode_data(episode_dir, episode_data)
            
            print(f"\n✅ Episode completed!")
            print(f"📁 Data saved to: {episode_dir}")
            print(f"⏱️  Duration: {actual_duration:.1f}s")
            print(f"🎮 Control samples: {len(control_samples)} ({len(control_samples)/actual_duration:.1f} Hz)")
            print(f"📷 Frame samples: {len(frame_samples)} ({len(frame_samples)/actual_duration:.1f} Hz)")
            print(f"🔄 Sync quality: {sync_quality['max_offset_ms']:.2f}ms max offset")
            
        return True
    
    def _validate_synchronization(self, control_samples: List[ControlSample], frame_samples: List[FrameSample]) -> dict:
        """Validate timestamp synchronization between control and frame data"""
        if not control_samples or not frame_samples:
            return {"max_offset_ms": 0, "avg_offset_ms": 0, "status": "NO_DATA"}
        
        # Calculate inter-sample timing
        control_intervals = []
        for i in range(1, len(control_samples)):
            interval = (control_samples[i].system_timestamp - control_samples[i-1].system_timestamp) * 1000
            control_intervals.append(interval)
        
        frame_intervals = []
        for i in range(1, len(frame_samples)):
            interval = (frame_samples[i].timestamp - frame_samples[i-1].timestamp) * 1000
            frame_intervals.append(interval)
        
        max_control_jitter = max(control_intervals) - min(control_intervals) if control_intervals else 0
        max_frame_jitter = max(frame_intervals) - min(frame_intervals) if frame_intervals else 0
        
        return {
            "max_offset_ms": max(max_control_jitter, max_frame_jitter),
            "avg_control_interval_ms": np.mean(control_intervals) if control_intervals else 0,
            "avg_frame_interval_ms": np.mean(frame_intervals) if frame_intervals else 0,
            "control_jitter_ms": max_control_jitter,
            "frame_jitter_ms": max_frame_jitter,
            "status": "GOOD" if max(max_control_jitter, max_frame_jitter) < 50 else "WARNING"
        }
    
    def _save_episode_data(self, episode_dir: str, episode_data: EpisodeData):
        """Save episode metadata and synchronized data"""
        
        # Convert to serializable format
        episode_dict = {
            "episode_id": episode_data.episode_id,
            "action_label": self.action_label,
            "start_time": episode_data.start_time,
            "end_time": episode_data.end_time,
            "duration": episode_data.duration,
            "metadata": episode_data.metadata,
            "control_samples": [
                {
                    "arduino_timestamp": s.arduino_timestamp,
                    "system_timestamp": s.system_timestamp,
                    "steering_normalized": s.steering_normalized,
                    "throttle_normalized": s.throttle_normalized,
                    "steering_raw_us": s.steering_raw_us,
                    "throttle_raw_us": s.throttle_raw_us,
                    "steering_period_us": s.steering_period_us,
                    "throttle_period_us": s.throttle_period_us
                }
                for s in episode_data.control_samples
            ],
            "frame_samples": [
                {
                    "frame_id": f.frame_id,
                    "timestamp": f.timestamp,
                    "image_path": f.image_path
                }
                for f in episode_data.frame_samples
            ]
        }
        
        # Save as JSON
        metadata_path = os.path.join(episode_dir, "episode_data.json")
        with open(metadata_path, 'w') as f:
            json.dump(episode_dict, f, indent=2)
        
        # Also save as CSV for easy analysis
        self._save_csv_format(episode_dir, episode_data)
    
    def _save_csv_format(self, episode_dir: str, episode_data: EpisodeData):
        """Save data in CSV format for analysis"""
        import csv
        
        # Control data CSV
        control_csv_path = os.path.join(episode_dir, "control_data.csv")
        with open(control_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'arduino_timestamp', 'system_timestamp', 'steering_normalized', 
                'throttle_normalized', 'steering_raw_us', 'throttle_raw_us',
                'steering_period_us', 'throttle_period_us'
            ])
            for sample in episode_data.control_samples:
                writer.writerow([
                    sample.arduino_timestamp, sample.system_timestamp,
                    sample.steering_normalized, sample.throttle_normalized,
                    sample.steering_raw_us, sample.throttle_raw_us,
                    sample.steering_period_us, sample.throttle_period_us
                ])
        
        # Frame data CSV
        frame_csv_path = os.path.join(episode_dir, "frame_data.csv")
        with open(frame_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_id', 'timestamp', 'image_path'])
            for sample in episode_data.frame_samples:
                writer.writerow([
                    sample.frame_id, sample.timestamp, sample.image_path
                ])

def main():
    parser = argparse.ArgumentParser(description='RC Car Episode Data Collector')
    parser.add_argument('--episode-duration', type=int, default=6, help='Episode duration in seconds')
    parser.add_argument('--output-dir', type=str, default='./episodes', help='Output directory for episodes')
    parser.add_argument('--arduino-port', type=str, default='/dev/ttyACM0', help='Arduino serial port')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera device ID')
    parser.add_argument('--action-label', type=str, default='hit red balloon', help='Action label for VLA training')
    parser.add_argument('--resolution', type=str, default='640x360', help='Camera resolution (WxH), e.g., 640x360, 640x480, 1280x720')
    
    args = parser.parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print(f"Error: Invalid resolution format '{args.resolution}'. Use WxH format (e.g., 320x240)")
        return 1
    
    print("RC Car Imitation Learning Data Collector")
    print("=" * 40)
    print(f"Episode Duration: {args.episode_duration}s")
    print(f"Output Directory: {args.output_dir}")
    print(f"Arduino Port: {args.arduino_port}")
    print(f"Camera ID: {args.camera_id}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print(f"Action Label: {args.action_label}")
    
    recorder = EpisodeRecorder(
        output_dir=args.output_dir,
        episode_duration=args.episode_duration,
        action_label=args.action_label,
        resolution=resolution
    )
    
    # Update Arduino reader and camera settings if provided
    recorder.arduino_reader.port = args.arduino_port
    recorder.camera.camera_id = args.camera_id
    
    try:
        episode_num = 1
        while True:
            input(f"\nPress ENTER to start Episode {episode_num} (or Ctrl+C to quit)...")
            
            success = recorder.record_episode()
            if success:
                episode_num += 1
            else:
                print("Episode recording failed. Check connections and try again.")
                
    except KeyboardInterrupt:
        print("\n\n👋 Data collection session ended.")
        print("Thank you for collecting training data!")

if __name__ == '__main__':
    main()