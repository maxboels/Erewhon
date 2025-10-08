#!/usr/bin/env python3
"""
Real-Time PWM Monitor
=====================

Live visualization of PWM signals from Arduino showing:
- Raw PWM values (pulse width in microseconds)
- Normalized values (-1.0 to 1.0 for steering, 0.0 to 1.0 for throttle)
- Real-time graphs with history
- Current calibration constants
- Statistics and quality indicators

This tool helps validate calibration and test data collection in real-time.

Usage:
    python3 realtime_pwm_monitor.py --arduino-port /dev/ttyACM0
    
Controls:
    - Press 'q' to quit
    - Press 'r' to reset graphs
    - Press 's' to save screenshot
    - Press 'c' to show calibration info
"""

import serial
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from datetime import datetime
import sys

class PWMMonitor:
    """Real-time PWM signal monitor with live visualization"""
    
    def __init__(self, port='/dev/ttyACM0', baudrate=115200, history_size=300):
        self.port = port
        self.baudrate = baudrate
        self.history_size = history_size  # ~10 seconds at 30Hz
        
        # Data buffers (deque for efficient append/pop)
        self.timestamps = deque(maxlen=history_size)
        self.steering_raw = deque(maxlen=history_size)
        self.steering_norm = deque(maxlen=history_size)
        self.throttle_raw = deque(maxlen=history_size)
        self.throttle_norm = deque(maxlen=history_size)
        self.steering_period = deque(maxlen=history_size)
        self.throttle_period = deque(maxlen=history_size)
        
        # Statistics
        self.sample_count = 0
        self.start_time = None
        self.last_update_time = 0
        self.update_rate = 0
        
        # Serial connection
        self.serial_conn = None
        self.running = False
        
        # Calibration constants (from Arduino code)
        self.STEERING_MIN_DUTY = 5.0
        self.STEERING_NEUTRAL_DUTY = 6.83
        self.STEERING_MAX_DUTY = 9.2
        self.THROTTLE_MIN_DUTY = 11.81
        self.THROTTLE_MAX_DUTY = 70.0
        
    def connect(self):
        """Establish serial connection with Arduino"""
        try:
            print(f"Connecting to Arduino on {self.port}...")
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Allow Arduino to reset
            
            # Wait for Arduino ready signal
            while True:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if "ARDUINO_READY" in line:
                    print("✓ Arduino connected and ready!")
                    break
                    
            self.start_time = time.time()
            return True
            
        except serial.SerialException as e:
            print(f"✗ Failed to connect to Arduino: {e}")
            return False
    
    def read_data(self):
        """Read and parse one data sample from Arduino"""
        if not self.serial_conn:
            return False
            
        try:
            line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
            
            if line.startswith("DATA,"):
                parts = line.split(',')
                if len(parts) == 8:
                    # Parse: DATA,timestamp,steer_norm,throttle_norm,steer_raw,throttle_raw,steer_period,throttle_period
                    current_time = time.time() - self.start_time
                    
                    steer_norm = float(parts[2])
                    throttle_norm = float(parts[3])
                    steer_raw = int(parts[4])
                    throttle_raw = int(parts[5])
                    steer_period = int(parts[6])
                    throttle_period = int(parts[7])
                    
                    # Update buffers
                    self.timestamps.append(current_time)
                    self.steering_norm.append(steer_norm)
                    self.throttle_norm.append(throttle_norm)
                    self.steering_raw.append(steer_raw)
                    self.throttle_raw.append(throttle_raw)
                    self.steering_period.append(steer_period)
                    self.throttle_period.append(throttle_period)
                    
                    self.sample_count += 1
                    
                    # Calculate update rate
                    if current_time - self.last_update_time > 0:
                        self.update_rate = 1.0 / (current_time - self.last_update_time)
                    self.last_update_time = current_time
                    
                    return True
                    
        except Exception as e:
            print(f"Error reading data: {e}")
            
        return False
    
    def get_statistics(self):
        """Calculate current statistics"""
        stats = {
            'samples': self.sample_count,
            'duration': time.time() - self.start_time if self.start_time else 0,
            'rate': self.update_rate,
            'steering_raw_avg': np.mean(self.steering_raw) if self.steering_raw else 0,
            'steering_raw_min': np.min(self.steering_raw) if self.steering_raw else 0,
            'steering_raw_max': np.max(self.steering_raw) if self.steering_raw else 0,
            'steering_norm_avg': np.mean(self.steering_norm) if self.steering_norm else 0,
            'steering_norm_min': np.min(self.steering_norm) if self.steering_norm else 0,
            'steering_norm_max': np.max(self.steering_norm) if self.steering_norm else 0,
            'throttle_raw_avg': np.mean(self.throttle_raw) if self.throttle_raw else 0,
            'throttle_raw_min': np.min(self.throttle_raw) if self.throttle_raw else 0,
            'throttle_raw_max': np.max(self.throttle_raw) if self.throttle_raw else 0,
            'throttle_norm_avg': np.mean(self.throttle_norm) if self.throttle_norm else 0,
            'throttle_norm_min': np.min(self.throttle_norm) if self.throttle_norm else 0,
            'throttle_norm_max': np.max(self.throttle_norm) if self.throttle_norm else 0,
            'steering_period_avg': np.mean(self.steering_period) if self.steering_period else 0,
            'throttle_period_avg': np.mean(self.throttle_period) if self.throttle_period else 0,
        }
        return stats
    
    def calculate_duty_cycle(self, raw_us, period_us):
        """Calculate duty cycle percentage"""
        if period_us > 0:
            return (raw_us / period_us) * 100.0
        return 0.0
    
    def reset(self):
        """Reset all data buffers"""
        self.timestamps.clear()
        self.steering_raw.clear()
        self.steering_norm.clear()
        self.throttle_raw.clear()
        self.throttle_norm.clear()
        self.steering_period.clear()
        self.throttle_period.clear()
        self.sample_count = 0
        self.start_time = time.time()
        print("Data buffers reset!")
    
    def close(self):
        """Close serial connection"""
        if self.serial_conn:
            self.serial_conn.close()
            print("Serial connection closed.")


class RealtimeVisualizer:
    """Real-time visualization using matplotlib"""
    
    def __init__(self, monitor):
        self.monitor = monitor
        
        # Create figure with subplots
        plt.style.use('seaborn-v0_8-darkgrid')
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title('Real-Time PWM Monitor')
        
        # Create grid layout (3 rows, 2 columns)
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # Steering plots
        self.ax_steering_raw = self.fig.add_subplot(gs[0, 0])
        self.ax_steering_norm = self.fig.add_subplot(gs[1, 0])
        
        # Throttle plots
        self.ax_throttle_raw = self.fig.add_subplot(gs[0, 1])
        self.ax_throttle_norm = self.fig.add_subplot(gs[1, 1])
        
        # Statistics display
        self.ax_stats = self.fig.add_subplot(gs[2, :])
        self.ax_stats.axis('off')
        
        # Initialize lines
        self.line_steer_raw, = self.ax_steering_raw.plot([], [], 'b-', linewidth=2, label='Raw PWM')
        self.line_steer_norm, = self.ax_steering_norm.plot([], [], 'g-', linewidth=2, label='Normalized')
        self.line_throttle_raw, = self.ax_throttle_raw.plot([], [], 'r-', linewidth=2, label='Raw PWM')
        self.line_throttle_norm, = self.ax_throttle_norm.plot([], [], 'm-', linewidth=2, label='Normalized')
        
        # Configure axes
        self._setup_axes()
        
        # Text for statistics
        self.stats_text = self.ax_stats.text(0.05, 0.5, '', fontsize=10, 
                                             verticalalignment='center', 
                                             family='monospace')
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def _setup_axes(self):
        """Configure plot axes"""
        # Steering Raw
        self.ax_steering_raw.set_title('Steering - Raw PWM (μs)', fontsize=12, fontweight='bold')
        self.ax_steering_raw.set_xlabel('Time (s)')
        self.ax_steering_raw.set_ylabel('Pulse Width (μs)')
        self.ax_steering_raw.set_ylim(900, 2100)
        self.ax_steering_raw.grid(True, alpha=0.3)
        self.ax_steering_raw.legend(loc='upper right')
        
        # Add reference lines for steering calibration
        self.ax_steering_raw.axhline(y=1456, color='gray', linestyle='--', alpha=0.5, label='Center (1456μs)')
        
        # Steering Normalized
        self.ax_steering_norm.set_title('Steering - Normalized', fontsize=12, fontweight='bold')
        self.ax_steering_norm.set_xlabel('Time (s)')
        self.ax_steering_norm.set_ylabel('Normalized Value')
        self.ax_steering_norm.set_ylim(-1.2, 1.2)
        self.ax_steering_norm.grid(True, alpha=0.3)
        self.ax_steering_norm.legend(loc='upper right')
        
        # Add reference lines
        self.ax_steering_norm.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Center (0.0)')
        self.ax_steering_norm.axhline(y=-1, color='red', linestyle=':', alpha=0.3)
        self.ax_steering_norm.axhline(y=1, color='red', linestyle=':', alpha=0.3)
        
        # Throttle Raw
        self.ax_throttle_raw.set_title('Throttle - Raw PWM (μs)', fontsize=12, fontweight='bold')
        self.ax_throttle_raw.set_xlabel('Time (s)')
        self.ax_throttle_raw.set_ylabel('Pulse Width (μs)')
        self.ax_throttle_raw.set_ylim(0, 1000)
        self.ax_throttle_raw.grid(True, alpha=0.3)
        self.ax_throttle_raw.legend(loc='upper right')
        
        # Add reference line for stopped throttle
        self.ax_throttle_raw.axhline(y=120, color='gray', linestyle='--', alpha=0.5, label='Stopped (120μs)')
        
        # Throttle Normalized
        self.ax_throttle_norm.set_title('Throttle - Normalized', fontsize=12, fontweight='bold')
        self.ax_throttle_norm.set_xlabel('Time (s)')
        self.ax_throttle_norm.set_ylabel('Normalized Value')
        self.ax_throttle_norm.set_ylim(-0.1, 1.2)
        self.ax_throttle_norm.grid(True, alpha=0.3)
        self.ax_throttle_norm.legend(loc='upper right')
        
        # Add reference lines
        self.ax_throttle_norm.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Stopped (0.0)')
        self.ax_throttle_norm.axhline(y=1, color='red', linestyle=':', alpha=0.3)
    
    def update(self, frame):
        """Update all plots (called by animation)"""
        # Read new data
        self.monitor.read_data()
        
        if len(self.monitor.timestamps) < 2:
            return self.line_steer_raw, self.line_steer_norm, self.line_throttle_raw, self.line_throttle_norm
        
        # Convert deques to arrays
        times = np.array(self.monitor.timestamps)
        steer_raw = np.array(self.monitor.steering_raw)
        steer_norm = np.array(self.monitor.steering_norm)
        throttle_raw = np.array(self.monitor.throttle_raw)
        throttle_norm = np.array(self.monitor.throttle_norm)
        
        # Update lines
        self.line_steer_raw.set_data(times, steer_raw)
        self.line_steer_norm.set_data(times, steer_norm)
        self.line_throttle_raw.set_data(times, throttle_raw)
        self.line_throttle_norm.set_data(times, throttle_norm)
        
        # Update x-axis limits to show last 10 seconds
        if len(times) > 0:
            x_max = times[-1]
            x_min = max(0, x_max - 10)
            
            self.ax_steering_raw.set_xlim(x_min, x_max)
            self.ax_steering_norm.set_xlim(x_min, x_max)
            self.ax_throttle_raw.set_xlim(x_min, x_max)
            self.ax_throttle_norm.set_xlim(x_min, x_max)
        
        # Update statistics display
        stats = self.monitor.get_statistics()
        
        # Calculate duty cycles for current values
        steer_duty = self.monitor.calculate_duty_cycle(
            stats['steering_raw_avg'], 
            stats['steering_period_avg']
        ) if stats['steering_period_avg'] > 0 else 0
        
        throttle_duty = self.monitor.calculate_duty_cycle(
            stats['throttle_raw_avg'],
            stats['throttle_period_avg']
        ) if stats['throttle_period_avg'] > 0 else 0
        
        stats_text = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║  REAL-TIME PWM MONITOR  │  Samples: {stats['samples']:6d}  │  Duration: {stats['duration']:6.1f}s  │  Rate: {stats['rate']:5.1f} Hz      ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  STEERING                │  Raw (μs)                │  Normalized              │  Period: {stats['steering_period_avg']:7.0f} μs ({stats['steering_period_avg'] and 1e6/stats['steering_period_avg'] or 0:.1f} Hz)  ║
║    Current               │  {stats['steering_raw_avg']:7.0f} μs               │  {stats['steering_norm_avg']:7.3f}              │  Duty Cycle: {steer_duty:5.2f}%                    ║
║    Range                 │  {stats['steering_raw_min']:7.0f} - {stats['steering_raw_max']:7.0f} μs        │  {stats['steering_norm_min']:7.3f} - {stats['steering_norm_max']:7.3f}    │                                           ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  THROTTLE                │  Raw (μs)                │  Normalized              │  Period: {stats['throttle_period_avg']:7.0f} μs ({stats['throttle_period_avg'] and 1e6/stats['throttle_period_avg'] or 0:.1f} Hz) ║
║    Current               │  {stats['throttle_raw_avg']:7.0f} μs               │  {stats['throttle_norm_avg']:7.3f}              │  Duty Cycle: {throttle_duty:5.2f}%                    ║
║    Range                 │  {stats['throttle_raw_min']:7.0f} - {stats['throttle_raw_max']:7.0f} μs        │  {stats['throttle_norm_min']:7.3f} - {stats['throttle_norm_max']:7.3f}    │                                           ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  CALIBRATION             │  Steering: {self.monitor.STEERING_MIN_DUTY}% (left) | {self.monitor.STEERING_NEUTRAL_DUTY}% (center) | {self.monitor.STEERING_MAX_DUTY}% (right)                                           ║
║                          │  Throttle: {self.monitor.THROTTLE_MIN_DUTY}% (stopped) | {self.monitor.THROTTLE_MAX_DUTY}% (full)                                                      ║
╠════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣
║  CONTROLS: [Q]uit  │  [R]eset  │  [S]ave Screenshot  │  [C]alibration Info                                           ║
╚════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
        """
        
        self.stats_text.set_text(stats_text)
        
        return self.line_steer_raw, self.line_steer_norm, self.line_throttle_raw, self.line_throttle_norm
    
    def on_key_press(self, event):
        """Handle keyboard events"""
        if event.key == 'q':
            print("\nShutting down...")
            plt.close(self.fig)
            self.monitor.close()
            sys.exit(0)
            
        elif event.key == 'r':
            self.monitor.reset()
            
        elif event.key == 's':
            filename = f"pwm_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.fig.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Screenshot saved: {filename}")
            
        elif event.key == 'c':
            self.print_calibration_info()
    
    def print_calibration_info(self):
        """Print detailed calibration information"""
        print("\n" + "="*60)
        print("CALIBRATION INFORMATION")
        print("="*60)
        print("\nSteering Calibration:")
        print(f"  Full Left:   {self.monitor.STEERING_MIN_DUTY}% duty cycle")
        print(f"  Center:      {self.monitor.STEERING_NEUTRAL_DUTY}% duty cycle (1456 μs @ 21316 μs period)")
        print(f"  Full Right:  {self.monitor.STEERING_MAX_DUTY}% duty cycle")
        print("\nThrottle Calibration:")
        print(f"  Stopped:     {self.monitor.THROTTLE_MIN_DUTY}% duty cycle (120 μs @ 1016 μs period)")
        print(f"  Full Speed:  {self.monitor.THROTTLE_MAX_DUTY}% duty cycle")
        print("\nCurrent Observations:")
        stats = self.monitor.get_statistics()
        if stats['samples'] > 0:
            steer_duty = self.monitor.calculate_duty_cycle(
                stats['steering_raw_avg'], 
                stats['steering_period_avg']
            )
            throttle_duty = self.monitor.calculate_duty_cycle(
                stats['throttle_raw_avg'],
                stats['throttle_period_avg']
            )
            print(f"  Steering:    {steer_duty:.2f}% duty ({stats['steering_raw_avg']:.0f} μs)")
            print(f"  Throttle:    {throttle_duty:.2f}% duty ({stats['throttle_raw_avg']:.0f} μs)")
        print("="*60 + "\n")
    
    def run(self):
        """Start the animation"""
        ani = FuncAnimation(self.fig, self.update, interval=33, blit=False, cache_frame_data=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Real-Time PWM Monitor')
    parser.add_argument('--arduino-port', type=str, default='/dev/ttyACM0', 
                       help='Arduino serial port')
    parser.add_argument('--baudrate', type=int, default=115200,
                       help='Serial baudrate')
    parser.add_argument('--history', type=int, default=300,
                       help='Number of samples to keep in history (default: 300 = ~10s @ 30Hz)')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("REAL-TIME PWM MONITOR")
    print("="*60)
    print(f"Port: {args.arduino_port}")
    print(f"Baudrate: {args.baudrate}")
    print(f"History: {args.history} samples")
    print("\nControls:")
    print("  [Q] - Quit")
    print("  [R] - Reset data buffers")
    print("  [S] - Save screenshot")
    print("  [C] - Show calibration info")
    print("="*60 + "\n")
    
    # Create monitor
    monitor = PWMMonitor(port=args.arduino_port, baudrate=args.baudrate, history_size=args.history)
    
    # Connect to Arduino
    if not monitor.connect():
        return 1
    
    # Create visualizer and run
    try:
        viz = RealtimeVisualizer(monitor)
        viz.run()
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    finally:
        monitor.close()
        print("Monitor closed.")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
