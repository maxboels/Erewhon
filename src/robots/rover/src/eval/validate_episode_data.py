#!/usr/bin/env python3
"""
Episode Data Validation Script
Checks for logging errors and validates PWM values in recorded episodes
"""

import json
import csv
import os
from pathlib import Path
from collections import defaultdict
import sys

# Expected PWM ranges from ESC/Receiver specifications
STEERING_MIN_US = 1065  # Full left
STEERING_NEUTRAL_US = 1491  # Center
STEERING_MAX_US = 1959  # Full right
STEERING_PERIOD_US = 21316  # ~47 Hz (21.3ms period)

THROTTLE_MIN_DUTY = 0  # Stop
THROTTLE_MAX_DUTY = 70  # Full forward
THROTTLE_PERIOD_US = 1016  # ~985 Hz (1.016ms period)

# Tolerance for variations
STEERING_PERIOD_TOLERANCE = 100  # ±100 us tolerance for period
THROTTLE_PERIOD_TOLERANCE = 50  # ±50 us tolerance for period

class EpisodeValidator:
    def __init__(self, episode_dir):
        self.episode_dir = Path(episode_dir)
        self.episode_id = self.episode_dir.name
        self.errors = []
        self.warnings = []
        self.stats = defaultdict(list)
        
    def validate(self):
        print(f"\n{'='*80}")
        print(f"Validating Episode: {self.episode_id}")
        print(f"{'='*80}\n")
        
        # Check if all required files exist
        self.check_file_structure()
        
        # Load and validate episode data
        self.validate_episode_json()
        
        # Validate control data
        self.validate_control_data()
        
        # Validate frame data
        self.validate_frame_data()
        
        # Print summary
        self.print_summary()
        
        return len(self.errors) == 0
    
    def check_file_structure(self):
        required_files = ['episode_data.json', 'control_data.csv', 'frame_data.csv']
        
        for filename in required_files:
            filepath = self.episode_dir / filename
            if not filepath.exists():
                self.errors.append(f"Missing required file: {filename}")
            else:
                print(f"✓ Found: {filename}")
        
        frames_dir = self.episode_dir / 'frames'
        if not frames_dir.exists():
            self.errors.append("Missing frames directory")
        else:
            frame_count = len(list(frames_dir.glob('*.jpg')))
            print(f"✓ Found frames directory with {frame_count} frames")
    
    def validate_episode_json(self):
        json_path = self.episode_dir / 'episode_data.json'
        if not json_path.exists():
            return
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Check metadata
            metadata = data.get('metadata', {})
            
            # Check for consistency
            total_control = metadata.get('total_control_samples', 0)
            total_frames = metadata.get('total_frames', 0)
            
            print(f"\nEpisode Metadata:")
            print(f"  Duration: {data.get('duration', 0):.2f} seconds")
            print(f"  Control samples: {total_control}")
            print(f"  Frames: {total_frames}")
            print(f"  Avg control rate: {metadata.get('avg_control_rate', 0):.2f} Hz")
            print(f"  Avg frame rate: {metadata.get('avg_frame_rate', 0):.2f} Hz")
            
            # Check control samples in JSON
            control_samples = data.get('control_samples', [])
            if len(control_samples) != total_control:
                self.warnings.append(
                    f"Control samples mismatch: metadata says {total_control}, "
                    f"but found {len(control_samples)}"
                )
                
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in episode_data.json: {e}")
        except Exception as e:
            self.errors.append(f"Error reading episode_data.json: {e}")
    
    def validate_control_data(self):
        csv_path = self.episode_dir / 'control_data.csv'
        if not csv_path.exists():
            return
        
        print(f"\nValidating Control Data...")
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            total_rows = len(rows)
            print(f"  Total control samples: {total_rows}")
            
            # Track anomalies
            timestamp_issues = 0
            steering_outliers = 0
            throttle_outliers = 0
            period_issues = 0
            
            prev_timestamp = None
            
            for i, row in enumerate(rows):
                # Parse values
                arduino_ts = int(row['arduino_timestamp'])
                system_ts = float(row['system_timestamp'])
                steering_norm = float(row['steering_normalized'])
                throttle_norm = float(row['throttle_normalized'])
                steering_raw = int(row['steering_raw_us'])
                throttle_raw = int(row['throttle_raw_us'])
                steering_period = int(row['steering_period_us'])
                throttle_period = int(row['throttle_period_us'])
                
                # Collect stats
                self.stats['steering_raw'].append(steering_raw)
                self.stats['throttle_raw'].append(throttle_raw)
                self.stats['steering_norm'].append(steering_norm)
                self.stats['throttle_norm'].append(throttle_norm)
                self.stats['steering_period'].append(steering_period)
                self.stats['throttle_period'].append(throttle_period)
                
                # Check timestamp ordering
                if prev_timestamp and system_ts < prev_timestamp:
                    timestamp_issues += 1
                    if timestamp_issues <= 3:
                        self.errors.append(
                            f"Row {i+1}: Timestamp out of order: {system_ts} < {prev_timestamp}"
                        )
                prev_timestamp = system_ts
                
                # Check steering range
                if steering_raw > 0:  # Only check non-zero values
                    if steering_raw < STEERING_MIN_US - 50 or steering_raw > STEERING_MAX_US + 50:
                        steering_outliers += 1
                        if steering_outliers <= 3:
                            self.warnings.append(
                                f"Row {i+1}: Steering PWM out of expected range: {steering_raw} μs "
                                f"(expected {STEERING_MIN_US}-{STEERING_MAX_US})"
                            )
                
                # Check throttle range (duty cycle 0-70%)
                if throttle_raw > 0:  # Only check non-zero values
                    duty_cycle = (throttle_raw / throttle_period * 100) if throttle_period > 0 else 0
                    if duty_cycle > THROTTLE_MAX_DUTY + 5:
                        throttle_outliers += 1
                        if throttle_outliers <= 3:
                            self.warnings.append(
                                f"Row {i+1}: Throttle duty cycle out of range: {duty_cycle:.1f}% "
                                f"(expected 0-{THROTTLE_MAX_DUTY}%)"
                            )
                
                # Check steering period (when active)
                if steering_period > 0:
                    expected_period = STEERING_PERIOD_US
                    if abs(steering_period - expected_period) > STEERING_PERIOD_TOLERANCE:
                        period_issues += 1
                        if period_issues <= 3:
                            self.warnings.append(
                                f"Row {i+1}: Steering period unusual: {steering_period} μs "
                                f"(expected ~{expected_period} μs)"
                            )
                
                # Check throttle period (when active)
                if throttle_period > 0:
                    expected_period = THROTTLE_PERIOD_US
                    if abs(throttle_period - expected_period) > THROTTLE_PERIOD_TOLERANCE:
                        period_issues += 1
                        if period_issues <= 3:
                            self.warnings.append(
                                f"Row {i+1}: Throttle period unusual: {throttle_period} μs "
                                f"(expected ~{expected_period} μs)"
                            )
            
            # Print statistics
            print(f"\n  Control Data Statistics:")
            if self.stats['steering_raw']:
                non_zero_steering = [s for s in self.stats['steering_raw'] if s > 0]
                if non_zero_steering:
                    print(f"    Steering (raw): min={min(non_zero_steering)} μs, "
                          f"max={max(non_zero_steering)} μs, "
                          f"avg={sum(non_zero_steering)/len(non_zero_steering):.1f} μs")
                print(f"    Steering (norm): min={min(self.stats['steering_norm']):.3f}, "
                      f"max={max(self.stats['steering_norm']):.3f}, "
                      f"avg={sum(self.stats['steering_norm'])/len(self.stats['steering_norm']):.3f}")
            
            if self.stats['throttle_raw']:
                non_zero_throttle = [t for t in self.stats['throttle_raw'] if t > 0]
                if non_zero_throttle:
                    print(f"    Throttle (raw): min={min(non_zero_throttle)}, "
                          f"max={max(non_zero_throttle)}, "
                          f"avg={sum(non_zero_throttle)/len(non_zero_throttle):.1f}")
                print(f"    Throttle (norm): min={min(self.stats['throttle_norm']):.3f}, "
                      f"max={max(self.stats['throttle_norm']):.3f}, "
                      f"avg={sum(self.stats['throttle_norm'])/len(self.stats['throttle_norm']):.3f}")
            
            # Summary of issues
            if timestamp_issues > 0:
                print(f"\n  ⚠ Found {timestamp_issues} timestamp ordering issues")
            if steering_outliers > 0:
                print(f"  ⚠ Found {steering_outliers} steering PWM outliers")
            if throttle_outliers > 0:
                print(f"  ⚠ Found {throttle_outliers} throttle duty cycle outliers")
            if period_issues > 0:
                print(f"  ⚠ Found {period_issues} period measurement issues")
                
        except Exception as e:
            self.errors.append(f"Error validating control_data.csv: {e}")
    
    def validate_frame_data(self):
        csv_path = self.episode_dir / 'frame_data.csv'
        if not csv_path.exists():
            return
        
        print(f"\nValidating Frame Data...")
        
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            total_frames = len(rows)
            print(f"  Total frames logged: {total_frames}")
            
            missing_frames = 0
            timestamp_issues = 0
            prev_timestamp = None
            
            for i, row in enumerate(rows):
                frame_id = row['frame_id']
                timestamp = float(row['timestamp'])
                image_path = row['image_path']
                
                # Check if frame file exists
                full_path = self.episode_dir / image_path
                if not full_path.exists():
                    missing_frames += 1
                    if missing_frames <= 3:
                        self.errors.append(f"Missing frame file: {image_path}")
                
                # Check timestamp ordering
                if prev_timestamp and timestamp < prev_timestamp:
                    timestamp_issues += 1
                    if timestamp_issues <= 3:
                        self.warnings.append(
                            f"Frame {frame_id}: Timestamp out of order: {timestamp} < {prev_timestamp}"
                        )
                prev_timestamp = timestamp
            
            if missing_frames > 0:
                print(f"  ✗ Missing {missing_frames} frame files")
            else:
                print(f"  ✓ All frame files present")
            
            if timestamp_issues > 0:
                print(f"  ⚠ Found {timestamp_issues} timestamp ordering issues")
                
        except Exception as e:
            self.errors.append(f"Error validating frame_data.csv: {e}")
    
    def print_summary(self):
        print(f"\n{'='*80}")
        print(f"Validation Summary for {self.episode_id}")
        print(f"{'='*80}\n")
        
        if not self.errors and not self.warnings:
            print("✓ All checks passed! Episode data looks good for training.")
        else:
            if self.errors:
                print(f"ERRORS ({len(self.errors)}):")
                for error in self.errors[:10]:  # Show first 10
                    print(f"  ✗ {error}")
                if len(self.errors) > 10:
                    print(f"  ... and {len(self.errors) - 10} more errors")
            
            if self.warnings:
                print(f"\nWARNINGS ({len(self.warnings)}):")
                for warning in self.warnings[:10]:  # Show first 10
                    print(f"  ⚠ {warning}")
                if len(self.warnings) > 10:
                    print(f"  ... and {len(self.warnings) - 10} more warnings")
        
        print()

def validate_all_episodes(episodes_base_dir):
    """Validate all episodes in the episodes directory"""
    episodes_dir = Path(episodes_base_dir)
    
    if not episodes_dir.exists():
        print(f"Episodes directory not found: {episodes_dir}")
        return False
    
    # Find all episode directories
    episode_dirs = sorted([d for d in episodes_dir.iterdir() if d.is_dir() and d.name.startswith('episode_')])
    
    print(f"Found {len(episode_dirs)} episodes to validate\n")
    
    all_valid = True
    results = []
    
    for episode_dir in episode_dirs:
        validator = EpisodeValidator(episode_dir)
        is_valid = validator.validate()
        results.append((episode_dir.name, is_valid, len(validator.errors), len(validator.warnings)))
        all_valid = all_valid and is_valid
    
    # Print overall summary
    print(f"\n{'='*80}")
    print("Overall Validation Summary")
    print(f"{'='*80}\n")
    
    for episode_name, is_valid, error_count, warning_count in results:
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{status} | {episode_name} | Errors: {error_count}, Warnings: {warning_count}")
    
    return all_valid

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Validate specific episode
        episode_path = sys.argv[1]
        validator = EpisodeValidator(episode_path)
        validator.validate()
    else:
        # Validate all episodes in default location
        script_dir = Path(__file__).parent.parent.parent
        episodes_dir = script_dir / "episodes"
        validate_all_episodes(episodes_dir)
