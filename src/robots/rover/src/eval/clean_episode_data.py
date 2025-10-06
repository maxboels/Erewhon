#!/usr/bin/env python3
"""
Episode Data Cleaner
Removes anomalous data points from recorded episodes based on validation criteria
"""

import json
import csv
import shutil
from pathlib import Path
from datetime import datetime

# Expected PWM ranges
STEERING_MIN_US = 1065
STEERING_MAX_US = 1959
THROTTLE_MAX_DUTY = 70
THROTTLE_PERIOD_US = 1016
STEERING_PERIOD_US = 21316

# Tolerances
STEERING_OUTLIER_MARGIN = 200  # Allow ±200 μs beyond expected range
THROTTLE_DUTY_MAX = 80  # Allow up to 80% duty cycle
THROTTLE_PERIOD_MIN = 900  # Minimum valid throttle period
THROTTLE_PERIOD_MAX = 2000  # Maximum valid throttle period

def is_steering_valid(steering_raw_us):
    """Check if steering PWM value is valid"""
    if steering_raw_us == 0:
        return True  # Zero is valid (no signal)
    return (STEERING_MIN_US - STEERING_OUTLIER_MARGIN <= steering_raw_us <= 
            STEERING_MAX_US + STEERING_OUTLIER_MARGIN)

def is_throttle_valid(throttle_raw, throttle_period_us):
    """Check if throttle values are valid"""
    if throttle_raw == 0 or throttle_period_us == 0:
        return True  # Zero is valid (no signal)
    
    # Check period range
    if throttle_period_us < THROTTLE_PERIOD_MIN or throttle_period_us > THROTTLE_PERIOD_MAX:
        return False
    
    # Check duty cycle
    duty_cycle = (throttle_raw / throttle_period_us * 100) if throttle_period_us > 0 else 0
    return duty_cycle <= THROTTLE_DUTY_MAX

def clean_episode(episode_dir, backup=True):
    """Clean anomalous data from an episode"""
    episode_path = Path(episode_dir)
    episode_id = episode_path.name
    
    print(f"\nCleaning episode: {episode_id}")
    
    # Create backup if requested
    if backup:
        backup_dir = episode_path.parent / f"{episode_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Creating backup at: {backup_dir}")
        shutil.copytree(episode_path, backup_dir)
    
    # Read control data
    control_csv = episode_path / 'control_data.csv'
    
    with open(control_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    original_count = len(rows)
    cleaned_rows = []
    removed_count = 0
    
    for row in rows:
        steering_raw = int(row['steering_raw_us'])
        throttle_raw = int(row['throttle_raw_us'])
        throttle_period = int(row['throttle_period_us'])
        
        # Check validity
        steering_ok = is_steering_valid(steering_raw)
        throttle_ok = is_throttle_valid(throttle_raw, throttle_period)
        
        if steering_ok and throttle_ok:
            cleaned_rows.append(row)
        else:
            removed_count += 1
            reasons = []
            if not steering_ok:
                reasons.append(f"steering={steering_raw}μs")
            if not throttle_ok:
                duty = (throttle_raw / throttle_period * 100) if throttle_period > 0 else 0
                reasons.append(f"throttle={throttle_raw}/{throttle_period}={duty:.1f}%")
            print(f"  Removing row: {', '.join(reasons)}")
    
    # Write cleaned data
    if removed_count > 0:
        with open(control_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(cleaned_rows)
        
        # Update episode_data.json
        json_path = episode_path / 'episode_data.json'
        with open(json_path, 'r') as f:
            episode_data = json.load(f)
        
        # Update metadata
        episode_data['metadata']['total_control_samples'] = len(cleaned_rows)
        
        # Filter control samples in JSON
        original_samples = episode_data['control_samples']
        if len(original_samples) == original_count:
            # Map cleaned rows to JSON samples by index
            valid_indices = set()
            for i, row in enumerate(rows):
                if row in cleaned_rows:
                    valid_indices.add(i)
            
            episode_data['control_samples'] = [
                sample for i, sample in enumerate(original_samples) 
                if i in valid_indices
            ]
        
        with open(json_path, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        print(f"  Cleaned: Removed {removed_count}/{original_count} samples ({removed_count/original_count*100:.1f}%)")
        print(f"  Remaining: {len(cleaned_rows)} valid samples")
    else:
        print(f"  No anomalies found - episode is clean!")
    
    return removed_count

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Clean specific episode
        episode_path = sys.argv[1]
        clean_episode(episode_path, backup=True)
    else:
        # Clean all episodes
        script_dir = Path(__file__).parent.parent.parent
        episodes_dir = script_dir / "episodes"
        
        episode_dirs = sorted([
            d for d in episodes_dir.iterdir() 
            if d.is_dir() and d.name.startswith('episode_')
        ])
        
        print(f"Found {len(episode_dirs)} episodes to clean\n")
        
        total_removed = 0
        for episode_dir in episode_dirs:
            removed = clean_episode(episode_dir, backup=True)
            total_removed += removed
        
        print(f"\n{'='*80}")
        print(f"Cleaning complete: Removed {total_removed} anomalous samples total")
        print(f"{'='*80}")
