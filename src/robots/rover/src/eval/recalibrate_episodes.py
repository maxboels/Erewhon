#!/usr/bin/env python3
"""
Recalibrate Episode Data
=========================

Apply corrected calibration to existing episode data.
This fixes the steering center offset and throttle neutral offset.

Corrections applied:
1. Steering: Re-center based on measured 6.83% duty cycle (was 7.0%)
2. Throttle: Re-zero based on measured 11.81% duty cycle (was 0.0%)

Usage:
    # Recalibrate all episodes (creates backups)
    python3 recalibrate_episodes.py --data-dir ./episodes

    # Recalibrate specific episode
    python3 recalibrate_episodes.py --episode-dir ./episodes/episode_20251006_220059

    # Dry run (show what would change)
    python3 recalibrate_episodes.py --data-dir ./episodes --dry-run
"""

import argparse
import json
import csv
import shutil
from pathlib import Path
from typing import Tuple

# Original (incorrect) calibration
OLD_STEERING_NEUTRAL = 7.0
OLD_THROTTLE_NEUTRAL = 0.0
OLD_THROTTLE_MAX = 70.0

# Corrected calibration (measured from actual hardware)
NEW_STEERING_NEUTRAL = 6.83
NEW_THROTTLE_NEUTRAL = 11.81
NEW_THROTTLE_MAX = 70.0

def recalculate_steering(raw_us: int, period_us: int) -> float:
    """Recalculate steering with corrected calibration"""
    if raw_us < 50 or period_us < 10000:
        return 0.0
    
    # Calculate duty cycle
    duty_cycle = (raw_us / period_us) * 100.0
    
    # Map with corrected neutral point
    STEERING_MIN_DUTY = 5.0
    STEERING_MAX_DUTY = 9.2
    
    normalized = (duty_cycle - NEW_STEERING_NEUTRAL) / (STEERING_MAX_DUTY - NEW_STEERING_NEUTRAL) * (2.0/2.2)
    return max(-1.0, min(1.0, normalized))

def recalculate_throttle(raw_us: int, period_us: int) -> float:
    """Recalculate throttle with corrected calibration"""
    if raw_us < 50 or period_us < 500:
        return 0.0
    
    # Calculate duty cycle
    duty_cycle = (raw_us / period_us) * 100.0
    
    # Map with corrected neutral and max
    normalized = (duty_cycle - NEW_THROTTLE_NEUTRAL) / (NEW_THROTTLE_MAX - NEW_THROTTLE_NEUTRAL)
    return max(0.0, min(1.0, normalized))

def recalibrate_episode(episode_dir: Path, dry_run: bool = False) -> Tuple[int, int]:
    """Recalibrate a single episode"""
    
    # Check if episode directory exists
    if not episode_dir.exists():
        print(f"✗ Episode directory not found: {episode_dir}")
        return 0, 0
    
    control_csv = episode_dir / "control_data.csv"
    episode_json = episode_dir / "episode_data.json"
    
    if not control_csv.exists() or not episode_json.exists():
        print(f"✗ Missing data files in {episode_dir.name}")
        return 0, 0
    
    print(f"\n📊 Processing: {episode_dir.name}")
    
    # Read CSV data
    rows = []
    with open(control_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    
    # Read JSON data
    with open(episode_json, 'r') as f:
        episode_data = json.load(f)
    
    # Track changes
    steering_changes = 0
    throttle_changes = 0
    
    # Recalibrate CSV
    for row in rows:
        steering_raw = int(row['steering_raw_us'])
        steering_period = int(row['steering_period_us'])
        throttle_raw = int(row['throttle_raw_us'])
        throttle_period = int(row['throttle_period_us'])
        
        old_steering = float(row['steering_normalized'])
        old_throttle = float(row['throttle_normalized'])
        
        new_steering = recalculate_steering(steering_raw, steering_period)
        new_throttle = recalculate_throttle(throttle_raw, throttle_period)
        
        if abs(new_steering - old_steering) > 0.001:
            steering_changes += 1
        if abs(new_throttle - old_throttle) > 0.001:
            throttle_changes += 1
        
        row['steering_normalized'] = f"{new_steering:.4f}"
        row['throttle_normalized'] = f"{new_throttle:.4f}"
    
    # Recalibrate JSON
    for sample in episode_data['control_samples']:
        steering_raw = sample['steering_raw_us']
        steering_period = sample['steering_period_us']
        throttle_raw = sample['throttle_raw_us']
        throttle_period = sample['throttle_period_us']
        
        sample['steering_normalized'] = recalculate_steering(steering_raw, steering_period)
        sample['throttle_normalized'] = recalculate_throttle(throttle_raw, throttle_period)
    
    if dry_run:
        print(f"  [DRY RUN] Would update {steering_changes} steering values, {throttle_changes} throttle values")
        return steering_changes, throttle_changes
    
    # Create backups
    backup_dir = episode_dir / "backup_original_calibration"
    backup_dir.mkdir(exist_ok=True)
    
    shutil.copy2(control_csv, backup_dir / "control_data.csv")
    shutil.copy2(episode_json, backup_dir / "episode_data.json")
    
    # Write corrected data
    with open(control_csv, 'w', newline='') as f:
        fieldnames = ['arduino_timestamp', 'system_timestamp', 'steering_normalized', 
                     'throttle_normalized', 'steering_raw_us', 'throttle_raw_us', 
                     'steering_period_us', 'throttle_period_us']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    with open(episode_json, 'w') as f:
        json.dump(episode_data, f, indent=2)
    
    print(f"  ✓ Updated {steering_changes} steering values, {throttle_changes} throttle values")
    print(f"  ✓ Backups saved to: {backup_dir}")
    
    return steering_changes, throttle_changes

def main():
    parser = argparse.ArgumentParser(description='Recalibrate episode data with corrected calibration')
    parser.add_argument('--data-dir', type=str, help='Directory containing multiple episodes')
    parser.add_argument('--episode-dir', type=str, help='Single episode directory')
    parser.add_argument('--dry-run', action='store_true', help='Show what would change without modifying files')
    
    args = parser.parse_args()
    
    print("Episode Data Recalibration Tool")
    print("=" * 60)
    print(f"Old calibration: Steering neutral = {OLD_STEERING_NEUTRAL}%, Throttle neutral = {OLD_THROTTLE_NEUTRAL}%")
    print(f"New calibration: Steering neutral = {NEW_STEERING_NEUTRAL}%, Throttle neutral = {NEW_THROTTLE_NEUTRAL}%")
    print()
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified")
        print()
    
    episode_dirs = []
    
    if args.episode_dir:
        episode_dirs.append(Path(args.episode_dir))
    elif args.data_dir:
        data_path = Path(args.data_dir)
        episode_dirs = sorted([d for d in data_path.iterdir() 
                              if d.is_dir() and d.name.startswith('episode_')])
    else:
        print("Error: Must specify --data-dir or --episode-dir")
        return
    
    if not episode_dirs:
        print("No episodes found!")
        return
    
    print(f"Found {len(episode_dirs)} episode(s) to process")
    
    total_steering = 0
    total_throttle = 0
    
    for episode_dir in episode_dirs:
        s_changes, t_changes = recalibrate_episode(episode_dir, args.dry_run)
        total_steering += s_changes
        total_throttle += t_changes
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episodes processed: {len(episode_dirs)}")
    print(f"Total steering values updated: {total_steering}")
    print(f"Total throttle values updated: {total_throttle}")
    
    if not args.dry_run:
        print("\n✓ Recalibration complete!")
        print("  Original data backed up to: backup_original_calibration/")
        print("  CSV and JSON files updated with corrected calibration")
    else:
        print("\n💡 Run without --dry-run to apply changes")

if __name__ == '__main__':
    main()
