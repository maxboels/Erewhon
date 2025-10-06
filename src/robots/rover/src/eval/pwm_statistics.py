#!/usr/bin/env python3
"""
Quick PWM Statistics Summary
Shows distribution of steering and throttle values across all episodes
"""

import csv
from pathlib import Path
from collections import defaultdict

def analyze_episodes(episodes_dir):
    episodes_path = Path(episodes_dir)
    episode_dirs = sorted([
        d for d in episodes_path.iterdir() 
        if d.is_dir() and d.name.startswith('episode_')
    ])
    
    all_steering = []
    all_throttle_duty = []
    all_steering_norm = []
    all_throttle_norm = []
    
    print("="*80)
    print("PWM Statistics Summary - All Episodes")
    print("="*80)
    
    for episode_dir in episode_dirs:
        control_csv = episode_dir / 'control_data.csv'
        
        with open(control_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                steering_raw = int(row['steering_raw_us'])
                throttle_raw = int(row['throttle_raw_us'])
                throttle_period = int(row['throttle_period_us'])
                steering_norm = float(row['steering_normalized'])
                throttle_norm = float(row['throttle_normalized'])
                
                # Filter outliers
                if 1000 < steering_raw < 2000:
                    all_steering.append(steering_raw)
                    all_steering_norm.append(steering_norm)
                
                if throttle_period > 0 and throttle_raw > 0:
                    duty = (throttle_raw / throttle_period * 100)
                    if duty < 80:  # Filter impossible values
                        all_throttle_duty.append(duty)
                        all_throttle_norm.append(throttle_norm)
    
    print(f"\nTotal valid samples analyzed: {len(all_steering)}")
    print(f"\nSTEERING (PWM Pulse Width):")
    print(f"  Spec range:     1065 μs (left) → 1491 μs (center) → 1959 μs (right)")
    print(f"  Actual range:   {min(all_steering)} μs → {int(sum(all_steering)/len(all_steering))} μs (avg) → {max(all_steering)} μs")
    print(f"  Normalized:     {min(all_steering_norm):.3f} (left) → {max(all_steering_norm):.3f} (right)")
    
    # Calculate steering distribution
    left_count = sum(1 for s in all_steering_norm if s < -0.1)
    center_count = sum(1 for s in all_steering_norm if -0.1 <= s <= 0.1)
    right_count = sum(1 for s in all_steering_norm if s > 0.1)
    
    print(f"\n  Distribution:")
    print(f"    Left  (<-0.1): {left_count:4d} samples ({left_count/len(all_steering)*100:5.1f}%)")
    print(f"    Center (±0.1): {center_count:4d} samples ({center_count/len(all_steering)*100:5.1f}%)")
    print(f"    Right (>+0.1): {right_count:4d} samples ({right_count/len(all_steering)*100:5.1f}%)")
    
    print(f"\nTHROTTLE (Duty Cycle %):")
    print(f"  Spec range:     0% (stop) → 70% (full)")
    print(f"  Actual range:   {min(all_throttle_duty):.1f}% → {sum(all_throttle_duty)/len(all_throttle_duty):.1f}% (avg) → {max(all_throttle_duty):.1f}%")
    print(f"  Normalized:     {min(all_throttle_norm):.3f} → {max(all_throttle_norm):.3f}")
    print(f"  Range usage:    {max(all_throttle_duty)/70*100:.1f}% of available throttle")
    
    # Throttle distribution
    bins = [0, 5, 10, 15, 20, 25, 100]
    bin_labels = ["0-5%", "5-10%", "10-15%", "15-20%", "20-25%", ">25%"]
    
    print(f"\n  Distribution:")
    for i in range(len(bins)-1):
        count = sum(1 for t in all_throttle_duty if bins[i] <= t < bins[i+1])
        print(f"    {bin_labels[i]:8s}: {count:4d} samples ({count/len(all_throttle_duty)*100:5.1f}%)")
    
    print("\n" + "="*80)
    
    # Check for data quality issues
    print("\nDATA QUALITY ASSESSMENT:")
    
    steering_range_pct = (max(all_steering) - min(all_steering)) / (1959 - 1065) * 100
    print(f"  ✓ Steering range coverage: {steering_range_pct:.1f}% of full servo range")
    
    throttle_range_pct = max(all_throttle_duty) / 70 * 100
    if throttle_range_pct < 50:
        print(f"  ⚠ Throttle range coverage: {throttle_range_pct:.1f}% (consider more aggressive driving)")
    else:
        print(f"  ✓ Throttle range coverage: {throttle_range_pct:.1f}%")
    
    if abs(sum(all_steering_norm)/len(all_steering_norm)) < 0.05:
        print(f"  ✓ Steering balanced: avg = {sum(all_steering_norm)/len(all_steering_norm):.3f} (well centered)")
    else:
        print(f"  ⚠ Steering bias: avg = {sum(all_steering_norm)/len(all_steering_norm):.3f}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        episodes_dir = sys.argv[1]
    else:
        script_dir = Path(__file__).parent.parent.parent
        episodes_dir = script_dir / "episodes"
    
    analyze_episodes(episodes_dir)
