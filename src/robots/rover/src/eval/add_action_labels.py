#!/usr/bin/env python3
"""
Add action labels to episode data for VLA training
"""

import json
from pathlib import Path
import sys

def add_action_label(episode_json_path, action_label="hit red balloon"):
    """Add action label to episode data"""
    with open(episode_json_path, 'r') as f:
        data = json.load(f)
    
    # Add action label at the top level
    data['action_label'] = action_label
    
    # Also add to metadata for clarity
    if 'metadata' not in data:
        data['metadata'] = {}
    data['metadata']['action_label'] = action_label
    
    # Write back with nice formatting
    with open(episode_json_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Added action label '{action_label}' to {episode_json_path.name}")

def add_labels_to_all_episodes(episodes_dir, action_label="hit red balloon"):
    """Add action labels to all episodes"""
    episodes_path = Path(episodes_dir)
    
    episode_dirs = sorted([
        d for d in episodes_path.iterdir() 
        if d.is_dir() and d.name.startswith('episode_')
    ])
    
    print(f"Adding action label '{action_label}' to {len(episode_dirs)} episodes...\n")
    
    for episode_dir in episode_dirs:
        json_path = episode_dir / 'episode_data.json'
        if json_path.exists():
            add_action_label(json_path, action_label)
        else:
            print(f"✗ Missing: {json_path}")
    
    print(f"\n✓ Complete! Added action labels to all episodes.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom action label provided
        action_label = ' '.join(sys.argv[1:])
    else:
        # Default action label
        action_label = "hit red balloon"
    
    script_dir = Path(__file__).parent.parent.parent
    episodes_dir = script_dir / "episodes"
    
    add_labels_to_all_episodes(episodes_dir, action_label)
