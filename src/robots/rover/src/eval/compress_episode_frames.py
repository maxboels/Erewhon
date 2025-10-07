#!/usr/bin/env python3
"""
Compress JPG frames in episode folders to reduce storage size.
This script compresses all frame images in the episodes directory while maintaining visual quality.
"""

import os
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm


def compress_image(input_path: Path, output_path: Path, quality: int = 85, max_size: tuple = None):
    """
    Compress a single JPG image.
    
    Args:
        input_path: Path to input image
        output_path: Path to save compressed image
        quality: JPEG quality (1-100, default 85)
        max_size: Optional tuple (width, height) to resize images
    """
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (removes alpha channel)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize if max_size is specified
            if max_size:
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Save with compression
            img.save(output_path, 'JPEG', quality=quality, optimize=True)
        
        return True
    except Exception as e:
        print(f"Error compressing {input_path}: {e}")
        return False


def get_total_size(directory: Path) -> int:
    """Calculate total size of all JPG files in directory."""
    total = 0
    for jpg_file in directory.rglob("*.jpg"):
        total += jpg_file.stat().st_size
    return total


def compress_episode_frames(episodes_dir: Path, quality: int = 85, max_size: tuple = None, dry_run: bool = False):
    """
    Compress all frames in episode folders.
    
    Args:
        episodes_dir: Path to episodes directory
        quality: JPEG quality (1-100)
        max_size: Optional tuple (width, height) to resize images
        dry_run: If True, only show what would be done
    """
    # Find all frame directories
    frame_dirs = []
    for episode_dir in sorted(episodes_dir.iterdir()):
        if episode_dir.is_dir() and episode_dir.name.startswith('episode_'):
            frames_dir = episode_dir / 'frames'
            if frames_dir.exists():
                frame_dirs.append(frames_dir)
    
    if not frame_dirs:
        print("No episode frame directories found!")
        return
    
    print(f"Found {len(frame_dirs)} episode frame directories")
    
    # Calculate initial size
    print("\nCalculating initial size...")
    initial_size = get_total_size(episodes_dir)
    print(f"Initial total size: {initial_size / (1024*1024):.2f} MB")
    
    if dry_run:
        print("\n[DRY RUN MODE - No files will be modified]")
        # Sample one image to show compression estimate
        sample_jpg = next(episodes_dir.rglob("*.jpg"), None)
        if sample_jpg:
            temp_path = sample_jpg.parent / "temp_compressed.jpg"
            compress_image(sample_jpg, temp_path, quality, max_size)
            original_size = sample_jpg.stat().st_size
            compressed_size = temp_path.stat().st_size
            temp_path.unlink()
            ratio = (1 - compressed_size / original_size) * 100
            print(f"\nSample compression: {original_size/1024:.1f}KB -> {compressed_size/1024:.1f}KB ({ratio:.1f}% reduction)")
            total_jpg_count = sum(1 for _ in episodes_dir.rglob("*.jpg"))
            estimated_final = initial_size * (compressed_size / original_size)
            print(f"Estimated final size: {estimated_final / (1024*1024):.2f} MB")
            print(f"Estimated savings: {(initial_size - estimated_final) / (1024*1024):.2f} MB")
        return
    
    # Process all frames
    total_files = sum(1 for _ in episodes_dir.rglob("*.jpg"))
    print(f"\nCompressing {total_files} frames with quality={quality}...")
    
    processed = 0
    errors = 0
    
    with tqdm(total=total_files, desc="Compressing frames") as pbar:
        for frames_dir in frame_dirs:
            jpg_files = sorted(frames_dir.glob("*.jpg"))
            
            for jpg_file in jpg_files:
                # Create temporary file
                temp_file = jpg_file.parent / f"{jpg_file.stem}_temp.jpg"
                
                # Compress to temporary file
                if compress_image(jpg_file, temp_file, quality, max_size):
                    # Replace original with compressed version
                    temp_file.replace(jpg_file)
                    processed += 1
                else:
                    # Clean up temp file if it exists
                    if temp_file.exists():
                        temp_file.unlink()
                    errors += 1
                
                pbar.update(1)
    
    # Calculate final size
    print("\nCalculating final size...")
    final_size = get_total_size(episodes_dir)
    saved = initial_size - final_size
    percentage = (saved / initial_size) * 100
    
    print(f"\nCompression complete!")
    print(f"Files processed: {processed}")
    print(f"Errors: {errors}")
    print(f"Initial size: {initial_size / (1024*1024):.2f} MB")
    print(f"Final size: {final_size / (1024*1024):.2f} MB")
    print(f"Space saved: {saved / (1024*1024):.2f} MB ({percentage:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Compress episode frame images')
    parser.add_argument(
        '--episodes-dir',
        type=Path,
        default=Path(__file__).parent.parent.parent / 'episodes',
        help='Path to episodes directory (default: ../../episodes)'
    )
    parser.add_argument(
        '--quality',
        type=int,
        default=85,
        help='JPEG quality 1-100 (default: 85)'
    )
    parser.add_argument(
        '--max-width',
        type=int,
        help='Maximum width for resizing (maintains aspect ratio)'
    )
    parser.add_argument(
        '--max-height',
        type=int,
        help='Maximum height for resizing (maintains aspect ratio)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    
    args = parser.parse_args()
    
    if not args.episodes_dir.exists():
        print(f"Error: Episodes directory not found: {args.episodes_dir}")
        return 1
    
    max_size = None
    if args.max_width and args.max_height:
        max_size = (args.max_width, args.max_height)
    
    compress_episode_frames(
        args.episodes_dir,
        quality=args.quality,
        max_size=max_size,
        dry_run=args.dry_run
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
