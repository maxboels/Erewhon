# Resolution Optimization for RC Car Training

**Date:** October 7, 2025  
**Decision:** Reduce camera resolution from 640x480 to 320x240

## Changes Made

### 1. Updated Episode Recorder
**File:** `src/record/episode_recorder.py`

**Changes:**
- ✅ Default resolution changed: `640x480` → `320x240`
- ✅ Resolution is now configurable via `--resolution` argument
- ✅ Removed JPEG quality compression parameter (using OpenCV defaults)
- ✅ Kept frames as JPEG with default quality (~95)

### 2. Deleted Old Episodes
- ✅ Removed all episodes recorded at 640x480
- ✅ Episodes directory cleared for fresh data collection

## Resolution Comparison

| Resolution | Pixels | Storage per Frame* | Use Case |
|------------|--------|-------------------|----------|
| **320x240** | **77K** | **~15KB** | **Low-power robotics (NEW DEFAULT)** |
| 224x224 | 50K | ~10KB | Standard CNN input (ResNet, EfficientNet) |
| 640x480 | 307K | ~60KB | High-detail (OLD, removed) |

*Approximate with default JPEG quality

## Benefits of 320x240

### Storage Efficiency
- **~75% reduction** in file size per frame
- **10-episode dataset**: ~15MB vs ~60MB at 640x480
- **100-episode dataset**: ~150MB vs ~600MB at 640x480

### Training Speed
- **4x fewer pixels** to process per image
- Faster data loading during training
- Reduced GPU memory requirements
- Can fit larger batch sizes

### Inference Speed on Raspberry Pi
- **Critical for 30 Hz control loop**
- Less camera bandwidth required
- Faster image preprocessing
- Lower RAM usage

### Model Performance
- Sufficient detail for path following and obstacle avoidance
- Standard resolution for robotics applications
- Forces model to learn robust features (not rely on fine details)
- Better generalization with limited training data

## Usage Examples

### Default Recording (320x240)
```bash
python3 src/record/episode_recorder.py
```

### Custom Resolution
```bash
# Use 224x224 for standard CNN architectures
python3 src/record/episode_recorder.py --resolution 224x224

# Return to higher resolution if needed for demos
python3 src/record/episode_recorder.py --resolution 640x480
```

### Full Command
```bash
python3 src/record/episode_recorder.py \
  --episode-duration 6 \
  --resolution 320x240 \
  --arduino-port /dev/ttyACM0 \
  --camera-id 0 \
  --action-label "hit red balloon"
```

## Storage Estimates

### Single Episode (6 seconds @ 30fps)
- **Frames**: ~180 images
- **320x240**: ~2.7 MB
- **224x224**: ~1.8 MB
- **640x480**: ~10.8 MB (old)

### Full Training Dataset (100 episodes)
- **320x240**: ~270 MB ✓ (fits easily in GitHub)
- **224x224**: ~180 MB ✓
- **640x480**: ~1.08 GB ✗ (too large for GitHub)

## Image Quality

### What You Keep at 320x240:
- ✓ Overall scene layout and structure
- ✓ Object positions and relationships
- ✓ Clear path and obstacles
- ✓ Color information
- ✓ Motion patterns

### What You Lose (acceptable):
- Fine texture details
- Text readability (not needed for RC car)
- Distant small objects (not relevant at driving speed)

## Future Flexibility

You can always **increase resolution later** if needed:
- For high-quality demos/presentations: `--resolution 640x480`
- For research comparison: `--resolution 224x224`
- For production: Keep optimal `--resolution 320x240`

## Train-Test Consistency

**IMPORTANT:** Whatever resolution you record training data at, use the **same resolution** during inference!

```python
# In autonomous_control.py, ensure camera is set to same resolution as training:
camera = CameraCapture(camera_id=0, resolution=(320, 240))  # Match training data
```

## Next Steps

1. ✅ Record new episodes at 320x240 resolution
2. ✅ Verify data quality with episode analyzer
3. ✅ Train model on laptop with new data
4. ✅ Deploy to Raspberry Pi with matching 320x240 inference

## Command Reference

```bash
# Record episodes (default 320x240)
python3 src/record/episode_recorder.py

# Analyze episode quality
python3 src/record/episode_analyzer.py --episode-dir ./episodes/episode_* --plots

# Create animations for review
python3 src/eval/create_episode_animation.py --data_dir ./episodes

# Validate data
python3 src/eval/validate_episode_data.py
```

---

**Status:** ✅ System configured for optimal 320x240 resolution  
**Ready for:** Fresh data collection and training
