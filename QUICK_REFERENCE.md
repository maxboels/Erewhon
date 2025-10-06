# Quick Command Reference

## Episode Recording & Analysis Workflow

### 1️⃣ Record Episode Data
```bash
cd /home/maxboels/projects/Erewhon/src/robots/rover

# Record a 15-second episode
python3 src/record/episode_recorder.py \
  --episode-duration 15 \
  --output-dir ./episodes \
  --action-label "hit red balloon"
```

### 2️⃣ Validate Episode Data
```bash
# Validate all episodes
python3 src/eval/validate_episode_data.py

# Validate specific episode
python3 src/eval/validate_episode_data.py episodes/episode_20251006_220059
```

### 3️⃣ Analyze Episode
```bash
# Generate plots for single episode
python3 src/record/episode_analyzer.py \
  --episode-dir ./episodes/episode_20251006_220059 \
  --plots
```

### 4️⃣ Create Animations

**Single Episode Animation (MP4):**
```bash
# By default, skips existing animations (fast!)
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --episodes episode_20251006_220059 \
  --format mp4 \
  --fps 10

# Force recreation even if animation exists
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --episodes episode_20251006_220059 \
  --format mp4 \
  --fps 10 \
  --force
```

**Multiple Episodes (GIF):**
```bash
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --format gif \
  --fps 8 \
  --output_dir ./animations
```

**Combined Multi-Episode Comparison:**
```bash
python3 src/eval/create_episode_animation.py \
  --data_dir ./episodes \
  --combined \
  --format mp4 \
  --fps 10
```

### 5️⃣ Interactive Frame Viewer
```bash
# Browse frames with keyboard (arrow keys)
python3 src/eval/episode_frame_viewer.py \
  --episode-dir ./episodes/episode_20251006_220059
```

## Dataset Analysis Tools

### Quick Dataset Overview
```bash
cd /home/maxboels/projects/Erewhon

python3 src/datasets/analysis/quick_dataset_analysis.py \
  --data-dir ./src/robots/rover/episodes
```

### Detailed Dataset Analysis
```bash
python3 src/datasets/analysis/analyze_dataset.py \
  --data-dir ./src/robots/rover/episodes
```

### Plot Training Signals
```bash
python3 src/datasets/analysis/plot_training_signals.py \
  --data-dir ./src/robots/rover/episodes \
  --output-dir ./plots
```

## ACT Policy Training

### Train ACT Model
```bash
cd /home/maxboels/projects/Erewhon/src/policies/ACT

# Simple training
python3 simple_act_trainer.py \
  --data-dir ../../robots/rover/episodes \
  --output-dir ./outputs

# Enhanced training with TensorBoard
python3 tensorboard_act_trainer.py \
  --data-dir ../../robots/rover/episodes \
  --output-dir ./outputs
```

### Run Inference
```bash
python3 inference_act.py \
  --model-path ./outputs/best_model.pth \
  --camera-id 0
```

## File Locations Quick Reference

| Task | Command Location |
|------|-----------------|
| Record Episodes | `src/robots/rover/src/record/episode_recorder.py` |
| Validate Data | `src/robots/rover/src/eval/validate_episode_data.py` |
| Create Animations | `src/robots/rover/src/eval/create_episode_animation.py` |
| View Frames | `src/robots/rover/src/eval/episode_frame_viewer.py` |
| Analyze Dataset | `src/datasets/analysis/analyze_dataset.py` |
| Train ACT | `src/policies/ACT/*_trainer.py` |

## Tips

💡 **Always validate before training**: Run `validate_episode_data.py` after recording
💡 **Use animations for debugging**: Create GIFs to quickly review episodes
💡 **Smart caching**: Animation script skips existing videos by default (use `--force` to recreate)
💡 **Monitor training**: Use `tensorboard_act_trainer.py` for real-time monitoring
💡 **Start simple**: Use `simple_act_trainer.py` for initial experiments
