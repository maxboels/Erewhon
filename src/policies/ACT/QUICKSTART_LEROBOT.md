# 🚀 Quick Start: Full LeRobot ACT for RC Car

**Last Updated:** October 8, 2025

## ✅ What Changed?

We switched from simplified custom implementations to the **full LeRobot ACT architecture**:
- ✅ Pre-trained ResNet18 backbone (11M params from ImageNet)
- ✅ VAE for stochastic policy (better generalization)
- ✅ Proven architecture from research paper
- ✅ 13.5M parameters vs 3.5M in simplified version

---

## 📦 Files Status

### ✅ Active (Use These):
```
src/policies/ACT/
├── official_lerobot_trainer.py      # Main training script ⭐
├── lerobot_act_inference.py         # Inference for deployment ⭐
├── README_LEROBOT_ACT.md            # Full documentation
└── lerobot/                          # Official LeRobot source code

src/datasets/
└── local_dataset_loader.py           # Dataset (LeRobot format) ⭐
```

### ❌ Deprecated (Moved):
```
src/policies/ACT/deprecated/
├── state_aware_act_trainer.py       # Old custom implementation
├── simple_act_trainer.py            # Old simplified version
├── enhanced_act_trainer.py          # Old enhanced version
└── README.md                         # Why deprecated
```

---

## 🏃 Quick Commands

### 1️⃣ Training (Default):
```bash
cd /home/maxboels/projects/Erewhon

python src/policies/ACT/official_lerobot_trainer.py \
    --data_dir ./episodes \
    --output_dir ./outputs/lerobot_act \
    --epochs 100 \
    --batch_size 16 \
    --device cuda
```

### 2️⃣ Training (Custom):
```bash
python src/policies/ACT/official_lerobot_trainer.py \
    --data_dir ./episodes \
    --output_dir ./outputs/lerobot_act \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 1e-4 \
    --device cuda
```

### 3️⃣ Inference Test:
```bash
# With test image
python src/policies/ACT/lerobot_act_inference.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --test_image path/to/test_frame.jpg \
    --device cuda

# With dummy image
python src/policies/ACT/lerobot_act_inference.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --device cuda
```

---

## 🔧 Configuration

### Key Settings (Already Updated):
```python
# Resolution
image_height = 360  # ✅ Fixed from 480
image_width = 640

# Input/Output shapes
input_shapes = {
    "observation.images.cam_front": [3, 360, 640],
    "observation.state": [2],  # [steering, throttle]
}
output_shapes = {
    "action": [2],  # [steering, throttle] PWM
}

# Backbone
vision_backbone = "resnet18"
pretrained_backbone = True  # ✅ ImageNet weights!

# VAE
use_vae = True  # ✅ Stochastic policy
latent_dim = 32
```

---

## 📊 What to Expect

### Training:
- ✅ Faster convergence (pre-trained weights)
- ✅ Better loss (~0.01-0.05 L1 loss)
- ✅ ~1-2 hours for 100 epochs (GPU)

### Inference:
- ✅ ~50-100 FPS on GPU
- ✅ ~10-30 FPS on Raspberry Pi 5 (AI HAT)
- ✅ Smooth control with action chunking

---

## 🎯 Python API

### Training:
```python
from src.policies.ACT.official_lerobot_trainer import LeRobotACTTrainer

# Create trainer
trainer = LeRobotACTTrainer(config={
    'data_dir': './episodes',
    'output_dir': './outputs/lerobot_act',
    'num_epochs': 100,
    'batch_size': 16,
    # ... other params
})

# Train
trainer.train()
```

### Inference:
```python
from src.policies.ACT.lerobot_act_inference import LeRobotACTInference
from PIL import Image

# Load model
inference = LeRobotACTInference(
    checkpoint_path='outputs/lerobot_act/best_model.pth',
    device='cuda'
)

# Predict
image = Image.open('camera_frame.jpg')
steering, throttle = inference.predict(image)

print(f"Steering: {steering:.4f}, Throttle: {throttle:.4f}")
```

---

## 🔍 Verification

### Check Dataset Format:
```python
from src.datasets.local_dataset_loader import TracerLocalDataset

dataset = TracerLocalDataset('./episodes')
sample = dataset[0]

# Should have these keys:
assert 'observation.images.cam_front' in sample  # [3, 360, 640]
assert 'observation.state' in sample             # [2]
assert 'action' in sample                        # [2]

print("✅ Dataset format correct!")
```

### Check Model Loading:
```python
from src.policies.ACT.official_lerobot_trainer import LeRobotACTTrainer

trainer = LeRobotACTTrainer({'data_dir': './episodes'})
trainer.setup_data()
trainer.setup_model()

print(f"✅ Model loaded with {sum(p.numel() for p in trainer.policy.parameters()):,} parameters")
```

---

## 📈 Monitoring

### During Training:
```bash
# Watch loss
tail -f outputs/lerobot_act/logs/batch_metrics.csv

# Check epoch summary
tail -f outputs/lerobot_act/logs/epoch_metrics.csv
```

### Checkpoints:
```
outputs/lerobot_act/
├── checkpoints/
│   ├── latest_model.pth       # Latest checkpoint
│   ├── best_model.pth         # Best validation loss
│   └── epoch_*.pth            # Periodic saves
└── logs/
    ├── batch_metrics.csv      # Per-batch loss
    └── epoch_metrics.csv      # Per-epoch summary
```

---

## 🐛 Common Issues

### Issue: "Input shapes mismatch"
```python
# Fix: Ensure dataset returns correct keys
# In local_dataset_loader.py line ~178
return {
    'observation.images.cam_front': image,  # ✅ Correct key
    'observation.state': state,              # ✅ Correct key
    'action': action,
}
```

### Issue: "action_is_pad missing"
```python
# Fix: Use lerobot_collate_fn
from src.datasets.local_dataset_loader import lerobot_collate_fn

train_loader = DataLoader(
    dataset,
    batch_size=16,
    collate_fn=lerobot_collate_fn  # ✅ Adds action_is_pad
)
```

### Issue: "No pre-trained weights"
```python
# Fix: Enable in config
act_config = ACTConfig(
    vision_backbone="resnet18",
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",  # ✅
)
```

---

## ✅ Migration Checklist

If migrating from old simplified version:

- [x] Updated resolution to 360×640
- [x] Dataset returns LeRobot format keys
- [x] Using `lerobot_collate_fn` in DataLoader
- [x] ACTConfig has input/output shapes
- [x] Pre-trained ResNet18 enabled
- [x] VAE enabled (use_vae=True)
- [x] Moved old trainers to deprecated/
- [x] Updated inference script

---

## 📚 Documentation

- **Full Guide:** `README_LEROBOT_ACT.md`
- **API Reference:** See docstrings in `official_lerobot_trainer.py`
- **Deprecated Info:** `deprecated/README.md`

---

## 🎉 Summary

**Everything is ready!** Just run:

```bash
# Train
python src/policies/ACT/official_lerobot_trainer.py \
    --data_dir ./episodes \
    --output_dir ./outputs/lerobot_act \
    --epochs 100 \
    --batch_size 16 \
    --device cuda

# Deploy
python src/policies/ACT/lerobot_act_inference.py \
    --checkpoint outputs/lerobot_act/best_model.pth \
    --test_image test.jpg
```

**The full LeRobot ACT is now configured for your RC car!** 🚗💨
