# Deprecated ACT Implementations

This folder contains older, simplified ACT implementations that have been deprecated in favor of the full LeRobot ACT architecture.

## Why Deprecated?

These implementations were created with the misconception that we needed a "simplified" version for RC car control. However, the full LeRobot ACT architecture is already flexible enough to handle any robot - from 14-DoF bimanual arms to 2-DoF RC cars.

### Files Moved Here:
- `state_aware_act_trainer.py` - Custom state-aware implementation
- `simple_act_trainer.py` - Simplified trainer
- `enhanced_act_trainer.py` - Enhanced trainer with extra features
- `tensorboard_act_trainer.py` - Trainer with TensorBoard logging

## What We Lost:
- ❌ No pre-trained ResNet18 backbone (random initialization)
- ❌ No VAE for stochastic policy
- ❌ Custom code requiring maintenance
- ❌ Smaller model capacity (3.5M vs 13.5M params)
- ❌ No normalization pipeline

## What to Use Instead:

### ✅ Use `official_lerobot_trainer.py`

This uses the full LeRobot ACT implementation with:
- ✅ Pre-trained ResNet18 backbone (ImageNet weights!)
- ✅ VAE for better generalization
- ✅ Proven architecture from research paper
- ✅ Community support and updates
- ✅ Proper normalization and preprocessing

### Configuration for RC Car:
```python
act_config = ACTConfig(
    input_shapes={
        "observation.images.cam_front": [3, 360, 640],
        "observation.state": [2],  # [steering, throttle]
    },
    output_shapes={
        "action": [2],  # [steering, throttle] PWM
    },
    vision_backbone="resnet18",
    pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1",
    use_vae=True,
    chunk_size=32,
    # ... other params
)
```

## Key Learnings:

1. **LeRobot ACT is flexible** - Just configure input/output shapes for your robot
2. **Joints vs PWM doesn't matter** - Model just predicts numbers
3. **Pre-trained weights are crucial** - Transfer learning from ImageNet helps a lot
4. **Don't reinvent the wheel** - Use proven implementations

## Date Deprecated:
October 8, 2025

## Reason:
Switched to full LeRobot ACT implementation for better performance and maintainability.
