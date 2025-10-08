# 🚀 State-Aware ACT Quick Commands

## ✅ The Correct Way to Train (State-Aware)

### Training
```bash
# Navigate to ACT directory
cd /home/maxboels/projects/Erewhon/src/policies/ACT

# Train state-aware model (RECOMMENDED)
python3 state_aware_act_trainer.py \
  --data_dir /home/maxboels/projects/Erewhon/src/robots/rover/episodes \
  --max_epochs 50 \
  --batch_size 8 \
  --device cuda

# Custom output directory
python3 state_aware_act_trainer.py \
  --data_dir ../robots/rover/episodes \
  --output_dir ./outputs/my_state_aware_model \
  --max_epochs 100 \
  --learning_rate 1e-4
```

### Testing
```bash
# Test on episode data
python3 state_aware_inference.py \
  --model outputs/state_aware_act_XXXXXX/best_model.pth \
  --episode ../robots/rover/episodes/episode_20251007_144013 \
  --num_samples 20 \
  --device cuda

# Test on CPU
python3 state_aware_inference.py \
  --model outputs/state_aware_act_XXXXXX/best_model.pth \
  --episode ../robots/rover/episodes/episode_20251007_144013 \
  --device cpu
```

---

## ❌ OLD Image-Only Models (Not Recommended)

These models DO NOT use current state:
- `simple_act_trainer.py` - Image only
- `enhanced_act_trainer.py` - Image only
- `test_inference.py` - Image only

**Don't use these for RC car control!**

---

## ✅ Official ACT (Also State-Aware)

Alternative using LeRobot's official implementation:

```bash
# Official ACT with full LeRobot framework
python3 official_act_trainer.py \
  --data_dir ../robots/rover/episodes \
  --epochs 100

# Test official ACT
python3 test_official_act.py \
  --checkpoint outputs/official_act_XXX/best_model.pth \
  --data_dir ../robots/rover/episodes
```

---

## 📊 Model Comparison

| Script | Uses State? | Recommended? | Notes |
|--------|------------|--------------|-------|
| `state_aware_act_trainer.py` | ✅ YES | ✅ **YES** | Custom, optimized for RC car |
| `official_act_trainer.py` | ✅ YES | ✅ Yes | Full LeRobot framework |
| `simple_act_trainer.py` | ❌ NO | ❌ No | Image-only, not optimal |
| `enhanced_act_trainer.py` | ❌ NO | ❌ No | Image-only, not optimal |

---

## 🎯 Recommended Workflow

### 1. Data Collection (On Raspberry Pi)
```bash
cd ~/EDTH2025/Erewhon/src/robots/rover

# Record episodes
python3 src/record/episode_recorder.py \
  --episode-duration 6 \
  --action-label "navigate track"
```

### 2. Transfer Data (To Laptop)
```bash
# On laptop
rsync -avz --progress \
  mboels@raspberrypi:~/EDTH2025/Erewhon/src/robots/rover/episodes/ \
  ./src/robots/rover/episodes/
```

### 3. Train Model (On Laptop)
```bash
cd src/policies/ACT

# State-aware training
python3 state_aware_act_trainer.py \
  --data_dir ../robots/rover/episodes \
  --max_epochs 50 \
  --device cuda
```

### 4. Test Inference (On Laptop)
```bash
# Test on episode data
python3 state_aware_inference.py \
  --model outputs/state_aware_act_XXXXXX/best_model.pth \
  --episode ../robots/rover/episodes/episode_XXXXXX \
  --device cuda
```

### 5. Deploy (On Raspberry Pi)
```bash
# Copy model back to Pi
scp outputs/state_aware_act_XXXXXX/best_model.pth \
  mboels@raspberrypi:~/EDTH2025/Erewhon/models/

# Run autonomous control
python3 autonomous_control.py \
  --model ~/EDTH2025/Erewhon/models/best_model.pth \
  --device cpu
```

---

## 🔍 Verify Your Model Uses State

### Check Training Logs
```bash
# Look for this in training output:
# "Model architecture: State-aware (image + current state)"
# "Dataset split: X train, Y validation"
# "Model receives: Image (640x480) + Current State [steering, throttle]"
```

### Check Model Architecture
```python
import torch

checkpoint = torch.load('best_model.pth')
config = checkpoint['config']

# Should have state_dim parameter
print(config.get('state_dim', 'NOT FOUND'))  # Should print: 2

# Check model forward signature
# State-aware: forward(self, images, states)
# Image-only:  forward(self, images)
```

### Check Dataset Output
```python
from state_aware_act_trainer import SimplifiedTracerDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((480, 640)),
    transforms.ToTensor()
])

dataset = SimplifiedTracerDataset(
    '../robots/rover/episodes',
    transforms=transform
)

# Should return 3 items: image, state, action
image, state, action = dataset[0]
print(f"Image shape: {image.shape}")    # [3, 480, 640]
print(f"State shape: {state.shape}")    # [2]
print(f"Action shape: {action.shape}")  # [2]
```

---

## 📈 Expected Training Output

```
🚀 Starting state-aware ACT training...
Training for 50 epochs
Model receives: Image (640x480) + Current State [steering, throttle]

Epoch 1/50 | Train Loss: 0.012345 | Val Loss: 0.010234 | Best Val: 0.010234
Epoch 2/50 | Train Loss: 0.008765 | Val Loss: 0.007654 | Best Val: 0.007654
...
Epoch 50/50 | Train Loss: 0.000345 | Val Loss: 0.000123 | Best Val: 0.000123

🎉 Training completed!
📊 Final Results:
  Best validation loss: 0.000123
  Model checkpoint: outputs/state_aware_act_XXX/best_model.pth
```

---

## 🚨 Common Mistakes to Avoid

### ❌ Don't do this:
```bash
# Using image-only trainers
python3 simple_act_trainer.py       # NO!
python3 enhanced_act_trainer.py     # NO!
```

### ✅ Do this instead:
```bash
# Use state-aware trainer
python3 state_aware_act_trainer.py  # YES!
```

### ❌ Don't do this during inference:
```python
# Forgetting to pass state
action = model(image)  # Missing state!
```

### ✅ Do this instead:
```python
# Use inference engine (handles state automatically)
from state_aware_inference import StateAwareInference

engine = StateAwareInference('best_model.pth')
result = engine.predict(image)  # State tracked internally!
```

---

## 💡 Pro Tips

1. **Always check model uses state**
   - Look for "State-aware" in training logs
   - Verify model has state encoder

2. **Start with small epochs for testing**
   - Use `--max_epochs 10` first
   - Verify training works before long runs

3. **Monitor validation loss**
   - Should decrease smoothly
   - If stuck, try lower learning rate

4. **Test on episode data first**
   - Verify inference before deploying
   - Check state progression is smooth

5. **Use GPU for training**
   - `--device cuda` for much faster training
   - CPU is fine for inference

---

## 📞 Need Help?

Check these files:
- `STATE_AWARE_TRAINING_GUIDE.md` - Full guide
- `STATE_INPUT_ANALYSIS.md` - Technical details
- `RESOLUTION_CONFIG.md` - Image resolution info

---

**Remember: Your model needs BOTH camera observations AND current state!** 🎯
