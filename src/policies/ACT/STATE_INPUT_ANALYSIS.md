# 🔍 ACT Model State Input Analysis

## ❓ Question: Do We Pass Current States (Steering & Throttle) to the ACT Model?

## ✅ **Answer: YES, but with Important Differences Between Implementations**

---

## 📊 Summary by Implementation

### 1. **Official ACT Trainer** (`official_act_trainer.py`) ✅ **USES STATE**

**Configuration:**
```python
input_features={
    'observation.images.front': PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 480, 640)
    ),
    'observation.state': PolicyFeature(
        type=FeatureType.STATE,
        shape=(2,)  # [steering, throttle]
    ),
}
```

**Data Format:**
```python
formatted_item = {
    'observation.images.front': image_tensor,     # Camera image [3, 480, 640]
    'observation.state': state_tensor,            # Current state [steering, throttle]
    'action': action_tensor,                      # Target action [steering, throttle]
}
```

**✅ The official ACT policy receives:**
- Camera image (640x480)
- **Current state: [steering, throttle]** (2D vector)
- Predicts: Next action [steering, throttle]

---

### 2. **Train Local ACT** (`train_local_act.py`) ✅ **USES STATE**

**Configuration:**
```python
input_features = {
    'observation.image_front': PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 480, 640)
    ),
    'observation.state': PolicyFeature(
        type=FeatureType.STATE,
        shape=(2,)  # [steering, throttle]
    )
}
```

**✅ This implementation also uses state as input**

---

### 3. **Simple ACT Trainer** (`simple_act_trainer.py`) ❌ **DOES NOT USE STATE**

**Model Architecture:**
```python
def forward(self, images):
    batch_size = images.shape[0]
    
    # Encode images
    visual_features = self.vision_encoder(images)  # Only uses images!
    
    # Add sequence dimension for transformer
    visual_features = visual_features.unsqueeze(1)
```

**Dataset Output:**
```python
# Action vector [steering, throttle]
action = torch.tensor([sample['steering'], sample['throttle']], dtype=torch.float32)

return image, action  # Only returns image and action, NO state
```

**❌ This simplified version is image-only:**
- Input: Camera image (640x480)
- Output: Action [steering, throttle]
- **Does NOT use current state as input**

---

### 4. **Enhanced ACT Trainer** (`enhanced_act_trainer.py`) ❌ **DOES NOT USE STATE**

```python
# Imports from simple_act_trainer
from simple_act_trainer import SimplifiedTracerDataset, SimpleACTModel
```

**❌ Uses the same image-only model as `simple_act_trainer.py`**

---

## 🔄 Data Flow Comparison

### **Official ACT (State-Aware)**
```
Input Pipeline:
1. Camera Frame (640×480) → Image Tensor [3, 480, 640]
2. Current State [steering, throttle] → State Tensor [2]
3. Concatenated Features → Transformer → Action Prediction

Model sees:
- Visual observation (where am I?)
- Current state (what am I doing?)
- Predicts next action (what should I do?)
```

### **Simple ACT (Image-Only)**
```
Input Pipeline:
1. Camera Frame (640×480) → Image Tensor [3, 480, 640]
2. CNN Encoder → Visual Features
3. Transformer → Action Prediction

Model sees:
- Visual observation (where am I?)
- Predicts action (what should I do?)
```

---

## 📁 Dataset State Handling

### **Local Dataset Loader** (`local_dataset_loader.py`)

The dataset **ALWAYS includes state**, even if the model doesn't use it:

```python
# In __getitem__ method:
action = torch.tensor([
    sample['steering'],
    sample['throttle']
], dtype=torch.float32)

# Create observation dict matching LeRobot format
observation = {
    'image_front': image,
    'state': action,  # Current state (for state-action pairs)
}

return {
    'observation': observation,
    'action': action,
    # ...
}
```

**Key Point:** The dataset provides `observation.state`, but:
- ✅ Official ACT **uses it** as additional input
- ❌ Simple ACT **ignores it** and only uses the image

---

## 🤖 Inference Behavior

### **Official ACT Inference** (`test_official_act.py`)

```python
def predict_action(policy, preprocessor, postprocessor, 
                   image_tensor, current_state=None, device='cuda'):
    
    # Default state if none provided (neutral steering/throttle)
    if current_state is None:
        current_state = torch.tensor([0.0, 0.0], dtype=torch.float32)
    
    # Prepare batch for ACT
    batch = {
        'observation.images.front': image_tensor.unsqueeze(0).to(device),
        'observation.state': current_state.unsqueeze(0).to(device),  # ✅ Passes state
    }
    
    # Run inference
    action_output = policy.select_action(batch)
```

**✅ During inference, you MUST provide:**
1. Camera image
2. Current state [steering, throttle]

**Default behavior:** If no state provided, uses `[0.0, 0.0]` (neutral)

### **Simple ACT Inference** (`test_inference.py`)

```python
# Model expects only images
predicted_actions = model(images_batch)  # No state needed
```

**✅ During inference, you only need:**
1. Camera image

---

## 🎯 Practical Implications

### **For Training Data Collection:**

Both approaches need the same data format:
```
episode_XXXXXX/
├── frame_000000.jpg              # 640×480 image
├── frame_000001.jpg
└── episode_data.json             # Contains control samples
    {
        "control_samples": [
            {
                "steering_normalized": 0.5,
                "throttle_normalized": 0.3,
                "system_timestamp": 1234567890.123
            },
            ...
        ],
        "frame_samples": [...]
    }
```

**✅ No change needed in data collection!**

---

### **For Real-Time Inference:**

#### **Official ACT (State-Aware):**
```python
# You need to track and provide current state
current_steering = 0.0
current_throttle = 0.0

while True:
    frame = camera.capture()
    
    # Create state tensor from current values
    current_state = torch.tensor([current_steering, current_throttle])
    
    # Predict next action
    result = policy.predict(frame, current_state)
    
    # Update current state for next iteration
    current_steering = result['steering']
    current_throttle = result['throttle']
    
    # Send to motors
    send_to_motors(current_steering, current_throttle)
```

#### **Simple ACT (Image-Only):**
```python
# Simpler - only needs camera frames
while True:
    frame = camera.capture()
    
    # Predict action directly from image
    result = model(frame)
    
    # Send to motors
    send_to_motors(result[0], result[1])
```

---

## 🔬 Theoretical Differences

### **State-Aware (Official ACT)**

**Advantages:**
- ✅ Model knows current vehicle dynamics
- ✅ Can learn smoother control (knows momentum)
- ✅ Better temporal consistency
- ✅ Can compensate for lag/delays

**Disadvantages:**
- ❌ More complex input pipeline
- ❌ Requires state tracking during inference
- ❌ Initialization issue (what's the first state?)

### **Image-Only (Simple ACT)**

**Advantages:**
- ✅ Simpler architecture
- ✅ Easier deployment (no state tracking)
- ✅ More robust to initialization
- ✅ Purely visual reasoning

**Disadvantages:**
- ❌ Cannot directly access current vehicle state
- ❌ Must infer dynamics from visual motion
- ❌ Potentially less smooth control

---

## 📋 Quick Reference Table

| Implementation | Uses State Input? | Input Dimensions | Model Type |
|---------------|------------------|------------------|------------|
| `official_act_trainer.py` | ✅ YES | Image: [3,480,640]<br>State: [2] | Full ACT |
| `train_local_act.py` | ✅ YES | Image: [3,480,640]<br>State: [2] | Full ACT |
| `simple_act_trainer.py` | ❌ NO | Image: [3,480,640] | Image-only |
| `enhanced_act_trainer.py` | ❌ NO | Image: [3,480,640] | Image-only |

---

## 🔧 Recommendation for Deployment

### **For Raspberry Pi 5 Deployment:**

1. **If using Official ACT model:**
   - Initialize state to `[0.0, 0.0]` at startup
   - Track and update state at each inference step
   - Pass both image and current state to model

2. **If using Simple ACT model:**
   - Only need camera feed
   - Direct image → action prediction
   - Simpler implementation

### **Which to Choose?**

**Use State-Aware (Official ACT) if:**
- You want smoother, more controlled behavior
- You can track vehicle state reliably
- Training data shows good state information

**Use Image-Only (Simple ACT) if:**
- You want simplicity and reliability
- State tracking adds complexity you don't need
- Visual information alone is sufficient

---

## 🚨 Important Notes

1. **Dataset provides state regardless** - both implementations can use the same training data
2. **Inference differs** - state-aware needs current state input, image-only doesn't
3. **Default values** - if state is missing, official ACT uses `[0.0, 0.0]` (neutral)
4. **State meaning** - `[steering, throttle]` both normalized to [-1, 1] or [0, 1] range

---

## ✅ Final Answer

**YES**, the **Official ACT implementation** (`official_act_trainer.py` and `train_local_act.py`) **DOES pass current states** (steering and throttle) to the model as a 2D state vector `[steering, throttle]` in addition to the camera observations.

**NO**, the **Simple ACT implementation** (`simple_act_trainer.py` and `enhanced_act_trainer.py`) **DOES NOT use state** - it's a purely vision-based model that only takes camera images as input.

**For SmolVLA integration**, you'll likely follow the state-aware pattern since VLA models typically use both visual and state information for grounded action prediction.
