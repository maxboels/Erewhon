# 🎓 Training Guide: ACT Policy for RC Car

**Professional training setup with tmux monitoring**

---

## 🚀 Quick Start: Training with tmux

### Why tmux?

- ✅ **Persistent sessions** - Survives SSH disconnections
- ✅ **Split panes** - Monitor GPU, training, and logs simultaneously  
- ✅ **Detach/Attach** - Close laptop, training continues
- ✅ **Professional workflow** - Industry standard for ML training

---

## 📋 Step-by-Step Setup

### 1. Open Separate Terminal (NOT VS Code)

```bash
# If remote (Raspberry Pi):
ssh pi@your-pi-address

# If local:
# Just open your system terminal app
```

### 2. Navigate to Project

```bash
cd /home/maxboels/projects/Erewhon
```

### 3. Start tmux Session

```bash
tmux new -s training
```

You should see a green status bar at the bottom - you're now in tmux! 🎉

### 4. Set Up 3-Pane Layout

#### Create Left + Right Panes (Vertical Split)

**Press:** `Ctrl+B` then `%`

You now have 2 panes side-by-side.

#### Left Pane: Training

```bash
# Activate conda
conda activate lerobot

# Start training
python src/policies/ACT/official_lerobot_trainer.py \
    --data_dir src/robots/rover/episodes \
    --output_dir ./outputs/lerobot_act \
    --epochs 100 \
    --batch_size 8 \
    --device cuda
```

**You'll see:**
```
================================================================================
🤖 LEROBOT ACT TRAINER FOR RC CAR
================================================================================

📁 Data Directory: src/robots/rover/episodes
🔍 Scanning for episodes...
✅ Found 88 episode directories

🔧 Configuration:
   Epochs: 100
   Batch Size: 8
   Device: cuda
   Output: ./outputs/lerobot_act
================================================================================

📂 Found 88 episode directories to scan...
   Scanning episodes: 10/88 (11%)
   ...
✅ Successfully loaded 84 valid episodes

================================================================================
🏃 EPOCH 1/100
================================================================================
📈 Epoch   1 [  10/1401] (  1%) | Loss: 2.846 | LR: 1.00e-04
📈 Epoch   1 [  20/1401] (  1%) | Loss: 2.494 | LR: 1.00e-04
...
```

#### Right Pane: GPU Monitor

**Switch to right pane:** `Ctrl+B` then `→`

```bash
watch -n 1 nvidia-smi
```

**You'll see:**
```
Every 1.0s: nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.104.05   Driver Version: 535.104.05   CUDA Version: 12.2   |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 45%   72C    P2   150W / 250W |   8192MiB / 24576MiB |     98%      Default |
+-------------------------------+----------------------+----------------------+
```

#### Split Right Pane: CSV Logs

**Split right pane horizontally:** `Ctrl+B` then `"`

**Move down to bottom pane:** `Ctrl+B` then `↓`

```bash
tail -f outputs/lerobot_act/lerobot_act_*/logs/batch_metrics.csv
```

**You'll see:**
```
step,epoch,batch_loss,learning_rate,timestamp
0,0,61.49240493774414,0.0001,2025-10-09T10:30:15.123456
10,0,5.6541547775268555,0.0001,2025-10-09T10:30:17.654321
20,0,4.231618881225586,0.0001,2025-10-09T10:30:19.987654
...
(new lines appear as training progresses)
```

---

## 🖥️ Your Final Layout

```
┌─────────────────────────────┬─────────────────────────────┐
│                             │ GPU  Name: GeForce RTX...   │
│ 🏃 EPOCH 12/100             │ Fan: 45% | Temp: 72°C       │
│                             │ Power: 150W / 250W          │
│ 📈 Epoch 12 [450/1401] (32%)│ Memory: 8192MB / 24576MB    │
│    Loss: 0.234              │ GPU-Util: 98%               │
│    LR: 8.5e-05              │                             │
│                             ├─────────────────────────────┤
│ Training output...          │ step,epoch,batch_loss,...   │
│ Saving checkpoint...        │ 4500,12,0.234,8.5e-05,...   │
│                             │ 4510,12,0.229,8.5e-05,...   │
│                             │ 4520,12,0.241,8.5e-05,...   │
│                             │ (updating in real-time)     │
└─────────────────────────────┴─────────────────────────────┘
```

---

## 🎮 tmux Controls (Cheat Sheet)

### Session Management

| Command | Action |
|---------|--------|
| `tmux new -s NAME` | Create new session |
| `Ctrl+B` then `D` | **Detach** (session keeps running) |
| `tmux ls` | List sessions |
| `tmux attach -t NAME` | Reattach to session |
| `tmux kill-session -t NAME` | Delete session |

### Pane Navigation

| Command | Action |
|---------|--------|
| `Ctrl+B` then `%` | Split **vertically** |
| `Ctrl+B` then `"` | Split **horizontally** |
| `Ctrl+B` then `←→↑↓` | Navigate between panes |
| `Ctrl+B` then `x` | Close current pane |
| `Ctrl+B` then `z` | Zoom/unzoom pane |

### Window Management

| Command | Action |
|---------|--------|
| `Ctrl+B` then `c` | New window |
| `Ctrl+B` then `n` | Next window |
| `Ctrl+B` then `p` | Previous window |
| `Ctrl+B` then `0-9` | Switch to window N |

---

## 📊 Monitoring During Training

### From tmux (Active Session)

You already have everything visible! Just look at your 3 panes:
- 📈 **Left:** Training progress
- 🖥️ **Top Right:** GPU usage
- 📝 **Bottom Right:** Loss metrics

### From Another Terminal

```bash
# Check if training is running
ps aux | grep official_lerobot_trainer

# View latest logs
tail -100 outputs/lerobot_act/lerobot_act_*/logs/batch_metrics.csv

# Watch GPU
nvidia-smi -l 1

# Reattach to tmux
tmux attach -t training
```

### From VS Code

While training runs in tmux, use VS Code for:

1. **View CSV files** (auto-refresh):
   - Open `outputs/lerobot_act/lerobot_act_*/logs/batch_metrics.csv`
   - New rows appear automatically!

2. **Edit code** for next experiment

3. **Run quick commands** in integrated terminal

---

## 🚪 Detaching & Reattaching

### Detach (Leave Training Running)

**Press:** `Ctrl+B` then `D`

```
[detached (from session training)]
```

**Now you can:**
- Close your laptop ✅
- Disconnect SSH ✅
- Do other work ✅
- Training keeps running! 🎉

### Reattach (Come Back Later)

```bash
# List sessions
tmux ls
# Output: training: 3 windows (created Wed Oct 9 10:30:00 2025)

# Reattach
tmux attach -t training

# You're back! See exactly where you left off
```

---

## 📈 Training Progress Examples

### Epoch 1 (Learning Starts)
```
📈 Epoch   1 [  10/1401] (  1%) | Loss: 61.492 | LR: 1.00e-04
📈 Epoch   1 [ 100/1401] (  7%) | Loss: 2.845 | LR: 1.00e-04
📈 Epoch   1 [ 500/1401] ( 36%) | Loss: 0.842 | LR: 1.00e-04
📈 Epoch   1 [1000/1401] ( 71%) | Loss: 0.377 | LR: 1.00e-04
📈 Epoch   1 [1400/1401] (100%) | Loss: 0.251 | LR: 1.00e-04

📊 Epoch   1 Summary:
   Train Loss: 0.251
   Val Loss:   0.198
   Best Loss:  0.198
   LR:         1.00e-04
   Time:       315.2s
🌟 New best model! Val loss: 0.198
```

### Epoch 50 (Well-Trained)
```
📈 Epoch  50 [  10/1401] (  1%) | Loss: 0.145 | LR: 5.00e-05
📈 Epoch  50 [ 500/1401] ( 36%) | Loss: 0.142 | LR: 5.00e-05
📈 Epoch  50 [1400/1401] (100%) | Loss: 0.138 | LR: 5.00e-05

📊 Epoch  50 Summary:
   Train Loss: 0.142
   Val Loss:   0.135
   Best Loss:  0.135
   LR:         5.00e-05
   Time:       298.7s
🌟 New best model! Val loss: 0.135
```

### Completion
```
================================================================================
🎉 TRAINING COMPLETED!
================================================================================
📊 Best validation loss: 0.135
⏱️  Total training time: 523.4 minutes
📁 Models saved in: outputs/lerobot_act/lerobot_act_20251009_103015
================================================================================
```

---

## 🗂️ Output Files

After training completes, you'll find:

```
outputs/lerobot_act/lerobot_act_20251009_103015/
├── training_config.json          # All hyperparameters
├── best_model.pth                 # Best model (lowest val loss)
├── checkpoint_epoch_50.pth        # Periodic checkpoints
├── checkpoint_epoch_100.pth
└── logs/
    ├── batch_metrics.csv          # Step-by-step losses
    └── epoch_metrics.csv          # Epoch summaries
```

---

## 🎯 Pro Tips

### Tip 1: Save Terminal Output to File
```bash
# In left pane, redirect output:
python src/policies/ACT/official_lerobot_trainer.py \
    --data_dir src/robots/rover/episodes \
    --output_dir ./outputs/lerobot_act \
    --epochs 100 \
    --batch_size 8 \
    --device cuda \
    2>&1 | tee training_$(date +%Y%m%d_%H%M%S).log
```

### Tip 2: Multiple Training Runs
```bash
# Create separate sessions:
tmux new -s training_exp1
tmux new -s training_exp2

# List all:
tmux ls
# training_exp1: 3 windows
# training_exp2: 3 windows

# Switch between:
tmux attach -t training_exp1
tmux attach -t training_exp2
```

### Tip 3: Resize Panes
```bash
# Make current pane bigger:
Ctrl+B then :
# Type: resize-pane -D 5  (down)
# Type: resize-pane -U 5  (up)
# Type: resize-pane -L 5  (left)
# Type: resize-pane -R 5  (right)
```

### Tip 4: Scroll in tmux
```bash
# Enter scroll mode:
Ctrl+B then [

# Navigate:
- Arrow keys: move cursor
- Page Up/Down: scroll pages
- q: exit scroll mode
```

---

## ❓ Troubleshooting

### "command not found: tmux"

Install it:
```bash
# Ubuntu/Debian
sudo apt-get update && sudo apt-get install tmux

# macOS
brew install tmux
```

### "no sessions" when trying to attach

Session doesn't exist. Create one:
```bash
tmux new -s training
```

### Panes are too small

Resize them:
```bash
Ctrl+B then :
resize-pane -R 20  # Make right pane bigger
```

### Lost which pane is which

Show pane numbers:
```bash
Ctrl+B then q
# Numbers appear briefly on each pane
```

### Want to kill a frozen pane

```bash
Ctrl+B then x
# Confirm with 'y'
```

---

## 🎓 Summary

**Your workflow:**

1. ✅ Open separate terminal
2. ✅ Start tmux: `tmux new -s training`
3. ✅ Set up 3-pane layout (splits + navigation)
4. ✅ Left: Training | Top-right: GPU | Bottom-right: Logs
5. ✅ Detach: `Ctrl+B then D`
6. ✅ Do other work, close laptop
7. ✅ Reattach anytime: `tmux attach -t training`
8. ✅ Training completes successfully! 🎉

**This is the professional way to train ML models!** 🚀

---

For questions or issues, check:
- Main README: `/README.md`
- ACT README: `/src/policies/ACT/README.md`
- Training status: `/TRAINING_STATUS.md`
