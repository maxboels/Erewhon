# 📊 Training Output Guide

**Updated:** October 9, 2025

## 🎨 New Improved Training Output

### What Changed:

1. ✅ **Progress bars** with tqdm (visual training progress)
2. ✅ **Better timestamps** (readable format: `2025-10-09 10:15:47` instead of ISO)
3. ✅ **Ratio display** (current/total batches and epochs)
4. ✅ **ETA** (estimated time remaining)
5. ✅ **Tree-style summaries** for better readability

---

## 📺 Example Output

### Startup:
```
================================================================================
🤖 LEROBOT ACT TRAINER FOR RC CAR
================================================================================

📁 Data Directory: src/robots/rover/episodes
🔍 Scanning for episodes...
   Scanning episodes: 10/88 (11%)
   Scanning episodes: 20/88 (22%)
   ...
✅ Successfully loaded 84 valid episodes

================================================================================
🚀 STARTING TRAINING
================================================================================
📊 Training samples: 9937
📊 Validation samples: 2484
🎯 Total epochs: 30
📦 Batch size: 16
💾 Output directory: outputs/lerobot_act/lerobot_act_20251009_103015
================================================================================
```

### Training Progress (NEW! 🎉):
```
================================================================================
🏃 EPOCH 1/30
================================================================================

Epoch 1/30:  50%|████████████          | 350/701 [02:15<02:15, 2.60it/s] Loss: 0.4523 (LR: 1.00e-04)
```

**What it shows:**
- `Epoch 1/30` - Current epoch / total epochs
- `50%` - Percentage complete
- `████████████` - Visual progress bar
- `350/701` - Current batch / total batches
- `[02:15<02:15]` - Elapsed time < Remaining time
- `2.60it/s` - Iterations (batches) per second
- `Loss: 0.4523` - Current average loss
- `LR: 1.00e-04` - Current learning rate

### Validation Progress:
```
Validating: 100%|█████████████████████████| 155/155 [00:15<00:00, 10.2it/s] Loss: 0.0561
```

### Epoch Summary (NEW! 🎉):
```
🌟 New best model! Val loss: 0.056080

📊 Epoch 1/30 Summary:
   ├─ Train Loss: 1.427284
   ├─ Val Loss:   0.056080
   ├─ Best Loss:  0.056080
   ├─ LR:         5.05e-05
   └─ Time:       5m 17s
```

**Tree-style layout:**
- `├─` = intermediate items
- `└─` = last item
- Easy to scan visually

### Completion:
```
================================================================================
🎉 TRAINING COMPLETED!
================================================================================
📊 Best validation loss: 0.045123
⏱️  Total training time: 2h 45m 32s
📁 Models saved in: outputs/lerobot_act/lerobot_act_20251009_103015
================================================================================
```

---

## 📁 CSV Log Format (Updated)

### Batch Metrics (`batch_metrics.csv`):

**Old format:**
```csv
step,epoch,batch_loss,learning_rate,timestamp
0,0,95.94695,0.0001,2025-10-09T10:15:47.602829
```

**New format:**
```csv
step,epoch,batch_loss,learning_rate,timestamp
0,0,95.94695,0.0001,2025-10-09 10:15:47
```

✅ Much easier to read!

### Epoch Metrics (`epoch_metrics.csv`):

**Old format:**
```csv
epoch,train_loss,val_loss,best_val_loss,learning_rate,epoch_time,total_samples,timestamp
0,1.427,0.056,0.056,5.05e-05,317.65,12421,2025-10-09T10:21:28.149100
```

**New format:**
```csv
epoch,train_loss,val_loss,best_val_loss,learning_rate,epoch_time,total_samples,timestamp
0,1.427,0.056,0.056,5.05e-05,317.65,12421,2025-10-09 10:21:28
```

✅ Human-readable timestamps!

---

## 🎯 Understanding Your Training Logs

### From Your Recent 2-Epoch Run:

#### Batch Metrics:
```
Step 0:    Loss = 95.95  ← Random initialization
Step 10:   Loss = 6.57   ← 93% drop in 10 steps!
Step 100:  Loss = 2.06   ← 98% drop
Step 500:  Loss = 0.55   ← 99.4% drop
Step 770:  Loss = 0.24   ← Epoch 0 complete
```

#### Epoch Summary:
```
Epoch 0:
  - Train Loss: 1.427 (average)
  - Val Loss: 0.056 (excellent!)
  - Time: 5m 17s
  - LR: 1e-4 → 5e-5 (scheduler reduced)

Epoch 1:
  - Train Loss: 0.175 (8x better!)
  - Val Loss: 0.060 (slight increase, normal)
  - Time: 5m 16s
  - LR: 5e-5 → 1e-6 (too aggressive!)
```

### Key Insights:

1. **Model learns FAST** 🚀
   - 99.7% loss reduction in epoch 0
   - Already converging well

2. **Not overfitting** ✅
   - Val loss (0.056) very close to train loss
   - Good generalization

3. **Learning rate issue** ⚠️
   - Dropped from 1e-4 to 1e-6 in 2 epochs
   - Too aggressive decay
   - Should tune scheduler

---

## 📊 Monitoring in tmux

### Your 3-Pane Setup:

```
┌─────────────────────────────────┬─────────────────────────────────┐
│ LEFT: Training Progress Bar    │ TOP-RIGHT: GPU Monitor          │
│                                 │                                 │
│ Epoch 12/30: 65%|████▌  | 456/│ GPU: RTX 3070                   │
│ 701 [03:12<01:45] Loss: 0.123  │ Memory: 4350MB / 8188MB (53%)   │
│                                 │ Utilization: 98%                │
│ Validating: 100%|███████| 155/ │ Temp: 72°C                      │
│ 155 Loss: 0.098                 │ Power: 150W                     │
│                                 │                                 │
│ 📊 Epoch 12/30 Summary:         ├─────────────────────────────────┤
│    ├─ Train Loss: 0.123         │ BOTTOM-RIGHT: CSV Logs          │
│    ├─ Val Loss:   0.098         │                                 │
│    ├─ Best Loss:  0.095         │ step,epoch,loss,lr,timestamp    │
│    ├─ LR:         3.21e-05      │ 8400,12,0.123,3.21e-05,10:15:47│
│    └─ Time:       2m 45s        │ 8410,12,0.121,3.21e-05,10:15:51│
│                                 │ 8420,12,0.125,3.21e-05,10:15:55│
└─────────────────────────────────┴─────────────────────────────────┘
```

### What Each Pane Shows:

**LEFT (Training):**
- ✅ Real-time progress bar
- ✅ Current/total batches
- ✅ Time elapsed & remaining
- ✅ Live loss updates
- ✅ Epoch summaries

**TOP-RIGHT (GPU):**
- ✅ GPU name & stats
- ✅ Memory usage %
- ✅ GPU utilization %
- ✅ Temperature
- ✅ Power draw

**BOTTOM-RIGHT (Logs):**
- ✅ CSV being written live
- ✅ Step-by-step metrics
- ✅ Readable timestamps

---

## 💡 Pro Tips

### Tip 1: Watch Training Speed
```
2.60it/s = 2.6 batches/second
→ 701 batches ÷ 2.6 = ~270 seconds = 4.5 minutes per epoch
```

### Tip 2: Estimate Total Time
```
Progress bar shows: [02:15<02:15]
                     ↑elapsed ↑remaining
Total epoch time = 4m 30s
```

### Tip 3: Monitor Loss Curve
Watch the loss in the progress bar:
- Decreasing steadily: ✅ Good
- Fluctuating wildly: ⚠️ LR might be too high
- Stuck/flat: ⚠️ LR might be too low or converged

### Tip 4: Check GPU Utilization
- 98%: ✅ Perfect, GPU-bound training
- <50%: ⚠️ CPU bottleneck (data loading slow)
- Memory near max: ⚠️ Risk of OOM

---

## 🐛 Troubleshooting

### Progress Bar Not Showing?
**Problem:** Running in environment without tty
**Solution:** Training still works, just no progress bar. Logs still appear.

### Timestamps Still ISO Format?
**Problem:** Old CSV files from previous runs
**Solution:** New runs will use new format. Old files unchanged.

### Progress Bar Glitchy in tmux?
**Problem:** Terminal size or TERM variable
**Solution:** 
```bash
export TERM=xterm-256color
# Or adjust ncols in tqdm: ncols=80
```

### Want to disable progress bar?
**Solution:**
```bash
# Set environment variable:
export TQDM_DISABLE=1
python training_script.py ...
```

---

## 🎉 Summary

Your new training output includes:

✅ **Visual progress bars** - See training progress in real-time  
✅ **Readable timestamps** - `2025-10-09 10:15:47` format  
✅ **Batch/Epoch ratios** - `350/701 batches`, `12/30 epochs`  
✅ **ETAs** - Know when training will finish  
✅ **Tree summaries** - Easy to scan epoch results  
✅ **Time formatting** - `2h 45m 32s` instead of `9932.5s`  

Much better training experience! 🚀
