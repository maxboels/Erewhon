#!/bin/bash

echo "🚀 Starting ACT Training Session..."

# Kill existing session if it exists
if tmux has-session -t training 2>/dev/null; then
    echo "⚠️  Existing 'training' session found. Killing it..."
    tmux kill-session -t training
    echo "✅ Old session killed."
fi

# Wait a moment for cleanup
sleep 1

echo "🆕 Creating new tmux session..."

# Start tmux session
tmux new-session -d -s training

# Left pane: training
tmux send-keys -t training "conda activate lerobot" C-m
tmux send-keys -t training "cd /home/maxboels/projects/Erewhon" C-m
tmux send-keys -t training "python src/policies/ACT/official_lerobot_trainer.py --data_dir src/robots/rover/episodes --output_dir ./outputs/lerobot_act --epochs 2 --batch_size 16 --device cuda" C-m

# Split vertical and setup GPU monitor
tmux split-window -h -t training
tmux send-keys -t training "watch -n 1 nvidia-smi" C-m

# Split right pane horizontal and setup log monitor
tmux split-window -v -t training
tmux send-keys -t training "sleep 5 && tail -f outputs/lerobot_act/lerobot_act_*/logs/batch_metrics.csv" C-m

echo "✅ Session created! Attaching..."
echo ""

# Attach to session
tmux attach -t training
