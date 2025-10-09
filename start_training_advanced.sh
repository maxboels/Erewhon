#!/bin/bash

# ACT Training Launcher with tmux
# Usage: ./start_training_advanced.sh [epochs] [batch_size] [session_name]

# Default values
EPOCHS=${1:-2}
BATCH_SIZE=${2:-16}
SESSION_NAME=${3:-training}

echo "🚀 Starting ACT Training Session..."
echo "   Session: $SESSION_NAME"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo ""

# Kill existing session if it exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "⚠️  Existing '$SESSION_NAME' session found. Killing it..."
    tmux kill-session -t "$SESSION_NAME"
    echo "✅ Old session killed."
    sleep 1
fi

echo "🆕 Creating new tmux session..."

# Start tmux session
tmux new-session -d -s "$SESSION_NAME"

# Left pane: training
tmux send-keys -t "$SESSION_NAME" "conda activate lerobot" C-m
tmux send-keys -t "$SESSION_NAME" "cd /home/maxboels/projects/Erewhon" C-m
tmux send-keys -t "$SESSION_NAME" "python src/policies/ACT/official_lerobot_trainer.py --data_dir src/robots/rover/episodes --output_dir ./outputs/lerobot_act --epochs $EPOCHS --batch_size $BATCH_SIZE --device cuda" C-m

# Split vertical and setup GPU monitor
tmux split-window -h -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "watch -n 1 nvidia-smi" C-m

# Split right pane horizontal and setup log monitor
tmux split-window -v -t "$SESSION_NAME"
tmux send-keys -t "$SESSION_NAME" "sleep 5 && tail -f outputs/lerobot_act/lerobot_act_*/logs/batch_metrics.csv" C-m

echo "✅ Session '$SESSION_NAME' created!"
echo ""
echo "📋 Quick commands:"
echo "   Detach: Ctrl+B then D"
echo "   Reattach: tmux attach -t $SESSION_NAME"
echo "   Kill session: tmux kill-session -t $SESSION_NAME"
echo ""
echo "🎯 Attaching to session..."
sleep 1

# Attach to session
tmux attach -t "$SESSION_NAME"
