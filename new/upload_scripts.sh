#!/bin/bash
# Upload all method scripts to server
# Run this from the baselines directory
# Make sure sshpass is not needed — we'll use SSH key or manual auth

REMOTE="wangchong@10.96.27.109"
PORT=2638
REMOTE_DIR="/home/wangchong/data/fwz/code/brlp_src/scripts"
BASE="BrLP-main/new"

# Method C
scp -P $PORT "$BASE/21_method_c_identity/train_identity.py" "$REMOTE:$REMOTE_DIR/method_c_identity.py"
# Method D
scp -P $PORT "$BASE/22_method_d_frequency/train_frequency.py" "$REMOTE:$REMOTE_DIR/method_d_frequency.py"
# Unified eval
scp -P $PORT "$BASE/23_unified_eval/evaluate_all_methods.py" "$REMOTE:$REMOTE_DIR/evaluate_all_methods.py"
# Enhanced eval
scp -P $PORT "$BASE/19_enhanced_eval/evaluate_enhanced.py" "$REMOTE:$REMOTE_DIR/evaluate_enhanced.py"
# Method B eval
scp -P $PORT "$BASE/20_method_b_time_aware/evaluate_method_b.py" "$REMOTE:$REMOTE_DIR/evaluate_method_b.py"
# __init__.py for scripts
echo "" | ssh -p $PORT "$REMOTE" "cat > $REMOTE_DIR/__init__.py"

echo "All scripts uploaded!"
