"""Monitor RLP evaluation progress."""
import paramiko, sys

HOST = '10.96.27.109'
PORT = 2638
USER = 'wangchong'
PASS = '123456'
EVAL_DIR = '/home/wangchong/data/fwz/output/priority_2_rlp/eval'

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, PORT, USER, PASS, timeout=10)

# GPU status
_, stdout, _ = client.exec_command('nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits')
print("GPU:", stdout.read().decode().strip())

# Processes
_, stdout, _ = client.exec_command('ps aux | grep evaluate_rlp | grep -v grep | wc -l')
n = stdout.read().decode().strip()
print(f"Eval processes: {n}")

# Log - key lines
_, stdout, _ = client.exec_command(f"grep -E 'Priority 2|Scale factor|Evaluating|Error|Results|overall_|hippocampus|amygdala|roi_|Pairs evaluated|saved to' {EVAL_DIR}/eval.log 2>/dev/null | tail -40")
key = stdout.read().decode().strip()
if key:
    print(f"\n=== Key Info ===")
    print(key)

# Current progress (last few lines)
_, stdout, _ = client.exec_command(f"tail -5 {EVAL_DIR}/eval.log 2>/dev/null")
tail = stdout.read().decode().strip()
if tail:
    print(f"\n=== Current ===")
    print(tail)

# Check for output files
_, stdout, _ = client.exec_command(f"ls -la {EVAL_DIR}/*.json {EVAL_DIR}/*.csv 2>/dev/null")
files = stdout.read().decode().strip()
if files:
    print(f"\n=== Output Files ===")
    print(files)

client.close()
