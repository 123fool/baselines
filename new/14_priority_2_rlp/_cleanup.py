"""Kill duplicate training processes and keep only one."""
import paramiko

SERVER = "10.96.27.109"
PORT = 2638
USER = "wangchong"
PASS = "123456"

def ssh_exec(client, cmd, timeout=15):
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return out, err

def main():
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(SERVER, port=PORT, username=USER, password=PASS, timeout=15)

    # Get all train_controlnet_rlp processes
    print("=== Current training processes ===")
    out, _ = ssh_exec(client, "ps aux | grep train_controlnet_rlp | grep -v grep | awk '{print $2, $9, $11}' | head -20")
    print(out)
    
    # Count processes
    out, _ = ssh_exec(client, "ps aux | grep 'train_controlnet_rlp.py' | grep -v grep | grep -v 'awk' | wc -l")
    n_procs = int(out.strip()) if out.strip().isdigit() else 0
    print(f"\nTotal train processes: {n_procs}")

    if n_procs > 1:
        # Kill ALL training processes, then restart cleanly
        print("\nKilling ALL training processes (will restart 1 cleanly)...")
        out, err = ssh_exec(client, "pkill -f train_controlnet_rlp.py")
        print(f"  pkill output: {out} {err}")
        
        # Wait and verify
        import time
        time.sleep(2)
        out, _ = ssh_exec(client, "ps aux | grep train_controlnet_rlp | grep -v grep | wc -l")
        print(f"  Remaining processes: {out}")
        
        # Check what was accomplished so far
        out, _ = ssh_exec(client, "ls -la /home/wangchong/data/fwz/output/priority_2_rlp/rlp_only/controlnet/ 2>/dev/null")
        print(f"\n=== Saved checkpoints ===\n{out}")
        
        out, _ = ssh_exec(client, "tail -20 /home/wangchong/data/fwz/output/priority_2_rlp/rlp_only/train.log 2>/dev/null")
        print(f"\n=== Last log entries ===\n{out}")
        
        # Check GPU memory freed
        out, _ = ssh_exec(client, "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader")
        print(f"\n=== GPU memory after kill ===\n{out}")
    else:
        print("Only 1 process running, no cleanup needed.")
        out, _ = ssh_exec(client, "tail -5 /home/wangchong/data/fwz/output/priority_2_rlp/rlp_only/train.log 2>/dev/null")
        print(f"\n=== Log tail ===\n{out}")

    client.close()

if __name__ == '__main__':
    main()
