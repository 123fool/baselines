"""Evaluate Priority 4 with epoch 2 checkpoint (best validation loss)."""
import paramiko, time

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect("10.96.27.109", port=2638, username="wangchong", password="123456")

cmd = (
    "export CUDA_VISIBLE_DEVICES=1; "
    "cd /home/wangchong/data/fwz/code/priority_4_palm_tel; "
    "PYTHONPATH=/home/wangchong/data/fwz/code/priority_4_palm_tel/brlp_src:"
    "/home/wangchong/data/fwz/code/priority_4_palm_tel/src:"
    "/home/wangchong/data/fwz/code/priority_4_palm_tel/innov2_src "
    "nohup /home/wangchong/miniconda3/envs/fwz/bin/python scripts/evaluate_palm_tel.py "
    "--dataset_csv /home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv "
    "--aekl_ckpt /home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth "
    "--diff_ckpt /home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth "
    "--cnet_ckpt /home/wangchong/data/fwz/output/priority_4_palm_tel/controlnet/cnet-btc-palm-tel-ep-2.pth "
    "--output_dir /home/wangchong/data/fwz/output/priority_4_palm_tel/eval "
    "--max_pairs 50 "
    "--model_name btc_palm_tel_ep2 "
    "> /home/wangchong/data/fwz/output/priority_4_palm_tel/eval_ep2.log 2>&1 &"
)

_, o, e = ssh.exec_command(cmd)
o.read()
time.sleep(5)

_, o, _ = ssh.exec_command("ps aux | grep evaluate_palm_tel | grep -v grep")
proc = o.read().decode().strip()
if proc:
    print("Evaluation started!")
    print(proc[:150])
else:
    print("Check log:")
    _, o, _ = ssh.exec_command("tail -5 /home/wangchong/data/fwz/output/priority_4_palm_tel/eval_ep2.log")
    print(o.read().decode())

ssh.close()
