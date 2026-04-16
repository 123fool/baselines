#!/usr/bin/env python3
"""
Verification Experiment Runner — Server Deployment Script.

Run from server: conda activate fwz && python run_verification.py

This script:
  1. Checks model checkpoints exist
  2. Runs all verification methods sequentially
  3. Logs progress in real-time
  4. Saves results for dashboard monitoring
"""

import os
import sys
import json
import subprocess
from datetime import datetime

# ─── Paths (server) ────────────────────────────────────────────
BASE_DIR = "/home/wangchong/data/fwz"
CODE_DIR = os.path.join(BASE_DIR, "code", "verification", "scripts")
DATA_CSV = os.path.join(BASE_DIR, "output", "innovation_5", "prepared", "B_mci.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "verification")

# Model checkpoints — using Innovation 2 BTR (best current model)
# AE: Innovation 5 improved decoder (autoencoder-ep-2.pth)
AEKL_CKPT = os.path.join(BASE_DIR, "output", "innovation_5", "ae", "autoencoder-ep-2.pth")
# Diffusion: pretrained latent diffusion UNet
DIFF_CKPT = os.path.join(BASE_DIR, "brlp-train", "pretrained", "latentdiffusion.pth")
# ControlNet: BTR epoch 1 (best per evaluation)
CNET_CKPT = os.path.join(BASE_DIR, "output", "innovation_2", "controlnet",
                          "cnet-btr-ep-1.pth")

# Fallback checkpoints
CNET_CKPT_ALT = os.path.join(BASE_DIR, "output", "innovation_2", "controlnet",
                               "cnet-btr-ep-2.pth")


def log(msg):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    log_file = os.path.join(OUTPUT_DIR, "runner.log")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(line + '\n')


def check_paths():
    """Verify all required files exist before running."""
    missing = []
    for name, path in [
        ("Dataset CSV", DATA_CSV),
        ("AutoencoderKL", AEKL_CKPT),
        ("Diffusion UNet", DIFF_CKPT),
    ]:
        if not os.path.exists(path):
            missing.append(f"{name}: {path}")

    # Check ControlNet (try primary, then fallback)
    global CNET_CKPT
    if not os.path.exists(CNET_CKPT):
        if os.path.exists(CNET_CKPT_ALT):
            log(f"Primary ControlNet not found, using fallback: {CNET_CKPT_ALT}")
            CNET_CKPT = CNET_CKPT_ALT
        else:
            missing.append(f"ControlNet: {CNET_CKPT} (and alt: {CNET_CKPT_ALT})")

    if missing:
        log("ERROR: Missing files:")
        for m in missing:
            log(f"  - {m}")
        return False
    return True


def run_experiment(exp_name, methods, n_candidates, max_pairs, las_m=3):
    """Run a single experiment configuration."""
    log(f"\n{'='*60}")
    log(f"Experiment: {exp_name}")
    log(f"Methods: {methods}")
    log(f"N candidates: {n_candidates}, Max pairs: {max_pairs}, LAS m: {las_m}")
    log(f"{'='*60}")

    exp_dir = os.path.join(OUTPUT_DIR, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(CODE_DIR, "evaluate_verification.py"),
        "--dataset_csv", DATA_CSV,
        "--aekl_ckpt", AEKL_CKPT,
        "--diff_ckpt", DIFF_CKPT,
        "--cnet_ckpt", CNET_CKPT,
        "--output_dir", exp_dir,
        "--max_pairs", str(max_pairs),
        "--n_candidates", str(n_candidates),
        "--las_m", str(las_m),
        "--model_name", exp_name,
    ]
    if methods:
        cmd.extend(["--methods", methods])

    log(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=False, text=True, timeout=7200)
        if result.returncode == 0:
            log(f"Experiment {exp_name} completed successfully")
        else:
            log(f"Experiment {exp_name} failed with code {result.returncode}")
    except subprocess.TimeoutExpired:
        log(f"Experiment {exp_name} timed out (2h limit)")
    except Exception as e:
        log(f"Experiment {exp_name} error: {e}")


def main():
    log("=== Verification Mechanism Evaluation Runner ===")
    log(f"Python: {sys.executable}")
    log(f"Working dir: {os.getcwd()}")

    if not check_paths():
        log("Aborting due to missing files.")
        sys.exit(1)

    log(f"Dataset CSV: {DATA_CSV}")
    log(f"AE checkpoint: {AEKL_CKPT}")
    log(f"Diffusion checkpoint: {DIFF_CKPT}")
    log(f"ControlNet checkpoint: {CNET_CKPT}")

    # ─── Experiment 1: Quick comparison (5 pairs, small N) ───
    run_experiment(
        exp_name="quick_compare",
        methods="las,single,bon_best1,bon_topk,bon_weighted",
        n_candidates=5,
        max_pairs=5,
        las_m=3,
    )

    # ─── Experiment 2: Full BoN comparison (20 pairs, N=8) ───
    run_experiment(
        exp_name="bon_n8",
        methods="las,single,bon_best1,bon_topk,bon_weighted",
        n_candidates=8,
        max_pairs=20,
        las_m=3,
    )

    # ─── Experiment 3: Round-Trip verification (10 pairs) ───
    run_experiment(
        exp_name="roundtrip",
        methods="las,roundtrip_bon",
        n_candidates=5,
        max_pairs=10,
        las_m=3,
    )

    # ─── Experiment 4: N ablation (N=4,8,16 on 10 pairs) ───
    for n in [4, 8, 16]:
        run_experiment(
            exp_name=f"ablation_n{n}",
            methods="bon_best1",
            n_candidates=n,
            max_pairs=10,
        )

    log("\n=== All experiments completed ===")

    # Collect all summaries into one comparison file
    all_summaries = {}
    for exp_name in os.listdir(OUTPUT_DIR):
        summary_path = os.path.join(OUTPUT_DIR, exp_name,
                                    f"summary_{exp_name}.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                all_summaries[exp_name] = json.load(f)

    master_path = os.path.join(OUTPUT_DIR, "master_summary.json")
    with open(master_path, 'w') as f:
        json.dump(all_summaries, f, indent=2, default=str)
    log(f"Master summary: {master_path}")


if __name__ == '__main__':
    main()
