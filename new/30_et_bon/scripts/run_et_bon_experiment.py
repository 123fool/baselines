"""
ET-BoN 多配置实验运行器
======================
测试不同 (N_initial → K_survivors, checkpoint_step) 配置，
与 LAS 和 BoN Weighted 做对比。

在服务器上运行:
  cd /home/wangchong/data/fwz/code/et_bon/scripts
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python run_et_bon_experiment.py --max-pairs 10

扩大规模:
  python run_et_bon_experiment.py --max-pairs 50 --configs best
"""

import os, sys, json, time, traceback, argparse
import numpy as np
import pandas as pd
import torch
from datetime import datetime

# ── Path setup (matching run_bon_fullscale.py layout) ──
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Server: code/et_bon/scripts/ -> code/src/ (../../src)
# Local: new/30_et_bon/scripts/ -> src/ (../../../src)
for _rel in [['..', '..', 'src'], ['..', '..', '..', 'src']]:
    _p = os.path.abspath(os.path.join(SCRIPT_DIR, *_rel))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
        break
sys.path.insert(0, SCRIPT_DIR)

from brlp import const, utils
from brlp.networks import init_autoencoder, init_latent_diffusion, init_controlnet
from brlp.sampling import sample_using_controlnet_and_z
from sampling_et_bon import sample_et_bon_weighted, sample_et_bon_with_details
from monai import transforms
from skimage.metrics import structural_similarity as ssim_fn

# ── Server paths (identical to run_bon_fullscale.py) ──
AEKL_CKPT = "/home/wangchong/data/fwz/output/innovation_5/ae/autoencoder-ep-2.pth"
DIFF_CKPT = "/home/wangchong/data/fwz/brlp-train/pretrained/latentdiffusion.pth"
CNET_CKPT = "/home/wangchong/data/fwz/output/innovation_2/controlnet/cnet-btr-ep-1.pth"
CSV_PATH  = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
OUTPUT_DIR = "/home/wangchong/data/fwz/output/verification/et_bon"

SCALE_FACTOR = 1.0469
LAS_M = 3

# ── Experiment configurations ──
ALL_CONFIGS = [
    # Group A: 8初始, 不同存活数
    {"name": "ET_8to3_cp10",   "n_initial": 8,  "n_survivors": 3, "checkpoint_step": 10, "decoded": True},
    {"name": "ET_8to3_cp15",   "n_initial": 8,  "n_survivors": 3, "checkpoint_step": 15, "decoded": True},
    {"name": "ET_8to5_cp10",   "n_initial": 8,  "n_survivors": 5, "checkpoint_step": 10, "decoded": True},
    {"name": "ET_8to5_cp15",   "n_initial": 8,  "n_survivors": 5, "checkpoint_step": 15, "decoded": True},
    # Group B: 16初始 (更激进筛选)
    {"name": "ET_16to4_cp10",  "n_initial": 16, "n_survivors": 4, "checkpoint_step": 10, "decoded": True},
    {"name": "ET_16to8_cp10",  "n_initial": 16, "n_survivors": 8, "checkpoint_step": 10, "decoded": True},
    {"name": "ET_16to8_cp15",  "n_initial": 16, "n_survivors": 8, "checkpoint_step": 15, "decoded": True},
    # Group C: 快速代理评分 (latent空间, 不解码)
    {"name": "ET_8to3_fast",   "n_initial": 8,  "n_survivors": 3, "checkpoint_step": 10, "decoded": False},
    {"name": "ET_16to4_fast",  "n_initial": 16, "n_survivors": 4, "checkpoint_step": 10, "decoded": False},
]

QUICK_CONFIGS = [
    {"name": "ET_8to3_cp10",   "n_initial": 8,  "n_survivors": 3, "checkpoint_step": 10, "decoded": True},
    {"name": "ET_8to5_cp10",   "n_initial": 8,  "n_survivors": 5, "checkpoint_step": 10, "decoded": True},
    {"name": "ET_16to8_cp10",  "n_initial": 16, "n_survivors": 8, "checkpoint_step": 10, "decoded": True},
]

BEST_CONFIGS = [
    {"name": "ET_8to5_cp10",   "n_initial": 8,  "n_survivors": 5, "checkpoint_step": 10, "decoded": True},
]


def ts():
    return datetime.now().strftime("[%H:%M:%S]")


def load_models(device):
    ae = init_autoencoder(AEKL_CKPT).to(device).eval()
    dm = init_latent_diffusion(DIFF_CKPT).to(device).eval()
    cn = init_controlnet(CNET_CKPT).to(device).eval()
    return ae, dm, cn


def _center_crop(vol, target_shape):
    starts = [(s - t) // 2 for s, t in zip(vol.shape, target_shape)]
    slices = tuple(slice(max(0, s), max(0, s) + t) for s, t in zip(starts, target_shape))
    return vol[slices]


def compute_metrics(pred_np, gt_np):
    min_shape = tuple(min(a, b) for a, b in zip(pred_np.shape, gt_np.shape))
    pred_np = _center_crop(pred_np, min_shape).clip(0, 1).astype(np.float64)
    gt_np = _center_crop(gt_np, min_shape).clip(0, 1).astype(np.float64)
    return {
        'ssim': float(ssim_fn(pred_np, gt_np, data_range=1.0)),
        'mae': float(np.abs(pred_np - gt_np).mean()),
    }


def prepare_pair(row, ae, device):
    """Prepare input data for one pair — identical to run_bon_fullscale.py."""
    loader = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    start_data = loader({'image_path': row['starting_image']})
    start_img = start_data['image'].unsqueeze(0).to(device)
    start_z = ae.encode(start_img)[0]
    start_z = transforms.DivisiblePad(k=4, mode='constant')(start_z.squeeze(0))

    follow_data = loader({'image_path': row['followup_image']})
    follow_img = follow_data['image']

    context = torch.tensor([
        row['followup_age'],
        row['sex'],
        row['starting_diagnosis'],
        row['followup_cerebral_cortex'],
        row['followup_hippocampus'],
        row['followup_amygdala'],
        row['followup_cerebral_white_matter'],
        row['followup_lateral_ventricle'],
    ], dtype=torch.float32)

    return start_z, row['starting_age'], context, follow_img


def run_experiment(args):
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "et_bon_experiment.log")

    def log(msg):
        line = f"{ts()} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log(f"ET-BoN Experiment Start | max_pairs={args.max_pairs} | configs={args.configs}")

    if args.configs == "all":
        configs = ALL_CONFIGS
    elif args.configs == "quick":
        configs = QUICK_CONFIGS
    elif args.configs == "best":
        configs = BEST_CONFIGS
    else:
        configs = ALL_CONFIGS

    log(f"Testing {len(configs)} configs: {[c['name'] for c in configs]}")

    ae, dm, cn = load_models(device)
    log("Models loaded.")

    df = pd.read_csv(args.csv_path)
    n_pairs = min(len(df), args.max_pairs)
    log(f"CSV has {len(df)} pairs, using {n_pairs}")

    results = {
        "experiment": "ET-BoN Multi-Config",
        "date": datetime.now().isoformat(),
        "n_pairs": n_pairs,
        "configs": {},
        "las_baseline": {},
    }

    # ── Phase A: LAS M=3 baseline ──
    log("=" * 60)
    log("Running LAS M=3 baseline...")
    las_results = []
    for idx in range(args.start_pair, n_pairs):
        row = df.iloc[idx].to_dict()
        subj = row.get('subject_id', row.get('ptid', f'pair_{idx}'))
        try:
            with torch.no_grad():
                start_z, start_a, context, gt_img = prepare_pair(row, ae, device)
                gt_np = gt_img.squeeze(0).numpy().clip(0, 1)

                t0 = time.time()
                las_img = sample_using_controlnet_and_z(
                    autoencoder=ae, diffusion=dm, controlnet=cn,
                    starting_z=start_z.float(), starting_a=start_a,
                    context=context.float(), device=device,
                    scale_factor=SCALE_FACTOR, average_over_n=LAS_M,
                    num_inference_steps=50, verbose=False,
                )
                las_time = time.time() - t0
                las_np = las_img.numpy().clip(0, 1)
                m = compute_metrics(las_np, gt_np)
                m['time_sec'] = round(las_time, 2)
                m['pair_idx'] = idx
                m['subject'] = subj
                las_results.append(m)
                log(f"  LAS pair {idx}: SSIM={m['ssim']:.4f} MAE={m['mae']:.4f} time={m['time_sec']:.1f}s")

            del start_z, start_a, context, gt_img, gt_np, las_img, las_np
            torch.cuda.empty_cache()
        except Exception as e:
            log(f"  LAS pair {idx} ERROR: {e}")
            las_results.append({"ssim": 0, "mae": 1, "time_sec": 0, "pair_idx": idx, "error": str(e)})
            torch.cuda.empty_cache()

    valid_las = [m for m in las_results if "error" not in m]
    results["las_baseline"] = {
        "method": "LAS_M3",
        "pairs": las_results,
        "avg_ssim": round(np.mean([m["ssim"] for m in valid_las]), 6) if valid_las else 0,
        "avg_mae": round(np.mean([m["mae"] for m in valid_las]), 6) if valid_las else 0,
        "avg_time": round(np.mean([m["time_sec"] for m in valid_las]), 2) if valid_las else 0,
    }
    log(f"LAS baseline: SSIM={results['las_baseline']['avg_ssim']:.4f} "
        f"MAE={results['las_baseline']['avg_mae']:.4f} "
        f"time={results['las_baseline']['avg_time']:.1f}s")

    # ── Phase B: Each ET-BoN configuration ──
    for ci, cfg in enumerate(configs):
        config_name = cfg["name"]
        log("=" * 60)
        total_steps = cfg["n_initial"] * cfg["checkpoint_step"] + \
                      cfg["n_survivors"] * (50 - cfg["checkpoint_step"])
        bon_steps = cfg["n_initial"] * 50
        log(f"Config [{ci+1}/{len(configs)}]: {config_name} "
            f"(N={cfg['n_initial']}→K={cfg['n_survivors']}, cp={cfg['checkpoint_step']}) "
            f"Steps: {total_steps} vs BoN {bon_steps} (save {(1-total_steps/bon_steps)*100:.0f}%)")

        cfg_metrics = []
        for idx in range(args.start_pair, n_pairs):
            row = df.iloc[idx].to_dict()
            subj = row.get('subject_id', row.get('ptid', f'pair_{idx}'))
            try:
                with torch.no_grad():
                    start_z, start_a, context, gt_img = prepare_pair(row, ae, device)
                    gt_np = gt_img.squeeze(0).numpy().clip(0, 1)

                    result = sample_et_bon_with_details(
                        autoencoder=ae, diffusion=dm, controlnet=cn,
                        starting_z=start_z.float(), starting_a=start_a,
                        context=context.float(), device=device,
                        scale_factor=SCALE_FACTOR,
                        n_initial=cfg["n_initial"],
                        n_survivors=cfg["n_survivors"],
                        checkpoint_step=cfg["checkpoint_step"],
                        use_decoded_proxy=cfg["decoded"],
                        num_inference_steps=50,
                        verbose=False,
                    )

                    gen_np = result["image"].numpy().clip(0, 1)
                    m = compute_metrics(gen_np, gt_np)
                    m['time_sec'] = result["timing"]["total_sec"]
                    m['proxy_scores'] = result["proxy_scores"]
                    m['survivor_indices'] = result["survivor_indices"]
                    m['final_scores'] = result["final_scores"]
                    m['pair_idx'] = idx
                    m['subject'] = subj
                    cfg_metrics.append(m)

                    # Compare with LAS
                    las_m = las_results[idx - args.start_pair]
                    las_ssim = las_m["ssim"] if "error" not in las_m else 0
                    delta = m["ssim"] - las_ssim
                    log(f"  Pair {idx}: SSIM={m['ssim']:.4f} (Δ vs LAS={delta:+.4f}) "
                        f"MAE={m['mae']:.4f} time={m['time_sec']:.1f}s "
                        f"survivors={m['survivor_indices']}")

                del start_z, start_a, context, gt_img, gt_np, result, gen_np
                torch.cuda.empty_cache()
            except Exception as e:
                log(f"  Pair {idx} ERROR: {e}")
                traceback.print_exc()
                cfg_metrics.append({"ssim": 0, "mae": 1, "time_sec": 0, "pair_idx": idx, "error": str(e)})
                torch.cuda.empty_cache()

        # Aggregate
        valid = [m for m in cfg_metrics if "error" not in m]
        if valid:
            avg_ssim = round(np.mean([m["ssim"] for m in valid]), 6)
            avg_mae = round(np.mean([m["mae"] for m in valid]), 6)
            avg_time = round(np.mean([m["time_sec"] for m in valid]), 2)

            et_ssims = [m["ssim"] for m in valid]
            las_ssims = [las_results[i]["ssim"] for i in range(len(valid))
                         if "error" not in las_results[i]]
            min_len = min(len(et_ssims), len(las_ssims))
            n_wins = sum(1 for a, b in zip(et_ssims[:min_len], las_ssims[:min_len]) if a > b)
            win_rate = round(n_wins / min_len * 100, 1) if min_len > 0 else 0

            from scipy import stats
            if min_len >= 5:
                _, p_val = stats.ttest_rel(et_ssims[:min_len], las_ssims[:min_len])
                p_val = round(float(p_val), 6)
            else:
                p_val = None
        else:
            avg_ssim = avg_mae = avg_time = win_rate = 0
            p_val = None

        results["configs"][config_name] = {
            "config": cfg,
            "pairs": cfg_metrics,
            "avg_ssim": avg_ssim,
            "avg_mae": avg_mae,
            "avg_time": avg_time,
            "vs_las_win_rate": win_rate,
            "vs_las_p_value": p_val,
            "step_savings_pct": round((1 - total_steps / bon_steps) * 100, 1),
        }
        log(f"  {config_name}: SSIM={avg_ssim:.4f} MAE={avg_mae:.4f} "
            f"time={avg_time:.1f}s vs_LAS={win_rate}% p={p_val}")

        # Save intermediate
        with open(os.path.join(output_dir, "et_bon_results.json"), "w") as f:
            json.dump(results, f, indent=2, default=str)

    # ── Final summary ──
    log("=" * 60)
    log("FINAL SUMMARY")
    log(f"  LAS M=3: SSIM={results['las_baseline']['avg_ssim']:.4f} time={results['las_baseline']['avg_time']:.1f}s")
    for name, r in results["configs"].items():
        log(f"  {name}: SSIM={r['avg_ssim']:.4f} time={r['avg_time']:.1f}s "
            f"vs_LAS={r['vs_las_win_rate']}% p={r['vs_las_p_value']} save={r['step_savings_pct']}%")

    if results["configs"]:
        best_name = max(results["configs"], key=lambda k: results["configs"][k]["avg_ssim"])
        best = results["configs"][best_name]
        log(f"\nBest: {best_name} SSIM={best['avg_ssim']:.4f}")

    with open(os.path.join(output_dir, "et_bon_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    log(f"Results saved to {output_dir}/et_bon_results.json")
    return results


def main():
    parser = argparse.ArgumentParser(description="ET-BoN Multi-Config Experiment")
    parser.add_argument("--csv-path", type=str, default=CSV_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--configs", type=str, default="quick",
                        choices=["all", "quick", "best"])
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
