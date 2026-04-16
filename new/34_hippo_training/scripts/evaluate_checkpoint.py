"""
Evaluate ControlNet checkpoint for hippocampus SSIM.

Generates samples using LAS + measures H-SSIM against ground truth.
Handles proper preprocessing alignment between generated (122,146,122)
and segmentation (120,144,120) spaces.

Usage:
    python evaluate_checkpoint.py \
        --dataset_csv B_mci.csv \
        --aekl_ckpt ae.pth \
        --diff_ckpt diffusion.pth \
        --cnet_ckpt checkpoint.pth \
        --n_test 5 --m_las 3

    For AE decoder fine-tuned model:
    python evaluate_checkpoint.py ... --ae_decoder_ckpt ae-hippo-dec.pth
"""
import os
import sys
import json
import argparse
import numpy as np
import nibabel as nib
import torch
from torch.cuda.amp import autocast
from monai import transforms
from monai.data.image_reader import NumpyReader
from skimage.metrics import structural_similarity as compute_ssim
import pandas as pd

# Patch torch.load for MONAI cache
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, '/home/wangchong/data/fwz/code')

from brlp import networks, utils, const
from brlp import get_dataset_from_pd, sample_using_controlnet_and_z

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HIPPO_LABELS = [17, 53]


def compute_hippo_ssim(img1, img2, seg_data):
    """SSIM within hippocampus bounding box, masked to hippocampus voxels."""
    hippo_mask = np.isin(seg_data, HIPPO_LABELS)
    if hippo_mask.sum() == 0:
        return float('nan')

    coords = np.where(hippo_mask)
    margin = 5
    slices = []
    for dim in range(3):
        lo = max(0, coords[dim].min() - margin)
        hi = min(hippo_mask.shape[dim], coords[dim].max() + margin + 1)
        slices.append(slice(lo, hi))

    roi1 = img1[slices[0], slices[1], slices[2]]
    roi2 = img2[slices[0], slices[1], slices[2]]
    mask_roi = hippo_mask[slices[0], slices[1], slices[2]]

    data_range = max(roi1.max(), roi2.max()) - min(roi1.min(), roi2.min())
    if data_range < 1e-8:
        return 1.0

    ssim_map = compute_ssim(roi1, roi2, data_range=data_range, full=True)[1]
    return float(ssim_map[mask_roi].mean())


def align_to_ae_space(image_np):
    """Crop image from (122,146,122) to AE input space (120,144,120)."""
    target = const.INPUT_SHAPE_AE  # (120, 144, 120)
    s = image_np.shape
    if s == target:
        return image_np

    # Center-crop to target (handles any shape >= target)
    starts = [(s[i] - target[i]) // 2 for i in range(3)]
    return image_np[
        starts[0]:starts[0]+target[0],
        starts[1]:starts[1]+target[1],
        starts[2]:starts[2]+target[2]
    ]


def concat_covariates(_dict):
    conditions = [
        _dict['followup_age'], _dict['sex'], _dict['followup_diagnosis'],
        _dict['followup_cerebral_cortex'], _dict['followup_hippocampus'],
        _dict['followup_amygdala'], _dict['followup_cerebral_white_matter'],
        _dict['followup_lateral_ventricle']
    ]
    _dict['context'] = torch.tensor(conditions)  # 1D (8,) — sampling.py does unsqueeze internally
    return _dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str)
    parser.add_argument('--ae_decoder_ckpt', default=None, type=str,
                        help='Optional: AE with fine-tuned decoder')
    parser.add_argument('--cache_dir', default='/tmp/eval_cache', type=str)
    parser.add_argument('--n_test', default=5, type=int)
    parser.add_argument('--m_las', default=3, type=int, help='LAS m parameter')
    parser.add_argument('--output_json', default=None, type=str)
    parser.add_argument('--label', default=None, type=str)
    args = parser.parse_args()

    label = args.label or os.path.basename(args.cnet_ckpt)
    print(f'\n=== Evaluating: {label} ===')

    # Models
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)

    # Optionally load fine-tuned AE decoder
    if args.ae_decoder_ckpt:
        print(f'  Loading fine-tuned AE: {args.ae_decoder_ckpt}')
        autoencoder.load_state_dict(
            torch.load(args.ae_decoder_ckpt, map_location='cpu'))

    diffusion = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet = networks.init_controlnet()
    controlnet.load_state_dict(
        torch.load(args.cnet_ckpt, map_location='cpu'))

    for p in diffusion.parameters():
        p.requires_grad = False
    for p in autoencoder.parameters():
        p.requires_grad = False

    autoencoder.to(DEVICE).eval()
    diffusion.to(DEVICE).eval()
    controlnet.to(DEVICE).eval()

    # Dataset
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'],
                              reader=npz_reader),
        transforms.EnsureChannelFirstD(
            keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(
            keys=['starting_latent', 'followup_latent'],
            k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates),
    ])

    df = pd.read_csv(args.dataset_csv)
    test_df = df[df.split == 'test'].head(args.n_test)
    testset = get_dataset_from_pd(test_df, transforms_fn, args.cache_dir)

    # Scale factor — compute from FIRST TRAINING sample for consistency with training
    train_df = df[df.split == 'train'].head(1)
    train_tmp = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    with torch.no_grad():
        with autocast(enabled=True):
            z = train_tmp[0]['followup_latent']
    scale_factor = 1 / torch.std(z)
    del train_tmp
    print(f'  Scale factor: {scale_factor:.4f}')

    # Preprocess ground truth exactly like AE training does
    gt_preprocess = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    results = []

    for idx in range(min(args.n_test, len(testset))):
        sample = testset[idx]

        # Ground truth: preprocessed same as AE input (120,144,120), [0,1]
        fu_img = gt_preprocess(sample['followup_image'])  # (1,120,144,120)
        fu_np = fu_img.squeeze().cpu().numpy()  # (120,144,120)

        seg_data = nib.load(sample['followup_segm']).get_fdata()

        # Generate with LAS
        starting_z = sample['starting_latent'] * scale_factor
        context = sample['context'].flatten()  # ensure 1D (8,) — sampling.py adds dims
        starting_a = sample['starting_age']

        pred = sample_using_controlnet_and_z(
            autoencoder=autoencoder, diffusion=diffusion,
            controlnet=controlnet, starting_z=starting_z,
            starting_a=starting_a, context=context,
            device=DEVICE, scale_factor=scale_factor,
            average_over_n=args.m_las, verbose=False
        )
        pred_np = pred.squeeze().numpy()

        # Align generated image to AE space (crop from 122,146,122 to 120,144,120)
        pred_np = align_to_ae_space(pred_np)

        # Clip to [0,1] for fair comparison (AE output range may vary)
        pred_np = np.clip(pred_np, 0, 1)

        h_ssim = compute_hippo_ssim(fu_np, pred_np, seg_data)
        overall = compute_ssim(fu_np, pred_np, data_range=1.0)

        # Also generate 2 more individual candidates for oracle
        oracle_h = h_ssim
        for _ in range(2):
            pred_i = sample_using_controlnet_and_z(
                autoencoder=autoencoder, diffusion=diffusion,
                controlnet=controlnet, starting_z=starting_z,
                starting_a=starting_a, context=context,
                device=DEVICE, scale_factor=scale_factor,
                average_over_n=1, verbose=False
            )
            pred_i_np = np.clip(align_to_ae_space(pred_i.squeeze().numpy()), 0, 1)
            h_i = compute_hippo_ssim(fu_np, pred_i_np, seg_data)
            oracle_h = max(oracle_h, h_i)

        results.append({
            'pair': idx,
            'h_ssim_las': h_ssim,
            'overall_ssim': overall,
            'oracle_h_ssim': oracle_h
        })
        print(f'  Pair {idx}: H-SSIM={h_ssim:.4f}, Overall={overall:.4f}, '
              f'Oracle-H={oracle_h:.4f}')

    # Summary
    avg_h = np.mean([r['h_ssim_las'] for r in results])
    std_h = np.std([r['h_ssim_las'] for r in results])
    avg_o = np.mean([r['overall_ssim'] for r in results])
    avg_oracle = np.mean([r['oracle_h_ssim'] for r in results])

    print(f'\n=== {label} Results ===')
    print(f'H-SSIM (LAS m={args.m_las}): {avg_h:.4f} +/- {std_h:.4f}')
    print(f'Overall SSIM: {avg_o:.4f}')
    print(f'Oracle H-SSIM: {avg_oracle:.4f}')

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump({
                'label': label,
                'checkpoint': args.cnet_ckpt,
                'ae_decoder_ckpt': args.ae_decoder_ckpt,
                'n_test': len(results),
                'm_las': args.m_las,
                'avg_h_ssim': avg_h, 'std_h_ssim': std_h,
                'avg_overall': avg_o, 'avg_oracle_h_ssim': avg_oracle,
                'per_pair': results
            }, f, indent=2)
        print(f'Saved: {args.output_json}')


if __name__ == '__main__':
    main()
