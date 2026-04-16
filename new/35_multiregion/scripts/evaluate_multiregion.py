"""
Multi-Region SSIM Evaluation for MCI→AD Brain Structures.

Measures SSIM for each AD-relevant brain region individually:
  - Hippocampus (labels 17, 53)
  - Amygdala (labels 18, 54)
  - Thalamus (labels 10, 49)
  - Lateral Ventricle (labels 4, 43)
  - Caudate (labels 11, 50)
  - Putamen (labels 12, 51)
  - Cerebral Cortex (labels 3, 42)

Also computes a composite "AD-Region SSIM" averaging the key regions.

Usage:
    python evaluate_multiregion.py \
        --dataset_csv B_mci.csv \
        --aekl_ckpt ae.pth \
        --diff_ckpt diffusion.pth \
        --cnet_ckpt checkpoint.pth \
        --n_test 5 --m_las 3
"""
import os, sys, json, argparse
import numpy as np
import nibabel as nib
import torch
from torch.cuda.amp import autocast
from monai import transforms
from monai.data.image_reader import NumpyReader
from skimage.metrics import structural_similarity as compute_ssim
import pandas as pd

# Patch torch.load for PyTorch 2.6 + MONAI cache
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, '/home/wangchong/data/fwz/code')

from brlp import networks, utils, const
from brlp import get_dataset_from_pd, sample_using_controlnet_and_z

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# AD-relevant brain regions with SynthSeg label pairs (left, right)
REGION_LABELS = {
    'hippocampus':       [17, 53],
    'amygdala':          [18, 54],
    'thalamus':          [10, 49],
    'lateral_ventricle': [4, 43],
    'caudate':           [11, 50],
    'putamen':           [12, 51],
    'cerebral_cortex':   [3, 42],
    'cerebral_wm':       [2, 41],
    'pallidum':          [13, 52],
}

# Primary AD regions for composite score
AD_PRIMARY = ['hippocampus', 'amygdala', 'thalamus', 'lateral_ventricle']


def compute_region_ssim(img1, img2, seg_data, labels, margin=5):
    """Compute SSIM within a brain region's bounding box, masked to region voxels."""
    mask = np.isin(seg_data, labels)
    if mask.sum() == 0:
        return float('nan'), 0

    coords = np.where(mask)
    slices = []
    for dim in range(3):
        lo = max(0, coords[dim].min() - margin)
        hi = min(mask.shape[dim], coords[dim].max() + margin + 1)
        slices.append(slice(lo, hi))

    roi1 = img1[slices[0], slices[1], slices[2]]
    roi2 = img2[slices[0], slices[1], slices[2]]
    mask_roi = mask[slices[0], slices[1], slices[2]]

    data_range = max(roi1.max(), roi2.max()) - min(roi1.min(), roi2.min())
    if data_range < 1e-8:
        return 1.0, int(mask.sum())

    ssim_map = compute_ssim(roi1, roi2, data_range=data_range, full=True)[1]
    return float(ssim_map[mask_roi].mean()), int(mask.sum())


def align_to_ae_space(image_np):
    """Crop image from (122,146,122) to AE input space (120,144,120)."""
    target = const.INPUT_SHAPE_AE
    s = image_np.shape
    if s == target:
        return image_np
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
    _dict['context'] = torch.tensor(conditions)
    return _dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str)
    parser.add_argument('--ae_decoder_ckpt', default=None, type=str)
    parser.add_argument('--cache_dir', default='/tmp/eval_cache_mr', type=str)
    parser.add_argument('--n_test', default=5, type=int)
    parser.add_argument('--m_las', default=3, type=int)
    parser.add_argument('--output_json', default=None, type=str)
    parser.add_argument('--label', default=None, type=str)
    args = parser.parse_args()

    # Ensure cache dir exists
    os.makedirs(args.cache_dir, exist_ok=True)

    label = args.label or 'multiregion_eval'
    print(f'\n=== Multi-Region Evaluation: {label} ===')

    # Load models
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    if args.ae_decoder_ckpt:
        print(f'  Loading fine-tuned AE: {args.ae_decoder_ckpt}')
        autoencoder.load_state_dict(
            torch.load(args.ae_decoder_ckpt, map_location='cpu'))

    diffusion = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet = networks.init_controlnet()
    controlnet.load_state_dict(
        torch.load(args.cnet_ckpt, map_location='cpu'))

    for m in [diffusion, autoencoder, controlnet]:
        for p in m.parameters():
            p.requires_grad = False
        m.to(DEVICE).eval()

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

    # Scale factor from first training sample
    train_df = df[df.split == 'train'].head(1)
    train_tmp = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    with torch.no_grad():
        with autocast(enabled=True):
            z = train_tmp[0]['followup_latent']
    scale_factor = 1 / torch.std(z)
    del train_tmp
    print(f'  Scale factor: {scale_factor:.4f}')

    # GT preprocessing
    gt_preprocess = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(
            spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    all_results = []

    for idx in range(min(args.n_test, len(testset))):
        sample = testset[idx]

        fu_img = gt_preprocess(sample['followup_image'])
        fu_np = fu_img.squeeze().cpu().numpy()

        seg_data = nib.load(sample['followup_segm']).get_fdata()

        # Generate with LAS
        starting_z = sample['starting_latent'] * scale_factor
        context = sample['context'].flatten()
        starting_a = sample['starting_age']

        pred = sample_using_controlnet_and_z(
            autoencoder=autoencoder, diffusion=diffusion,
            controlnet=controlnet, starting_z=starting_z,
            starting_a=starting_a, context=context,
            device=DEVICE, scale_factor=scale_factor,
            average_over_n=args.m_las, verbose=False
        )
        pred_np = np.clip(align_to_ae_space(pred.squeeze().numpy()), 0, 1)

        # Overall SSIM
        overall = compute_ssim(fu_np, pred_np, data_range=1.0)

        # Per-region SSIM
        region_results = {}
        for rname, rlabels in REGION_LABELS.items():
            ssim_val, nvoxels = compute_region_ssim(fu_np, pred_np, seg_data, rlabels)
            region_results[rname] = {'ssim': ssim_val, 'nvoxels': nvoxels}

        # AD composite score (average of primary AD regions)
        ad_ssims = [region_results[r]['ssim'] for r in AD_PRIMARY
                    if not np.isnan(region_results[r]['ssim'])]
        ad_composite = np.mean(ad_ssims) if ad_ssims else float('nan')

        result = {
            'pair': idx,
            'overall_ssim': overall,
            'ad_composite': ad_composite,
            'regions': region_results
        }
        all_results.append(result)

        print(f'  Pair {idx}: Overall={overall:.4f}, AD-Composite={ad_composite:.4f}')
        for rname in AD_PRIMARY:
            r = region_results[rname]
            print(f'    {rname}: SSIM={r["ssim"]:.4f} ({r["nvoxels"]} voxels)')

    # Summary
    print(f'\n=== {label} Summary ===')
    summary = {'overall': [], 'ad_composite': []}
    for rname in REGION_LABELS:
        summary[rname] = []

    for r in all_results:
        summary['overall'].append(r['overall_ssim'])
        summary['ad_composite'].append(r['ad_composite'])
        for rname in REGION_LABELS:
            if rname in r['regions'] and not np.isnan(r['regions'][rname]['ssim']):
                summary[rname].append(r['regions'][rname]['ssim'])

    report = {
        'label': label,
        'checkpoint': args.cnet_ckpt,
        'ae_decoder_ckpt': args.ae_decoder_ckpt,
        'n_test': len(all_results),
        'm_las': args.m_las,
    }

    print(f'{"Region":<22} {"Mean SSIM":>10} {"±Std":>8} {"Voxels":>8}')
    print('-' * 52)

    for key in ['overall', 'ad_composite'] + list(REGION_LABELS.keys()):
        vals = summary[key]
        if vals:
            mean_v = np.mean(vals)
            std_v = np.std(vals)
            report[f'{key}_mean'] = float(mean_v)
            report[f'{key}_std'] = float(std_v)
            # Average voxel count for regions
            vox = ''
            if key in REGION_LABELS:
                vox_list = [r['regions'][key]['nvoxels'] for r in all_results
                           if key in r['regions']]
                vox = str(int(np.mean(vox_list))) if vox_list else ''
            print(f'{key:<22} {mean_v:>10.4f} {std_v:>8.4f} {vox:>8}')

    report['per_pair'] = all_results

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'Saved: {args.output_json}')


if __name__ == '__main__':
    main()
