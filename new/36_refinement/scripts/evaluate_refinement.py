"""
Section 36: Refinement Network Evaluation
==========================================
Evaluate the refinement network on multi-region SSIM metrics.

Pipeline:
  1. Run standard BrLP pipeline: BL → diffusion → AE decode → pred_AD
  2. Load real BL image
  3. Refinement: (pred_AD, BL) → RefinementUNet → refined_AD
  4. Compute multi-region SSIM against real FU

Usage:
    python evaluate_refinement.py \
        --csv /path/to/B_mci.csv \
        --aekl_ckpt autoencoder.pth \
        --diff_ckpt diffusion.pth \
        --cnet_ckpt controlnet.pth \
        --ref_ckpt refnet.pth \
        --n_test 5 --m_las 3 --gpu 0 \
        --output_json results.json
"""
import os, sys, json, argparse, warnings
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from monai import transforms
from monai.data.image_reader import NumpyReader
from skimage.metrics import structural_similarity as compute_ssim

# Patch torch.load
_orig_torch_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_load

sys.path.insert(0, '/home/wangchong/data/fwz/code')

try:
    from monai.data.meta_tensor import MetaTensor
    from monai.utils.enums import MetaKeys, SpaceKeys
    torch.serialization.add_safe_globals([MetaTensor, MetaKeys, SpaceKeys])
except Exception:
    pass

from brlp import networks, utils, const
from brlp import get_dataset_from_pd, sample_using_controlnet_and_z

warnings.filterwarnings("ignore")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
AD_PRIMARY = ['hippocampus', 'amygdala', 'thalamus', 'lateral_ventricle']


# ── Refinement U-Net (must match train_refinement.py) ──

class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.InstanceNorm3d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(ch, ch, 3, padding=1),
        )
    def forward(self, x):
        return x + self.net(x)


class RefinementUNet3D(nn.Module):
    def __init__(self, in_ch=2, out_ch=1, base_ch=32):
        super().__init__()
        c = base_ch
        self.enc0 = nn.Sequential(nn.Conv3d(in_ch, c, 3, padding=1), ResBlock3D(c))
        self.down0 = nn.Conv3d(c, c*2, 3, stride=2, padding=1)
        self.enc1 = nn.Sequential(ResBlock3D(c*2))
        self.down1 = nn.Conv3d(c*2, c*4, 3, stride=2, padding=1)
        self.mid = nn.Sequential(ResBlock3D(c*4), ResBlock3D(c*4))
        self.up1 = nn.ConvTranspose3d(c*4, c*2, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv3d(c*4, c*2, 3, padding=1), ResBlock3D(c*2))
        self.up0 = nn.ConvTranspose3d(c*2, c, 2, stride=2)
        self.dec0 = nn.Sequential(nn.Conv3d(c*2, c, 3, padding=1), ResBlock3D(c))
        self.out_conv = nn.Conv3d(c, out_ch, 1)

    def forward(self, x):
        pred_ad = x[:, 0:1]
        e0 = self.enc0(x)
        e1 = self.enc1(self.down0(e0))
        m = self.mid(self.down1(e1))
        d1 = self.dec1(torch.cat([self.up1(m), e1], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e0], dim=1))
        return pred_ad + self.out_conv(d0)


# ── Helper functions ──

def compute_region_ssim(img1, img2, seg_data, labels, margin=5):
    mask = np.isin(seg_data, labels)
    if mask.sum() == 0:
        return float('nan'), 0
    coords = np.where(mask)
    slices = []
    for dim in range(3):
        lo = max(0, coords[dim].min() - margin)
        hi = min(mask.shape[dim], coords[dim].max() + margin + 1)
        slices.append(slice(lo, hi))
    roi1, roi2 = img1[slices[0], slices[1], slices[2]], img2[slices[0], slices[1], slices[2]]
    mask_roi = mask[slices[0], slices[1], slices[2]]
    data_range = max(roi1.max(), roi2.max()) - min(roi1.min(), roi2.min())
    if data_range < 1e-8:
        return 1.0, int(mask.sum())
    ssim_map = compute_ssim(roi1, roi2, data_range=data_range, full=True)[1]
    return float(ssim_map[mask_roi].mean()), int(mask.sum())


def align_to_ae_space(image_np):
    target = const.INPUT_SHAPE_AE
    s = image_np.shape
    if s == target:
        return image_np
    starts = [(s[i] - target[i]) // 2 for i in range(3)]
    return image_np[starts[0]:starts[0]+target[0],
                    starts[1]:starts[1]+target[1],
                    starts[2]:starts[2]+target[2]]


def load_and_crop(path, target_shape=const.INPUT_SHAPE_AE):
    img = nib.load(path).get_fdata().astype(np.float32)
    s = img.shape
    if s == target_shape:
        return img
    if s == (122, 146, 122):
        return img[1:121, 1:145, 1:121]
    starts = [(s[i] - target_shape[i]) // 2 for i in range(3)]
    return img[starts[0]:starts[0]+target_shape[0],
               starts[1]:starts[1]+target_shape[1],
               starts[2]:starts[2]+target_shape[2]]


def normalize_01(img):
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return img
    return (img - mn) / (mx - mn)


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
    parser.add_argument('--csv', required=True, type=str)
    parser.add_argument('--aekl_ckpt', required=True, type=str)
    parser.add_argument('--diff_ckpt', required=True, type=str)
    parser.add_argument('--cnet_ckpt', required=True, type=str)
    parser.add_argument('--ae_decoder_ckpt', default=None, type=str)
    parser.add_argument('--ref_ckpt', default=None, type=str, help='Refinement network checkpoint')
    parser.add_argument('--base_ch', default=32, type=int, help='Refinement net base channels')
    parser.add_argument('--cache_dir', default='/tmp/eval_cache_ref', type=str)
    parser.add_argument('--n_test', default=5, type=int)
    parser.add_argument('--m_las', default=3, type=int)
    parser.add_argument('--output_json', default=None, type=str)
    parser.add_argument('--label', default=None, type=str)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    os.makedirs(args.cache_dir, exist_ok=True)

    label = args.label or 'refinement_eval'
    print(f'\n=== Refinement Evaluation: {label} ===')

    # ── Load models ──
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    if args.ae_decoder_ckpt:
        print(f'  AE decoder: {args.ae_decoder_ckpt}')
        autoencoder.load_state_dict(
            torch.load(args.ae_decoder_ckpt, map_location='cpu'))

    diffusion = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet = networks.init_controlnet()
    controlnet.load_state_dict(torch.load(args.cnet_ckpt, map_location='cpu'))

    for m in [diffusion, autoencoder, controlnet]:
        for p in m.parameters():
            p.requires_grad = False
        m.to(device).eval()

    # Load refinement network (if provided)
    refnet = None
    if args.ref_ckpt and os.path.exists(args.ref_ckpt):
        print(f'  Refinement net: {args.ref_ckpt}')
        refnet = RefinementUNet3D(in_ch=2, out_ch=1, base_ch=args.base_ch).to(device)
        refnet.load_state_dict(torch.load(args.ref_ckpt, map_location='cpu'))
        refnet.eval()
        for p in refnet.parameters():
            p.requires_grad = False

    # ── Dataset ──
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'],
                              reader=npz_reader),
        transforms.EnsureChannelFirstD(
            keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(
            keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates),
    ])

    df = pd.read_csv(args.csv)
    test_df = df[df.split == 'test'].head(args.n_test) if 'split' in df.columns else df.tail(args.n_test)
    testset = get_dataset_from_pd(test_df, transforms_fn, args.cache_dir)

    # Scale factor
    train_df = df[df.split == 'train'].head(1) if 'split' in df.columns else df.head(1)
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
        transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    # ── Evaluation loop ──
    all_results = []
    test_df_reset = test_df.reset_index(drop=True)

    for idx in range(min(args.n_test, len(testset))):
        sample = testset[idx]

        # Load ground truth FU
        fu_img = gt_preprocess(sample['followup_image'])
        fu_np = fu_img.squeeze().cpu().numpy()

        # Load segmentation
        seg_data = nib.load(sample['followup_segm']).get_fdata()

        # Generate prediction via standard pipeline
        starting_z = sample['starting_latent'] * scale_factor
        context = sample['context'].flatten()
        starting_a = sample['starting_age']

        pred = sample_using_controlnet_and_z(
            autoencoder=autoencoder, diffusion=diffusion,
            controlnet=controlnet, starting_z=starting_z,
            starting_a=starting_a, context=context,
            device=device, scale_factor=scale_factor,
            average_over_n=args.m_las, verbose=False
        )
        pred_np = np.clip(align_to_ae_space(pred.squeeze().numpy()), 0, 1)

        # Apply refinement network if available
        refined_np = pred_np
        if refnet is not None:
            # Load real BL image
            bl_path = test_df_reset.iloc[idx]['starting_image']
            bl_np = normalize_01(load_and_crop(bl_path))
            bl_tensor = torch.tensor(bl_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            pred_tensor = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            ref_input = torch.cat([pred_tensor, bl_tensor], dim=1)
            with torch.no_grad():
                with autocast(enabled=True):
                    refined_tensor = refnet(ref_input)
            refined_np = np.clip(refined_tensor.squeeze().cpu().float().numpy(), 0, 1)

        # ── Compute metrics ──
        overall = compute_ssim(fu_np, refined_np, data_range=1.0)

        region_results = {}
        for rname, rlabels in REGION_LABELS.items():
            ssim_val, nvoxels = compute_region_ssim(fu_np, refined_np, seg_data, rlabels)
            region_results[rname] = {'ssim': ssim_val, 'nvoxels': nvoxels}

        ad_ssims = [region_results[r]['ssim'] for r in AD_PRIMARY
                    if not np.isnan(region_results[r]['ssim'])]
        ad_composite = np.mean(ad_ssims) if ad_ssims else float('nan')

        result = {
            'pair': idx,
            'overall_ssim': overall,
            'ad_composite': ad_composite,
            'regions': region_results,
        }
        all_results.append(result)

        print(f'  Pair {idx}: Overall={overall:.4f}, AD-Comp={ad_composite:.4f}')
        for rname in AD_PRIMARY:
            r = region_results[rname]
            print(f'    {rname}: SSIM={r["ssim"]:.4f} ({r["nvoxels"]} vox)')

    # ── Summary ──
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
        'cnet_ckpt': args.cnet_ckpt,
        'ae_decoder_ckpt': args.ae_decoder_ckpt,
        'ref_ckpt': args.ref_ckpt,
        'n_test': len(all_results),
        'm_las': args.m_las,
        'base_ch': args.base_ch,
    }

    print(f'{"Region":<22} {"Mean SSIM":>10} {"±Std":>8}')
    print('-' * 44)
    for key in ['overall', 'ad_composite'] + list(REGION_LABELS.keys()):
        vals = summary[key]
        if vals:
            mean_v, std_v = np.mean(vals), np.std(vals)
            report[f'{key}_mean'] = float(mean_v)
            report[f'{key}_std'] = float(std_v)
            print(f'{key:<22} {mean_v:>10.4f} {std_v:>8.4f}')

    report['per_pair'] = all_results

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nSaved: {args.output_json}')


if __name__ == '__main__':
    main()
