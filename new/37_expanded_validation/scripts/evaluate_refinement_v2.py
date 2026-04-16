"""
Section 37: Full-Scale Refinement Evaluation (50 test subjects)
================================================================
Key improvements over S36:
1. Uses ALL 50 test subjects (vs 5)
2. Computes 95% confidence intervals
3. Per-subject metrics saved for statistical analysis
4. Proper AD-composite with all 4 primary AD regions
5. Progress logging for dashboard monitoring

Usage:
    python evaluate_refinement_v2.py \
        --csv /path/to/B_mci.csv \
        --aekl_ckpt autoencoder.pth \
        --diff_ckpt diffusion.pth \
        --cnet_ckpt controlnet.pth \
        --ref_ckpt refnet.pth \
        --n_test 50 --m_las 3 --gpu 0 \
        --output_json results.json
"""
import os, sys, json, argparse, warnings, time
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


# Network (same as train_refinement_v2.py)
class ResBlock3D(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.InstanceNorm3d(ch), nn.LeakyReLU(0.2, True),
            nn.Conv3d(ch, ch, 3, padding=1),
            nn.InstanceNorm3d(ch), nn.LeakyReLU(0.2, True),
            nn.Conv3d(ch, ch, 3, padding=1),
        )
    def forward(self, x): return x + self.net(x)


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
    roi1 = img1[slices[0], slices[1], slices[2]]
    roi2 = img2[slices[0], slices[1], slices[2]]
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
    if mx - mn < 1e-8: return img
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


def confidence_interval_95(values):
    """Compute mean and 95% CI using t-distribution."""
    n = len(values)
    if n < 2:
        return np.mean(values), 0.0, 0.0
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    from scipy import stats
    t_crit = stats.t.ppf(0.975, n - 1)
    margin = t_crit * std / np.sqrt(n)
    return mean, mean - margin, mean + margin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--aekl_ckpt', required=True)
    parser.add_argument('--diff_ckpt', required=True)
    parser.add_argument('--cnet_ckpt', required=True)
    parser.add_argument('--ae_decoder_ckpt', default=None)
    parser.add_argument('--ref_ckpt', default=None)
    parser.add_argument('--base_ch', default=32, type=int)
    parser.add_argument('--cache_dir', default='/tmp/eval_cache_v2', type=str)
    parser.add_argument('--n_test', default=50, type=int, help='Number of test subjects (default ALL 50)')
    parser.add_argument('--eval_split', default='test', type=str, help='Which split to evaluate: test or valid')
    parser.add_argument('--m_las', default=3, type=int)
    parser.add_argument('--output_json', default=None)
    parser.add_argument('--progress_file', default=None, help='File to write progress for monitoring')
    parser.add_argument('--label', default=None)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    torch.cuda.set_device(device)
    os.makedirs(args.cache_dir, exist_ok=True)

    label = args.label or 'eval_v2'
    print(f'\n=== Full-Scale Evaluation: {label} ===')

    # Load models
    autoencoder = networks.init_autoencoder(args.aekl_ckpt)
    if args.ae_decoder_ckpt:
        autoencoder.load_state_dict(torch.load(args.ae_decoder_ckpt, map_location='cpu'))
    diffusion = networks.init_latent_diffusion(args.diff_ckpt)
    controlnet = networks.init_controlnet()
    controlnet.load_state_dict(torch.load(args.cnet_ckpt, map_location='cpu'))
    for m in [diffusion, autoencoder, controlnet]:
        for p in m.parameters(): p.requires_grad = False
        m.to(device).eval()

    refnet = None
    if args.ref_ckpt and os.path.exists(args.ref_ckpt):
        refnet = RefinementUNet3D(in_ch=2, out_ch=1, base_ch=args.base_ch).to(device)
        refnet.load_state_dict(torch.load(args.ref_ckpt, map_location='cpu'))
        refnet.eval()
        for p in refnet.parameters(): p.requires_grad = False
        print(f'  Refinement net loaded: {args.ref_ckpt}')

    # Dataset
    npz_reader = NumpyReader(npz_keys=['data'])
    transforms_fn = transforms.Compose([
        transforms.LoadImageD(keys=['starting_latent', 'followup_latent'], reader=npz_reader),
        transforms.EnsureChannelFirstD(keys=['starting_latent', 'followup_latent'], channel_dim=0),
        transforms.DivisiblePadD(keys=['starting_latent', 'followup_latent'], k=4, mode='constant'),
        transforms.Lambda(func=concat_covariates),
    ])

    df = pd.read_csv(args.csv)
    eval_split = getattr(args, 'eval_split', 'test')
    test_df = df[df.split == eval_split].head(args.n_test) if 'split' in df.columns else df.tail(args.n_test)
    n_actual = len(test_df)
    testset = get_dataset_from_pd(test_df, transforms_fn, args.cache_dir)
    print(f'  Eval split: {eval_split}, subjects: {n_actual}')

    # Scale factor
    train_df = df[df.split == 'train'].head(1) if 'split' in df.columns else df.head(1)
    train_tmp = get_dataset_from_pd(train_df, transforms_fn, args.cache_dir)
    with torch.no_grad():
        with autocast(enabled=True):
            z = train_tmp[0]['followup_latent']
    scale_factor = 1 / torch.std(z)
    del train_tmp
    print(f'  Scale factor: {scale_factor:.4f}')

    gt_preprocess = transforms.Compose([
        transforms.LoadImage(image_only=True),
        transforms.EnsureChannelFirst(),
        transforms.Spacing(pixdim=const.RESOLUTION),
        transforms.ResizeWithPadOrCrop(spatial_size=const.INPUT_SHAPE_AE, mode='minimum'),
        transforms.ScaleIntensity(minv=0, maxv=1),
    ])

    # Evaluation loop
    all_results = []
    test_df_reset = test_df.reset_index(drop=True)
    total_start = time.time()

    for idx in range(min(args.n_test, len(testset))):
        t0 = time.time()
        sample = testset[idx]

        fu_img = gt_preprocess(sample['followup_image'])
        fu_np = fu_img.squeeze().cpu().numpy()
        seg_data = nib.load(sample['followup_segm']).get_fdata()

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

        refined_np = pred_np
        if refnet is not None:
            bl_path = test_df_reset.iloc[idx]['starting_image']
            bl_np = normalize_01(load_and_crop(bl_path))
            bl_t = torch.tensor(bl_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            pred_t = torch.tensor(pred_np, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    refined_t = refnet(torch.cat([pred_t, bl_t], dim=1))
            refined_np = np.clip(refined_t.squeeze().cpu().float().numpy(), 0, 1)

        overall = compute_ssim(fu_np, refined_np, data_range=1.0)
        region_results = {}
        for rname, rlabels in REGION_LABELS.items():
            ssim_val, nvoxels = compute_region_ssim(fu_np, refined_np, seg_data, rlabels)
            region_results[rname] = {'ssim': ssim_val, 'nvoxels': nvoxels}

        ad_ssims = [region_results[r]['ssim'] for r in AD_PRIMARY
                    if not np.isnan(region_results[r]['ssim'])]
        ad_composite = np.mean(ad_ssims) if ad_ssims else float('nan')

        elapsed = time.time() - t0
        result = {
            'pair': idx, 'overall_ssim': overall, 'ad_composite': ad_composite,
            'regions': region_results, 'time': elapsed,
            'subject_id': str(test_df_reset.iloc[idx].get('subject_id', f'subj_{idx}')),
        }
        all_results.append(result)

        # Progress output
        remaining = (time.time() - total_start) / (idx + 1) * (n_actual - idx - 1)
        mins = int(remaining // 60)
        print(f'  [{idx+1}/{n_actual}] Overall={overall:.4f}, AD-Comp={ad_composite:.4f}, '
              f'Hippo={region_results["hippocampus"]["ssim"]:.4f}, '
              f'time={elapsed:.0f}s, ETA={mins}min')

        # Write progress file for monitoring
        if args.progress_file:
            progress = {
                'completed': idx + 1, 'total': n_actual,
                'current_overall': overall, 'current_ad_comp': ad_composite,
                'eta_minutes': mins,
            }
            with open(args.progress_file, 'w') as f:
                json.dump(progress, f)

    # Summary with 95% CI
    print(f'\n=== {label} Full Summary ({len(all_results)} subjects) ===')
    summary = {}
    for key in ['overall', 'ad_composite'] + list(REGION_LABELS.keys()):
        if key == 'overall':
            vals = [r['overall_ssim'] for r in all_results]
        elif key == 'ad_composite':
            vals = [r['ad_composite'] for r in all_results if not np.isnan(r['ad_composite'])]
        else:
            vals = [r['regions'][key]['ssim'] for r in all_results
                    if key in r['regions'] and not np.isnan(r['regions'][key]['ssim'])]
        
        if vals:
            mean, ci_low, ci_high = confidence_interval_95(vals)
            std = np.std(vals, ddof=1) if len(vals) > 1 else 0
            summary[key] = {
                'mean': float(mean), 'std': float(std),
                'ci95_low': float(ci_low), 'ci95_high': float(ci_high),
                'n': len(vals),
            }
            print(f'  {key:25s}: {mean:.4f} ± {std:.4f}  95%CI=[{ci_low:.4f}, {ci_high:.4f}]  n={len(vals)}')

    report = {
        'label': label,
        'cnet_ckpt': args.cnet_ckpt,
        'ae_decoder_ckpt': args.ae_decoder_ckpt,
        'ref_ckpt': args.ref_ckpt,
        'n_test': len(all_results),
        'm_las': args.m_las,
        'base_ch': args.base_ch,
        'summary': summary,
        'per_subject': all_results,
    }

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or '.', exist_ok=True)
        with open(args.output_json, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'\nSaved: {args.output_json}')


if __name__ == '__main__':
    main()
