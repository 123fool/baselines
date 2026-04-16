"""
Multi-timepoint BoN Weighted Prediction.

Given a starting brain MRI, predict future brain MRIs at multiple timepoints
(e.g., 3 years, 6-month intervals) using BoN Weighted sampling.

This is a modified version of brlp.cli.infer() that replaces the standard
LAS sampling with BoN Weighted sampling for higher-quality predictions.

Usage (on server):
    python infer_bon_multitime.py \
        --input /path/to/input.csv \
        --output /path/to/output/ \
        --confs /path/to/confs.yaml \
        --target_age 76 \
        --target_diagnosis 2 \
        --steps 7 \
        --n_candidates 8
"""

import os, sys, argparse, yaml, time
import numpy as np
import torch
import pandas as pd
import nibabel as nib
from monai import transforms
from datetime import datetime

# Path setup
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, SCRIPT_DIR)

from brlp import networks, const, utils
from brlp.sampling import sample_using_controlnet_and_z
from sampling_bon_integrated import sample_bon_weighted

try:
    from leaspy import Leaspy, AlgorithmSettings, Data
    HAS_LEASPY = True
except ImportError:
    HAS_LEASPY = False
    print("Warning: Leaspy not available. Will use linear interpolation for volume trajectories.")


def _measure_synthseg(segm_path, confs):
    """Measure SynthSeg segmentation — returns normalized brain region volumes."""
    segm = nib.load(segm_path).get_fdata().round()
    full_regions = set()
    regions_names = [r for r in const.SYNTHSEG_CODEMAP.values() if 'background' not in r]
    for region in regions_names:
        full_regions.add(region.replace('left_', '').replace('right_', ''))

    record = {}
    for region in full_regions:
        if region in const.CONDITIONING_VARIABLES:
            record[region] = 0

    for code, region in const.SYNTHSEG_CODEMAP.items():
        fregion = region.replace('left_', '').replace('right_', '')
        if fregion in const.CONDITIONING_VARIABLES:
            record[fregion] += (segm == code).sum()

    for region, raw_v in record.items():
        _min, _max = confs['minmax_params'][region]
        record[region] = (raw_v - _min) / (_max - _min)
    return record


def _map_to_data(df):
    """Convert DataFrame to Leaspy Data object."""
    df = df.copy()
    if df.iloc[0].TIME >= 0 and df.iloc[0].TIME <= 1:
        df['TIME'] *= 100
    df = df.drop_duplicates(['ID', 'TIME'], keep='first')
    for region in const.CONDITIONING_REGIONS:
        if region == 'lateral_ventricle':
            continue
        df[region] = 1 - df[region]
    df = df.set_index(["ID", "TIME"], verify_integrity=False).sort_index()
    df = df[const.CONDITIONING_REGIONS]
    return Data.from_dataframe(df)


def _reverse_and_correct(estimates, confs):
    """Reverse Leaspy normalization and apply median correction."""
    for i in range(len(estimates)):
        for j, region_name in enumerate(const.CONDITIONING_REGIONS):
            if region_name != 'lateral_ventricle':
                estimates[i][j] = 1 - estimates[i][j]
            x = estimates[i][j]
            m, q = confs['median_corrections'][region_name]
            x = (x - q) / m
            x = max(x, -0.2)
            x = min(x, 1.2)
            estimates[i][j] = x
    return estimates


def _interpolate_volumes(starting_volumes, timepoints, starting_age):
    """
    Simple linear interpolation for volume trajectories when Leaspy is not available.
    Assumes gradual atrophy trend based on typical MCI progression rates.
    """
    # Typical annual atrophy rates for MCI (from literature)
    atrophy_rates = {
        'cerebral_cortex': -0.008,          # ~0.8% per year
        'hippocampus': -0.035,              # ~3.5% per year (highest in MCI)
        'amygdala': -0.025,                 # ~2.5% per year
        'cerebral_white_matter': -0.006,    # ~0.6% per year
        'lateral_ventricle': 0.04,          # +4% per year (expansion)
    }
    estimates = []
    for age in timepoints:
        years_from_start = (age - starting_age)
        row = []
        for region in const.CONDITIONING_REGIONS:
            base = starting_volumes.get(region, 0.5)
            rate = atrophy_rates.get(region, -0.005)
            predicted = base + rate * years_from_start
            predicted = max(-0.2, min(1.2, predicted))
            row.append(predicted)
        estimates.append(row)
    return np.array(estimates)


def main():
    parser = argparse.ArgumentParser(description='Multi-timepoint BoN Weighted prediction')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--confs', type=str, required=True, help='Configuration YAML')
    parser.add_argument('--target_age', type=int, required=True, help='Final target age')
    parser.add_argument('--target_diagnosis', type=int, required=True, help='1=CN, 2=MCI, 3=AD')
    parser.add_argument('--steps', type=int, required=True, help='Number of timepoints')
    parser.add_argument('--n_candidates', type=int, default=8, help='BoN candidates (0=use LAS)')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    device = 'cpu' if args.cpu else 'cuda'
    confs = yaml.safe_load(open(args.confs, 'r'))
    os.makedirs(args.output, exist_ok=True)

    ts = lambda: datetime.now().strftime("[%H:%M:%S]")
    use_bon = args.n_candidates > 0
    method_name = f"BoN Weighted (N={args.n_candidates})" if use_bon else f"LAS (m={confs['las']['m']})"
    print(f"{ts()} Method: {method_name}")
    print(f"{ts()} Device: {device}")

    # ── Load models ──
    print(f"{ts()} Loading models...")
    autoencoder = networks.init_autoencoder(confs['autoencoder']).to(device).eval()
    diffusion = networks.init_latent_diffusion(confs['unet']).to(device).eval()
    controlnet = networks.init_controlnet(confs['controlnet']).to(device).eval()

    # ── Leaspy auxiliary model ──
    aux_type = {1: 'cn', 2: 'mci', 3: 'ad'}[args.target_diagnosis]
    if HAS_LEASPY:
        auxiliary = Leaspy.load(confs['aux'][aux_type])
        print(f"{ts()} Leaspy auxiliary model loaded.")
    else:
        auxiliary = None
        print(f"{ts()} Using linear interpolation (no Leaspy).")

    print(f"{ts()} Models loaded.")

    # ── Load input ──
    input_df = pd.read_csv(args.input).sort_values('age')
    segm_dir = os.path.join(args.output, 'segmentations')
    os.makedirs(segm_dir, exist_ok=True)

    # ── Extract regions (for Leaspy) ──
    if HAS_LEASPY and auxiliary is not None:
        print(f"{ts()} Extracting volumes from past MRIs...")
        records = []
        for input_row in input_df.itertuples():
            if 'segm_path' not in input_df.columns:
                scan_path = input_row.image_path
                segm_path = os.path.join(segm_dir, f'{input_row.image_uid}_segm.nii.gz')
                os.system(f'mri_synthseg --i {scan_path} --o {segm_path} --fast --threads {args.threads}')
            else:
                segm_path = input_row.segm_path
            measurements = _measure_synthseg(segm_path, confs)
            records.append({'ID': 'pt', 'TIME': input_row.age} | measurements)

        patient_records = pd.DataFrame(records)
        patient_records = _map_to_data(patient_records)

        # Estimate trajectories
        print(f"{ts()} Estimating volumetric trajectories...")
        auxiliary_settings = AlgorithmSettings('scipy_minimize')
        ip = auxiliary.personalize(patient_records, auxiliary_settings)
        last_record = input_df.iloc[-1]
        last_age = last_record.age
        timepoints = np.linspace(last_age, args.target_age, args.steps)
        estimates = auxiliary.estimate({'pt': timepoints}, ip)['pt']
        estimates = _reverse_and_correct(estimates, confs)
    else:
        # Fallback: linear interpolation from CSV volumes
        last_record = input_df.iloc[-1]
        last_age = last_record.age
        timepoints = np.linspace(last_age, args.target_age, args.steps)

        # Get starting volumes from CSV if available
        starting_volumes = {}
        for region in const.CONDITIONING_REGIONS:
            col = f'starting_{region}' if hasattr(last_record, f'starting_{region}') else region
            starting_volumes[region] = getattr(last_record, col, 0.5)
        estimates = _interpolate_volumes(starting_volumes, timepoints, last_age)

    # ── Load starting MRI ──
    load_volume = transforms.Compose([
        transforms.CopyItemsD(keys={'image_path'}, names=['image']),
        transforms.LoadImageD(image_only=True, keys=['image']),
        transforms.EnsureChannelFirstD(keys=['image']),
        transforms.SpacingD(pixdim=const.RESOLUTION, keys=['image']),
        transforms.ResizeWithPadOrCropD(spatial_size=const.INPUT_SHAPE_AE, mode='minimum', keys=['image']),
        transforms.ScaleIntensityD(minv=0, maxv=1, keys=['image']),
    ])

    input_image = load_volume({'image_path': last_record.image_path})['image']
    input_image = input_image.unsqueeze(0).to(device)
    input_latent = autoencoder.encode(input_image)[0]
    input_latent = transforms.DivisiblePad(k=4, mode='constant')(input_latent.squeeze(0))

    print(f"{ts()} Predicting {len(timepoints)} timepoints: ages {timepoints[0]:.1f} → {timepoints[-1]:.1f}")

    # ── Generate predictions at each timepoint ──
    results = []
    for i, target_age in enumerate(timepoints):
        t0 = time.time()

        # Build context vector (same order as CONDITIONING_VARIABLES)
        s = [(target_age - const.AGE_MIN) / const.AGE_DELTA,
             (last_record.sex - const.SEX_MIN) / const.SEX_DELTA,
             (args.target_diagnosis - const.DIA_MIN) / const.DIA_DELTA]
        v = list(estimates[i])
        covariates = torch.tensor(s + v)

        if use_bon:
            # BoN Weighted sampling
            mri = sample_bon_weighted(
                autoencoder=autoencoder,
                diffusion=diffusion,
                controlnet=controlnet,
                starting_z=input_latent.float(),
                starting_a=last_record.age / 100,
                context=covariates.float(),
                device=device,
                scale_factor=confs.get('scale_factor', 1),
                n_candidates=args.n_candidates,
                num_inference_steps=50,
                verbose=False,
            )
        else:
            # Standard LAS sampling
            mri = sample_using_controlnet_and_z(
                autoencoder=autoencoder,
                diffusion=diffusion,
                controlnet=controlnet,
                starting_z=input_latent.float(),
                starting_a=last_record.age / 100,
                context=covariates.float(),
                device=device,
                scale_factor=confs.get('scale_factor', 1),
                average_over_n=confs['las']['m'],
                num_inference_steps=25,
                verbose=False,
            )

        elapsed = time.time() - t0

        # Save NIfTI
        out_name = f'followup_age_{target_age:.0f}_{("bon" if use_bon else "las")}.nii.gz'
        nifti = nib.Nifti1Image(mri.numpy(), affine=const.MNI152_1P5MM_AFFINE)
        nifti = utils.percnorm_nifti(nifti, 5, 99.5)
        nifti.to_filename(os.path.join(args.output, out_name))

        results.append({
            'timepoint': i,
            'target_age': float(target_age),
            'method': 'bon_weighted' if use_bon else 'las',
            'time_sec': elapsed,
            'output_file': out_name,
        })
        print(f"{ts()} Timepoint {i+1}/{len(timepoints)}: age {target_age:.1f} — {elapsed:.1f}s — saved {out_name}")

    # ── Save summary ──
    import json
    summary = {
        'method': method_name,
        'n_timepoints': len(timepoints),
        'timepoints': timepoints.tolist(),
        'starting_age': float(last_age),
        'target_age': args.target_age,
        'results': results,
        'total_time_sec': sum(r['time_sec'] for r in results),
        'timestamp': datetime.now().isoformat(),
    }
    with open(os.path.join(args.output, 'multitime_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    total_time = sum(r['time_sec'] for r in results)
    print(f"\n{ts()} Done! {len(timepoints)} timepoints in {total_time:.0f}s")
    print(f"{ts()} Saved to: {args.output}")


if __name__ == '__main__':
    main()
