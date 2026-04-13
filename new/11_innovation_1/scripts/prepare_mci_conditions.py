"""
Data Preparation for Innovation 1: MCI Dynamic Conditioning.

Reads B_mci.csv and enriches it with:
  - hippocampal_atrophy_rate:   (start_hipp - followup_hipp) / time_delta
  - ventricular_expansion_rate: (followup_vent - start_vent) / time_delta

time_delta is derived from (followup_age - starting_age), already in normalized 0-1 scale.

Usage:
    python prepare_mci_conditions.py \
        --input_csv  /path/to/B_mci.csv \
        --output_csv /path/to/B_mci_inn1.csv
"""

import argparse
import pandas as pd
import numpy as np


def compute_rates(df):
    """
    Add hippocampal_atrophy_rate and ventricular_expansion_rate columns.

    Uses days_from_first_visit to compute time delta in years for more stable rates.
    Falls back to age difference if days columns not available.
    """
    # Prefer actual days between visits (more stable than normalized age delta)
    if 'followup_days_from_first_visit' in df.columns and 'starting_days_from_first_visit' in df.columns:
        time_delta_years = (
            (df['followup_days_from_first_visit'] - df['starting_days_from_first_visit']) / 365.25
        ).clip(lower=0.25)  # minimum 3 months
        print(f"  Using days_from_first_visit for time delta (mean={time_delta_years.mean():.2f} years)")
    else:
        # Fallback: age in 0-1 scale, assume range ~50 years
        time_delta_years = ((df['followup_age'] - df['starting_age']) * 100).clip(lower=0.25)
        print(f"  Using normalized age for time delta")

    # Hippocampal atrophy: positive means volume loss (decline) per year
    raw_atrophy = (df['starting_hippocampus'] - df['followup_hippocampus']) / time_delta_years
    # Ventricular expansion: positive means volume growth (dilation) per year
    raw_vent = (df['followup_lateral_ventricle'] - df['starting_lateral_ventricle']) / time_delta_years

    # Normalize to zero-mean unit-variance, then clip to [-3, 3] for stability
    for name, raw in [('hippocampal_atrophy_rate', raw_atrophy),
                       ('ventricular_expansion_rate', raw_vent)]:
        mu = raw.mean()
        sigma = raw.std().clip(min=1e-6)
        normalized = (raw - mu) / sigma
        df[name] = normalized.clip(-3.0, 3.0)
        print(f"  {name}: raw_mean={mu:.4f}, raw_std={sigma:.4f}")

    return df


def print_stats(df):
    """Print statistics of the new columns."""
    for col in ['hippocampal_atrophy_rate', 'ventricular_expansion_rate']:
        vals = df[col]
        print(f"\n{col}:")
        print(f"  mean:   {vals.mean():.4f}")
        print(f"  std:    {vals.std():.4f}")
        print(f"  min:    {vals.min():.4f}")
        print(f"  25%:    {vals.quantile(0.25):.4f}")
        print(f"  50%:    {vals.quantile(0.50):.4f}")
        print(f"  75%:    {vals.quantile(0.75):.4f}")
        print(f"  max:    {vals.max():.4f}")
        print(f"  zeros:  {(vals == 0).sum()}")
        print(f"  >0:     {(vals > 0).sum()} ({(vals > 0).mean()*100:.1f}%)")
        print(f"  <0:     {(vals < 0).sum()} ({(vals < 0).mean()*100:.1f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare MCI conditioning data for Innovation 1')
    parser.add_argument('--input_csv', required=True, help='Path to B_mci.csv')
    parser.add_argument('--output_csv', required=True, help='Output path for enriched CSV')
    args = parser.parse_args()

    print(f"Reading {args.input_csv} ...")
    df = pd.read_csv(args.input_csv)
    print(f"  {len(df)} rows, {df['subject_id'].nunique()} subjects")

    # Verify required columns
    required = ['starting_hippocampus', 'followup_hippocampus',
                'starting_lateral_ventricle', 'followup_lateral_ventricle',
                'starting_age', 'followup_age']
    for col in required:
        assert col in df.columns, f"Missing column: {col}"

    df = compute_rates(df)
    print_stats(df)

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved to {args.output_csv}")
