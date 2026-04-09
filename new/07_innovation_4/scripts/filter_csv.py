"""Pre-filter CSV to remove rows where image files don't exist."""
import os
import sys
import pandas as pd

csv_path = sys.argv[1]
output_path = sys.argv[2] if len(sys.argv) > 2 else csv_path.replace('.csv', '_filtered.csv')

df = pd.read_csv(csv_path)
print(f"Original rows: {len(df)}")

# Check which image_path files exist
mask = df['image_path'].apply(lambda p: os.path.exists(str(p)))
missing = df[~mask]
if len(missing) > 0:
    print(f"Missing {len(missing)} files:")
    for _, row in missing.iterrows():
        print(f"  {row['image_path']}")

df_filtered = df[mask]
print(f"Filtered rows: {len(df_filtered)}")
df_filtered.to_csv(output_path, index=False)
print(f"Saved to {output_path}")
