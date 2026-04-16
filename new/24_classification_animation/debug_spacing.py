"""Debug: compare SSIM computation methods using saved pipeline outputs."""
import torch
import numpy as np
from monai import transforms
import nibabel as nib
from skimage.metrics import structural_similarity
import csv, json, os

BMCI_CSV = "/home/wangchong/data/fwz/output/innovation_5/prepared/B_mci.csv"
RESULT_DIR = "/home/wangchong/data/fwz/output/classification_animation"

# Step 1: Find a followup image path for subject 005_S_0572
print("=" * 60)
print("Finding GT image path from B_mci.csv")
print("=" * 60)
with open(BMCI_CSV) as f:
    reader = csv.DictReader(f)
    rows = [r for r in reader if r['subject_id'] == '005_S_0572']
print(f"Found {len(rows)} pairs for 005_S_0572")

# Get the second visit's image (first followup)
gt_path = rows[0]['followup_image']
print(f"GT (followup) image path: {gt_path}")
print(f"GT exists: {os.path.exists(gt_path)}")

if not os.path.exists(gt_path):
    gt_path = rows[0]['starting_image']
    print(f"Trying starting image: {gt_path}")
    print(f"Exists: {os.path.exists(gt_path)}")

gt_img = nib.load(gt_path)
gt_np = gt_img.get_fdata()
print(f"GT shape: {gt_np.shape}, range: [{gt_np.min():.4f}, {gt_np.max():.4f}]")
print(f"GT pixdim: {gt_img.header.get_zooms()}")

# Step 2: Load saved prediction
print("\n" + "=" * 60)
print("Loading saved prediction NIfTI")
print("=" * 60)
pred_path = os.path.join(RESULT_DIR, "005_S_0572_visit2_pred.nii.gz")
print(f"Pred path: {pred_path}")
print(f"Pred exists: {os.path.exists(pred_path)}")

for f in sorted(os.listdir(RESULT_DIR)):
    if f.endswith('.nii.gz'):
        print(f"  Found: {f}")

if os.path.exists(pred_path):
    pred_img = nib.load(pred_path)
    pred_np = pred_img.get_fdata()
    print(f"\nPred shape: {pred_np.shape}, range: [{pred_np.min():.4f}, {pred_np.max():.4f}]")
    
    print("\n" + "=" * 60)
    print("SSIM comparison with different alignment methods")
    print("=" * 60)
    
    resample_fn = transforms.Spacing(pixdim=1.5)
    
    # Method A: evaluate_all_methods.py style (Spacing + corner crop)
    gt_tensor = torch.from_numpy(gt_np).unsqueeze(0).float()
    gt_resampled = resample_fn(gt_tensor).squeeze(0).numpy()
    print(f"\nMethod A (evaluate_all_methods.py): Spacing(1.5) + corner crop")
    print(f"  GT resampled shape: {gt_resampled.shape}")
    ms_a = tuple(min(a,b) for a,b in zip(pred_np.shape, gt_resampled.shape))
    p_a = pred_np[:ms_a[0], :ms_a[1], :ms_a[2]]
    g_a = gt_resampled[:ms_a[0], :ms_a[1], :ms_a[2]]
    p_a_n = (p_a - p_a.min()) / (p_a.max() - p_a.min() + 1e-8)
    g_a_n = (g_a - g_a.min()) / (g_a.max() - g_a.min() + 1e-8)
    ssim_a = structural_similarity(g_a_n, p_a_n, data_range=g_a_n.max()-g_a_n.min())
    print(f"  Crop shape: {ms_a}, SSIM = {ssim_a:.4f}")
    
    # Method B: ResizeWithPadOrCrop (center crop/pad)
    ts_b = tuple(min(a,b) for a,b in zip(pred_np.shape, gt_np.shape))
    resizer = transforms.ResizeWithPadOrCrop(spatial_size=ts_b, mode='minimum')
    p_b = resizer(torch.from_numpy(pred_np).unsqueeze(0).float()).squeeze(0).numpy()
    g_b = resizer(torch.from_numpy(gt_np).unsqueeze(0).float()).squeeze(0).numpy()
    p_b_n = (p_b - p_b.min()) / (p_b.max() - p_b.min() + 1e-8)
    g_b_n = (g_b - g_b.min()) / (g_b.max() - g_b.min() + 1e-8)
    ssim_b = structural_similarity(g_b_n, p_b_n, data_range=g_b_n.max()-g_b_n.min())
    print(f"\nMethod B (run_pipeline v2): ResizeWithPadOrCrop center")
    print(f"  Crop shape: {ts_b}, SSIM = {ssim_b:.4f}")
    
    # Method C: Direct corner crop (no Spacing)
    ms_c = tuple(min(a,b) for a,b in zip(pred_np.shape, gt_np.shape))
    p_c = pred_np[:ms_c[0], :ms_c[1], :ms_c[2]]
    g_c = gt_np[:ms_c[0], :ms_c[1], :ms_c[2]]
    p_c_n = (p_c - p_c.min()) / (p_c.max() - p_c.min() + 1e-8)
    g_c_n = (g_c - g_c.min()) / (g_c.max() - g_c.min() + 1e-8)
    ssim_c = structural_similarity(g_c_n, p_c_n, data_range=g_c_n.max()-g_c_n.min())
    print(f"\nMethod C: Direct corner crop (no resampling)")
    print(f"  Crop shape: {ms_c}, SSIM = {ssim_c:.4f}")
    
    # Method D: Center crop pred to GT shape (120,144,120)
    d = tuple((p - g) for p, g in zip(pred_np.shape, gt_np.shape))
    print(f"\nMethod D: Center-crop pred only (diff={d})")
    if all(dd >= 0 for dd in d):
        slices = tuple(slice(dd//2, dd//2 + gs) for dd, gs in zip(d, gt_np.shape))
        p_d = pred_np[slices]
        g_d = gt_np.copy()
        p_d_n = (p_d - p_d.min()) / (p_d.max() - p_d.min() + 1e-8)
        g_d_n = (g_d - g_d.min()) / (g_d.max() - g_d.min() + 1e-8)
        ssim_d = structural_similarity(g_d_n, p_d_n, data_range=g_d_n.max()-g_d_n.min())
        print(f"  Pred cropped: {p_d.shape}, SSIM = {ssim_d:.4f}")
    
    # Method E: MetaTensor with correct pixdim=1.5
    from monai.data import MetaTensor
    affine_15 = np.diag([1.5, 1.5, 1.5, 1.0]).astype(np.float32)
    pred_mt = MetaTensor(torch.from_numpy(pred_np).unsqueeze(0).float(), 
                         affine=torch.tensor(affine_15))
    gt_mt = MetaTensor(torch.from_numpy(gt_np).unsqueeze(0).float(),
                       affine=torch.tensor(affine_15))
    pred_res = resample_fn(pred_mt).squeeze(0).numpy()
    gt_res = resample_fn(gt_mt).squeeze(0).numpy()
    ms_e = tuple(min(a,b) for a,b in zip(pred_res.shape, gt_res.shape))
    p_e = pred_res[:ms_e[0], :ms_e[1], :ms_e[2]]
    g_e = gt_res[:ms_e[0], :ms_e[1], :ms_e[2]]
    p_e_n = (p_e - p_e.min()) / (p_e.max() - p_e.min() + 1e-8)
    g_e_n = (g_e - g_e.min()) / (g_e.max() - g_e.min() + 1e-8)
    ssim_e = structural_similarity(g_e_n, p_e_n, data_range=g_e_n.max()-g_e_n.min())
    print(f"\nMethod E: Both with MetaTensor(pixdim=1.5) + Spacing(1.5)")
    print(f"  Pred resampled: {pred_res.shape}, GT resampled: {gt_res.shape}")
    print(f"  Crop shape: {ms_e}, SSIM = {ssim_e:.4f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Method A (eval script style - Spacing bug):    SSIM = {ssim_a:.4f}")
    print(f"Method B (center ResizeWithPadOrCrop):          SSIM = {ssim_b:.4f}")
    print(f"Method C (corner crop, no resample):            SSIM = {ssim_c:.4f}")
    if all(dd >= 0 for dd in d):
        print(f"Method D (center-crop pred to GT):             SSIM = {ssim_d:.4f}")
    print(f"Method E (correct MetaTensor pixdim=1.5):       SSIM = {ssim_e:.4f}")

print("\nDone!")
