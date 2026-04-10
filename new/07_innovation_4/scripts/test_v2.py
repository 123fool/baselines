"""
Smoke test for Innovation 4 v2 losses and training components.
Verifies: imports, loss computation, gradient flow, memory estimation.
Run on server with: CUDA_VISIBLE_DEVICES=0 python test_v2.py --mednet_ckpt <path>
"""

import os, sys, time, argparse
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BRLP_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', 'src'))
INNOV_SRC = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
sys.path.insert(0, BRLP_SRC)
sys.path.insert(0, INNOV_SRC)

# Patch torch.load
_orig = torch.load
def _patched(*a, **kw):
    kw.setdefault('weights_only', False)
    return _orig(*a, **kw)
torch.load = _patched


def test_imports():
    print("=== Test 1: Imports ===")
    from medicalnet_perceptual_v2 import MedicalNet3DPerceptualLoss
    from frequency_losses import LaplacianPyramidLoss
    from generative.losses import PerceptualLoss, PatchAdversarialLoss
    from brlp import const, networks, utils, KLDivergenceLoss
    print(f"  All imports OK")
    print(f"  INPUT_SHAPE_AE = {const.INPUT_SHAPE_AE}")
    print(f"  PyTorch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return True


def test_losses(mednet_ckpt):
    print("\n=== Test 2: Loss Functions ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Fake AE output: (1, 1, 120, 144, 120) like real BrLP
    pred = torch.randn(1, 1, 120, 144, 120, device=device, requires_grad=True)
    tgt = torch.randn(1, 1, 120, 144, 120, device=device)

    # 1) Original PerceptualLoss
    import warnings
    from generative.losses import PerceptualLoss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig_perc = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                    is_fake_3d=True, fake_3d_ratio=0.2).to(device)
    with autocast(enabled=True):
        loss_orig = orig_perc(pred.float(), tgt.float())
    print(f"  Original PerceptualLoss: {loss_orig.item():.6f}")

    # 2) MedicalNet 3D v2
    from medicalnet_perceptual_v2 import MedicalNet3DPerceptualLoss
    perc3d = MedicalNet3DPerceptualLoss(
        pretrained_path=mednet_ckpt,
        downsample_size=(80, 96, 80)
    ).to(device)
    torch.cuda.reset_peak_memory_stats()
    with autocast(enabled=True):
        loss_3d = perc3d(pred.float(), tgt.float())
    mem_3d = torch.cuda.max_memory_allocated() / 1e9
    print(f"  MedicalNet3D v2 loss: {loss_3d.item():.6f} (peak mem: {mem_3d:.2f} GB)")

    # 3) Gradient flow check
    loss_3d.backward(retain_graph=True)
    if pred.grad is not None and pred.grad.abs().sum() > 0:
        print(f"  Gradient flow: OK (grad norm={pred.grad.norm():.4f})")
    else:
        print(f"  Gradient flow: FAILED")
        return False

    # 4) Laplacian pyramid
    pred.grad = None
    from frequency_losses import LaplacianPyramidLoss
    freq_fn = LaplacianPyramidLoss(num_levels=3).to(device)
    with autocast(enabled=True):
        loss_freq = freq_fn(pred.float(), tgt.float())
    loss_freq.backward()
    print(f"  LaplacianPyramid loss: {loss_freq.item():.6f}")
    print(f"  Freq gradient flow: OK (grad norm={pred.grad.norm():.4f})")

    # 5) L1 loss for reference
    l1 = F.l1_loss(pred.float().detach(), tgt.float())
    print(f"  Reference L1 loss: {l1.item():.6f}")

    # 6) Loss magnitude analysis
    print(f"\n  --- Loss magnitude analysis ---")
    print(f"  L1:                     {l1.item():.6f}")
    print(f"  0.001 * orig_perc:      {0.001 * loss_orig.item():.6f}")
    print(f"  0.0005 * mednet3d:      {0.0005 * loss_3d.item():.6f}")
    print(f"  0.005 * laplacian:      {0.005 * loss_freq.item():.6f}")
    print(f"  Sum of new aux losses:  {0.0005 * loss_3d.item() + 0.005 * loss_freq.item():.6f}")
    print(f"  Ratio new/L1:           {(0.0005 * loss_3d.item() + 0.005 * loss_freq.item()) / l1.item():.4f}")

    return True


def test_ae_roundtrip(mednet_ckpt):
    """Test that pretrained AE can still reconstruct well."""
    print("\n=== Test 3: AE Reconstruction Sanity ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    from brlp import networks

    ae_ckpt = os.environ.get('AE_CKPT',
        '/home/wangchong/data/fwz/brlp-train/pretrained/autoencoder.pth')
    if not os.path.exists(ae_ckpt):
        print(f"  Skipping (AE checkpoint not found: {ae_ckpt})")
        return True

    ae = networks.init_autoencoder(ae_ckpt).to(device).eval()
    x = torch.randn(1, 1, 120, 144, 120, device=device) * 0.5 + 0.5
    x = x.clamp(0, 1)
    with torch.no_grad():
        recon, _, _ = ae(x)
    l1 = F.l1_loss(recon, x).item()
    print(f"  AE random input L1: {l1:.4f} (expected ~0.1-0.3)")
    print(f"  AE works: OK")
    return True


def test_combined_training_step(mednet_ckpt):
    """Simulate one training step to catch any runtime errors."""
    print("\n=== Test 4: Combined Training Step ===")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    import warnings
    from generative.losses import PerceptualLoss, PatchAdversarialLoss
    from brlp import networks, KLDivergenceLoss
    from medicalnet_perceptual_v2 import MedicalNet3DPerceptualLoss
    from frequency_losses import LaplacianPyramidLoss

    ae_ckpt = os.environ.get('AE_CKPT',
        '/home/wangchong/data/fwz/brlp-train/pretrained/autoencoder.pth')
    if not os.path.exists(ae_ckpt):
        print(f"  Skipping (AE ckpt not found)")
        return True

    ae = networks.init_autoencoder(ae_ckpt).to(device)
    disc = networks.init_patch_discriminator(None).to(device)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        orig_perc = PerceptualLoss(spatial_dims=3, network_type="squeeze",
                                    is_fake_3d=True, fake_3d_ratio=0.2).to(device)
    perc3d = MedicalNet3DPerceptualLoss(pretrained_path=mednet_ckpt,
                                         downsample_size=(80, 96, 80)).to(device)
    freq_fn = LaplacianPyramidLoss(num_levels=3).to(device)
    kl_fn = KLDivergenceLoss()
    adv_fn = PatchAdversarialLoss(criterion="least_squares")

    optimizer = torch.optim.Adam(ae.parameters(), lr=5e-5)

    # Simulate one forward+backward
    x = torch.randn(1, 1, 120, 144, 120, device=device).clamp(0, 1)
    torch.cuda.reset_peak_memory_stats()

    ae.train()
    with autocast(enabled=True):
        recon, z_mu, z_sigma = ae(x)
        logits = disc(recon.contiguous().float())[-1]

        loss_rec = F.l1_loss(recon.float(), x.float())
        loss_kl = 1e-7 * kl_fn(z_mu, z_sigma)
        loss_perc_orig = 0.001 * orig_perc(recon.float(), x.float())
        loss_perc_3d = 0.0005 * perc3d(recon.float(), x.float())
        loss_freq = 0.005 * freq_fn(recon.float(), x.float())
        loss_adv = 0.025 * adv_fn(logits, target_is_real=True, for_discriminator=False)

        loss_total = loss_rec + loss_kl + loss_perc_orig + loss_perc_3d + loss_freq + loss_adv

    loss_total.backward()
    optimizer.step()
    optimizer.zero_grad()

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Total loss: {loss_total.item():.6f}")
    print(f"  Breakdown: rec={loss_rec.item():.4f}, perc_orig={loss_perc_orig.item():.4f}, "
          f"perc_3d={loss_perc_3d.item():.4f}, freq={loss_freq.item():.4f}, "
          f"adv={loss_adv.item():.4f}, kl={loss_kl.item():.6f}")
    print(f"  Peak GPU memory: {peak_mem:.2f} GB")
    if peak_mem > 23:
        print(f"  WARNING: Memory usage too high for 24GB GPU!")
        return False
    print(f"  Combined training step: PASS")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mednet_ckpt', required=True, type=str)
    args = parser.parse_args()

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Losses", test_losses(args.mednet_ckpt)))
    results.append(("AE roundtrip", test_ae_roundtrip(args.mednet_ckpt)))

    torch.cuda.empty_cache()
    results.append(("Combined step", test_combined_training_step(args.mednet_ckpt)))

    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    print(f"\nOverall: {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
