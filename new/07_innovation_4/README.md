# Innovation 4: 3D Perceptual Loss + Frequency Domain Constraint

## Overview

Replace BrLP's fake-3D perceptual loss (2D VGG squeeze + 20% slice sampling) with:

1. **True 3D MedicalNet perceptual loss** — ResNet-10 pretrained on 23 medical imaging datasets
2. **Laplacian pyramid frequency constraint** — Multi-scale high-frequency preservation

## Motivation

BrLP original AE uses `PerceptualLoss(is_fake_3d=True)`, which processes only ~20% of 2D slices through a VGG network. This misses volumetric spatial relationships. MedicalNet operates natively in 3D, understanding true volumetric features.

The frequency constraint combats AE over-smoothing by explicitly penalizing loss of high-frequency detail (brain sulci, gyri edges, hippocampal boundaries).

## Structure

```
07_innovation_4/
├── src/
│   ├── __init__.py
│   ├── medicalnet_perceptual.py    # 3D ResNet-10 + perceptual loss
│   └── frequency_losses.py         # Laplacian pyramid + FFT losses
├── scripts/
│   ├── train_autoencoder_3d_perceptual.py  # Modified AE training
│   └── evaluate_innovation4.py     # Evaluation with region metrics
├── configs/
│   └── train_mci.yaml              # Training configuration
├── ../dashboard/
│   └── server_monitor.py           # Unified monitoring dashboard
├── run.sh                          # Server run script
├── deploy.ps1                      # Windows → server deploy
├── changelog.json                  # Auto-updated training log
└── README.md
```

## Key Changes vs Baseline

| Component         | Baseline                         | Innovation 4                  |
| ----------------- | -------------------------------- | ----------------------------- |
| Perceptual Loss   | VGG squeeze, fake-3D, 20% slices | MedicalNet ResNet-10, true 3D |
| Frequency Loss    | None                             | Laplacian pyramid (3 levels)  |
| Perceptual Weight | 0.001                            | 0.001                         |
| Frequency Weight  | N/A                              | 0.01                          |

## Usage

### Deploy to server

```powershell
.\deploy.ps1
```

### Train AE

```bash
bash run.sh train
```

### Evaluate

```bash
bash run.sh eval
```

### Full pipeline

```bash
bash run.sh all
```

### Unified dashboard

Use the single monitoring page shared across experiments:

```bash
python ../dashboard/server_monitor.py --port 8080
```

## References

- MedicalNet (Chen et al., 2019): 3D ResNet pretrained on 23 medical datasets
- 3D MedDiffusion (Gao et al., IEEE TMI 2025): True 3D perceptual loss design
- AG-LDM: Laplacian pyramid frequency constraints for medical image synthesis
