"""Quick sanity test for Innovation 4 modules."""
import sys, os
sys.path.insert(0, '/home/wangchong/data/fwz/brlp-code/src')
sys.path.insert(0, '/home/wangchong/data/fwz/code/innovation_4/src')

import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available(), 'GPUs:', torch.cuda.device_count())

# Test MedicalNet perceptual loss
from medicalnet_perceptual import MedicalNet3DPerceptualLoss
perc = MedicalNet3DPerceptualLoss(pretrained_path='/home/wangchong/data/fwz/code/innovation_4/pretrained/resnet_10_23dataset.pth')
perc = perc.cuda()
x = torch.randn(1, 1, 120, 144, 120).cuda()
y = torch.randn(1, 1, 120, 144, 120).cuda()
loss = perc(x, y)
print('MedicalNet perceptual loss:', loss.item())

# Test frequency losses
from frequency_losses import LaplacianPyramidLoss, FFTFrequencyLoss
lap = LaplacianPyramidLoss(num_levels=3).cuda()
fft_fn = FFTFrequencyLoss().cuda()
l1 = lap(x, y)
l2 = fft_fn(x, y)
print('Laplacian loss:', l1.item())
print('FFT loss:', l2.item())

# Test BrLP imports
from brlp import init_autoencoder, init_patch_discriminator, KLDivergenceLoss, GradientAccumulation
print('BrLP imports OK')

print('ALL TESTS PASSED')
