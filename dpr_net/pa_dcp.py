import torch
import torch.nn as nn

def compute_physics_aware_dark_channel(rgb_linear, beta_d=None, patch_size=41, use_physics_aware=True):
    if len(rgb_linear.size()) == 3:
        rgb_linear = rgb_linear.unsqueeze(0)
    if beta_d is not None and len(beta_d.size()) == 1:
        beta_d = beta_d.unsqueeze(0)
    B, C, H, W = rgb_linear.shape
    if use_physics_aware and beta_d is not None:
        epsilon = 1e-06
        weights_raw = 1.0 / (beta_d + epsilon)
        weights_max = weights_raw.max(dim=1, keepdim=True)[0]
        weights = weights_raw / (weights_max + epsilon)
        w_view = weights.view(B, 3, 1, 1)
        weighted_rgb = rgb_linear * w_view
    else:
        weighted_rgb = rgb_linear
    padding = patch_size // 2
    input_pool = -weighted_rgb.unsqueeze(2)
    max_pool = nn.MaxPool3d(kernel_size=(1, patch_size, patch_size), stride=1, padding=(0, padding, padding))
    min_weighted = -max_pool(input_pool)
    dark_channel, _ = torch.min(min_weighted, dim=1, keepdim=True)
    return dark_channel.squeeze(2)
