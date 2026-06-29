import math
from kornia.color import rgb_to_lab
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttenuateLoss(nn.Module):

    def __init__(self, override=None, target_intensity: float=0.5, w_intensity: float=1.0, w_spatial: float=1.0, w_saturation: float=0.1, w_tv: float=0.05, w_order: float=0.05, margin: float=0.02, edge_k: float=10.0, use_chromatic_order: bool=True):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.SmoothL1Loss()
        self.relu = nn.ReLU()
        self.target_intensity = target_intensity
        self.override = override
        self.w_intensity = w_intensity
        self.w_spatial = w_spatial
        self.w_saturation = w_saturation
        self.w_tv = w_tv
        self.w_order = w_order
        self.margin = margin
        self.edge_k = edge_k
        self.use_chromatic_order = use_chromatic_order

    def forward(self, direct: torch.Tensor, J: torch.Tensor) -> torch.Tensor:
        if self.override is not None:
            return self.override
        sat_low = F.softplus(-(J - self.margin))
        sat_high = F.softplus(J - 1 + self.margin)
        saturation_loss = (sat_low + sat_high).mean()
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        intensity_loss = self.huber(channel_intensities, torch.full_like(channel_intensities, self.target_intensity))
        init_spatial = torch.std(direct.detach(), dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.huber(channel_spatial, init_spatial)
        try:
            L = rgb_to_lab(torch.clamp(direct, 0, 1))[:, :1]
            dxL = L[:, :, :, 1:] - L[:, :, :, :-1]
            dyL = L[:, :, 1:, :] - L[:, :, :-1, :]
            wx = torch.exp(-self.edge_k * dxL.abs())
            wy = torch.exp(-self.edge_k * dyL.abs())
            dxJ = J[:, :, :, 1:] - J[:, :, :, :-1]
            dyJ = J[:, :, 1:, :] - J[:, :, :-1, :]
            tv_x = (dxJ.abs() * wx).mean()
            tv_y = (dyJ.abs() * wy).mean()
            tv_loss = 0.5 * (tv_x + tv_y)
        except Exception:
            dxJ = J[:, :, :, 1:] - J[:, :, :, :-1]
            dyJ = J[:, :, 1:, :] - J[:, :, :-1, :]
            tv_loss = 0.5 * (dxJ.abs().mean() + dyJ.abs().mean())
        chroma_order_loss = J.new_tensor(0.0)
        if self.use_chromatic_order and J.shape[1] == 3:
            Jr, Jg, Jb = (J[:, 0:1], J[:, 1:2], J[:, 2:3])
            chroma_order_loss = (self.relu(Jr - Jg) + self.relu(Jg - Jb)).mean()
        if torch.any(torch.isnan(saturation_loss)):
            print('NaN saturation loss!')
        if torch.any(torch.isnan(intensity_loss)):
            print('NaN intensity loss!')
        if torch.any(torch.isnan(spatial_variation_loss)):
            print('NaN spatial variation loss!')
        if torch.any(torch.isnan(tv_loss)):
            print('NaN TV loss!')
        if torch.any(torch.isnan(chroma_order_loss)):
            print('NaN chromatic order loss!')
        loss = self.w_intensity * intensity_loss + self.w_spatial * spatial_variation_loss + self.w_saturation * saturation_loss + self.w_tv * tv_loss + self.w_order * chroma_order_loss
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

class BackscatterLoss(nn.Module):

    def __init__(self, override=None, cost_ratio: float=1000.0, upper_bound: float=1.0, w_upper: float=0.1, w_corr: float=0.0, debug: bool=False):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.cost_ratio = cost_ratio
        self.override = override
        self.upper_bound = upper_bound
        self.w_upper = w_upper
        self.w_corr = w_corr
        self.debug = debug

    def forward(self, direct: torch.Tensor, depth: torch.Tensor=None, upper_bound: float=None) -> torch.Tensor:
        if self.override is not None:
            return self.override
        neg_loss = F.softplus(-direct, beta=5.0).mean()
        pos_loss = self.smooth_l1(F.relu(direct), torch.zeros_like(direct))
        ub = self.upper_bound if upper_bound is None else upper_bound
        upper_loss = direct.new_tensor(0.0)
        if ub is not None:
            ub_broadcast = ub
            if not torch.is_tensor(ub_broadcast):
                ub_broadcast = torch.as_tensor(ub_broadcast, dtype=direct.dtype, device=direct.device)
            upper_loss = F.softplus(direct - ub_broadcast, beta=5.0).mean()
        corr_pen = direct.new_tensor(0.0)
        if depth is not None and self.w_corr > 0:
            B, C, H, W = direct.shape
            x = direct.view(B, C, -1)
            y = depth.detach().expand_as(direct).contiguous().view(B, C, -1)
            x = x - x.mean(dim=-1, keepdim=True)
            y = y - y.mean(dim=-1, keepdim=True)
            x = x / (x.std(dim=-1, keepdim=True) + 1e-06)
            y = y / (y.std(dim=-1, keepdim=True) + 1e-06)
            corr = (x * y).mean(dim=-1)
            corr_pen = self.relu(-corr).mean()
        loss = self.cost_ratio * neg_loss + pos_loss + self.w_upper * upper_loss + self.w_corr * corr_pen
        loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
        if self.debug:
            with torch.no_grad():
                print(f'[BackscatterLoss] neg:{neg_loss.item():.4e} pos:{pos_loss.item():.4e} ub:{upper_loss.item():.4e} corr:{corr_pen.item():.4e} total:{loss.item():.4e}')
        return loss

class DarkChannelPriorLossV3(nn.Module):

    def __init__(self, lambda_neg=1000.0):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.2)
        self.lambda_neg = lambda_neg

    def forward(self, dark_channel_map, depth=None):
        pos_part = torch.relu(dark_channel_map)
        loss_pos = self.l1(pos_part, torch.zeros_like(pos_part))
        neg_part = torch.relu(-dark_channel_map)
        loss_neg = self.smooth_l1(neg_part, torch.zeros_like(neg_part))
        total_loss = loss_pos + self.lambda_neg * loss_neg
        return (total_loss, dark_channel_map)

class DeattenuateLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.relu = nn.ReLU()
        self.target_intensity = 0.5

    def forward(self, direct, J):
        saturation_loss = (self.relu(-J) + self.relu(J - 1)).square().mean()
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_intensities = torch.mean(J, dim=[2, 3], keepdim=True)
        channel_spatial = torch.std(J, dim=[2, 3])
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(saturation_loss)):
            print('NaN saturation loss!')
        if torch.any(torch.isnan(intensity_loss)):
            print('NaN intensity loss!')
        if torch.any(torch.isnan(spatial_variation_loss)):
            print('NaN spatial variation loss!')
        return saturation_loss + intensity_loss + spatial_variation_loss

class GrayWorldPriorLoss(nn.Module):

    def __init__(self, target_intensity=0.5):
        super().__init__()
        self.target_intensity = target_intensity

    def forward(self, J):
        if len(J.size()) == 4:
            channel_intensities = torch.mean(J, dim=[-2, -1], keepdim=True)
        elif len(J.size()) == 3:
            channel_intensities = torch.mean(J, dim=[-1])
        else:
            assert False
        intensity_loss = (channel_intensities - self.target_intensity).square().mean()
        if torch.any(torch.isnan(intensity_loss)):
            print('NaN intensity loss!')
        return intensity_loss

class RgbSpatialVariationLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, J, direct):
        init_spatial = torch.std(direct, dim=[2, 3])
        channel_spatial = torch.std(J, dim=[2, 3])
        spatial_variation_loss = self.mse(channel_spatial, init_spatial)
        if torch.any(torch.isnan(spatial_variation_loss)):
            print('NaN spatial variation loss!')
        return spatial_variation_loss

class RgbSaturationLoss(nn.Module):

    def __init__(self, saturation_val: float):
        super().__init__()
        self.relu = nn.ReLU()
        self.saturation_val = saturation_val

    def forward(self, rgb):
        saturation_loss = (self.relu(-rgb) + self.relu(rgb - self.saturation_val)).square().mean()
        if torch.any(torch.isnan(saturation_loss)):
            print('NaN saturation loss!')
        return saturation_loss

class AlphaBackgroundLoss(nn.Module):

    def __init__(self, use_kornia: bool=False):
        super().__init__()
        self.use_kornia = use_kornia
        if use_kornia:
            self.range = math.sqrt(100 * 100 + 255 * 255 + 255 * 255)
            self.threshold = 50
        else:
            self.range = math.sqrt(3)
            self.threshold = 0.2 * math.sqrt(3)
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, rgb, background, alpha):
        if self.use_kornia:
            lab_background = rgb_to_lab(background.reshape(3, 1, 1))
            lab_image = rgb_to_lab(rgb)
            diff = lab_image - lab_background
            dist = torch.linalg.vector_norm(diff, dim=0)
        elif len(rgb.size()) == 2:
            diff = rgb - background
            dist = torch.linalg.vector_norm(diff, dim=1)
        else:
            diff = rgb - background.reshape(3, 1, 1)
            dist = torch.linalg.vector_norm(diff, dim=0)
        new_approach = False
        if new_approach:
            clamped_diff = torch.max(dist - self.threshold, torch.Tensor([0.0]).cuda())
            if self.use_kornia:
                mask = torch.exp(-clamped_diff / 10)
            else:
                mask = torch.exp(-clamped_diff / 0.05)
            masked_alpha = alpha * mask
            try:
                if torch.sum(mask) == 0:
                    loss = torch.Tensor([0.0]).squeeze().cuda()
                else:
                    loss = self.mse(masked_alpha, torch.zeros_like(masked_alpha))
            except:
                import pdb
                pdb.set_trace()
        else:
            mask = dist < self.threshold
            if len(alpha.size()) == 1:
                masked_alpha = alpha[mask]
            else:
                masked_alpha = alpha[:, mask]
            try:
                if torch.sum(mask) == 0:
                    loss = torch.Tensor([0.0]).squeeze().cuda()
                else:
                    loss = self.l1(masked_alpha, torch.zeros_like(masked_alpha))
            except:
                import pdb
                pdb.set_trace()
        return loss

def mixture_of_laplacians_loss(x):
    lp1 = torch.exp(-torch.abs(x) / 0.1)
    lp2 = torch.exp(-torch.abs(1 - x) / 0.1)
    return -torch.mean(torch.log(lp1 + lp2))
