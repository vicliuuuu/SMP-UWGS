import torch
import torch.nn as nn
import torch.nn.functional as F

class BackscatterNetV2(nn.Module):

    def __init__(self, use_residual: bool=False, scale: float=1.0, do_sigmoid: bool=False, init_vals: bool=False):
        super().__init__()
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        self.use_residual = use_residual
        if init_vals:
            self.backscatter_conv_params = nn.Parameter(torch.Tensor([0.95, 0.8, 0.8]).reshape(3, 1, 1, 1))
        else:
            self.backscatter_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if use_residual:
            self.residual_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
            self.J_prime = nn.Parameter(torch.rand(3, 1, 1))
        self.B_inf = nn.Parameter(torch.rand(3, 1, 1))
        self.relu = nn.ReLU()
        self.l2 = torch.nn.MSELoss()
        print(f'Using backscatterv2 with scale: {self.scale}, sigmoid: {self.do_sigmoid}')

    def forward(self, depth):
        raw_params = self.backscatter_conv_params
        if self.do_sigmoid:
            beta_val = self.scale * torch.sigmoid(raw_params)
        else:
            beta_val = F.softplus(raw_params)
        beta_b_conv = F.conv2d(depth, beta_val)
        Bc = torch.sigmoid(self.B_inf) * (1 - torch.exp(-beta_b_conv))
        if self.use_residual:
            raw_res_params = self.residual_conv_params
            if self.do_sigmoid:
                beta_res_val = self.scale * torch.sigmoid(raw_res_params)
            else:
                beta_res_val = F.softplus(raw_res_params)
            beta_d_conv = F.conv2d(depth, beta_res_val)
            Bc += torch.sigmoid(self.J_prime) * torch.exp(-beta_d_conv)
        return Bc

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

class AttenuateNetV3(nn.Module):

    def __init__(self, scale: float=1.0, do_sigmoid: bool=False, init_vals: bool=True):
        super().__init__()
        self.attenuation_conv_params = nn.Parameter(torch.rand(3, 1, 1, 1))
        if init_vals:
            self.attenuation_conv_params = nn.Parameter(torch.Tensor([1.1, 0.95, 0.95]).reshape(3, 1, 1, 1))
        self.scale = scale
        self.do_sigmoid = do_sigmoid
        self.attenuation_coef = None
        self.relu = nn.ReLU()
        print(f'Using attenuatenetv3 with scale: {self.scale}, sigmoid: {self.do_sigmoid}')

    def forward(self, depth):
        raw_params = self.attenuation_conv_params
        if self.do_sigmoid:
            beta_val = self.scale * torch.sigmoid(raw_params)
        else:
            beta_val = F.softplus(raw_params)
        beta_d_conv = F.conv2d(depth, beta_val)
        attenuation_map = torch.exp(-beta_d_conv)
        return attenuation_map
