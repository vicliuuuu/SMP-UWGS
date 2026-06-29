import torch
import torch.nn.functional as F

def apply_underwater_formation(direct: torch.Tensor, depth: torch.Tensor, attenuate_net, backscatter_net, b_inf: torch.Tensor | None=None) -> torch.Tensor:
    attenuation = attenuate_net(depth)
    backscatter = backscatter_net(depth)
    if b_inf is None:
        b_inf = torch.sigmoid(backscatter_net.B_inf).view(3, 1, 1)
    else:
        b_inf = b_inf.view(-1, 3, 1, 1) if b_inf.dim() == 2 else b_inf.view(3, 1, 1)
    if direct.dim() == 3:
        direct = direct.unsqueeze(0)
    if b_inf.dim() == 2:
        underwater = direct * attenuation + b_inf * backscatter
    else:
        underwater = direct * attenuation + b_inf.view(3, 1, 1) * backscatter
    return torch.clamp(underwater.squeeze(0) if underwater.shape[0] == 1 else underwater, 0.0, 1.0)

def restore_from_underwater(underwater_img: torch.Tensor, depth: torch.Tensor, attenuate_net, backscatter_net, b_inf: torch.Tensor | None=None) -> torch.Tensor:
    if underwater_img.dim() == 3:
        underwater_img = underwater_img.unsqueeze(0)
    if depth.dim() == 3:
        depth = depth.unsqueeze(0)
    attenuation = attenuate_net(depth)
    backscatter = backscatter_net(depth)
    if b_inf is None:
        b_inf = torch.sigmoid(backscatter_net.B_inf)
    b_inf = b_inf.view(-1, 3, 1, 1) if b_inf.dim() == 2 else b_inf.view(1, 3, 1, 1)
    direct = (underwater_img - b_inf * backscatter) / (attenuation + 1e-08)
    return torch.clamp(direct.squeeze(0), 0.0, 1.0)
