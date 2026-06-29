import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int=32):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=True)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.sigmoid(self.psi(psi))
        return x * psi

class ResidualBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, in_channels // 16, kernel_size=1), nn.ReLU(), nn.Conv2d(in_channels // 16, in_channels, kernel_size=1), nn.Sigmoid())

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        se_weight = self.se(x)
        x = x * se_weight
        x += identity
        return F.relu(x)

class WaterParamPredictor(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), ResidualBlock(64), nn.MaxPool2d(2))
        self.encoder2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), ResidualBlock(128), nn.MaxPool2d(2))
        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2), ResidualBlock(64))
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2), ResidualBlock(64))
        self.attention_gate = AttentionGate(64, 64)
        self.global_conv = nn.Sequential(nn.Conv2d(64, 256, kernel_size=3, padding=1), nn.AdaptiveAvgPool2d((1, 1)))
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 9))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self._init_physics_prior()

    def _init_physics_prior(self):
        import torch.nn.functional as F
        target_beta_d = torch.tensor([0.75, 0.3, 0.5])
        target_beta_b = torch.tensor([0.75, 0.4, 0.4])
        target_b_inf = torch.tensor([0.1, 0.25, 0.45])
        eps = 1e-06
        logit_beta_d = torch.log(torch.exp(target_beta_d) - 1 + eps)
        logit_beta_b = torch.log(torch.exp(target_beta_b) - 1 + eps)
        logit_b_inf = torch.log(target_b_inf / (1.0 - target_b_inf + eps))
        initial_bias = torch.cat([logit_beta_d, logit_beta_b, logit_b_inf])
        last_layer = self.fc[-1]
        if last_layer.bias is not None:
            last_layer.bias.data = initial_bias.clone()
        nn.init.normal_(last_layer.weight, mean=0.0, std=0.05)
        mid_layer = self.fc[0]
        nn.init.xavier_normal_(mid_layer.weight, gain=0.35)
        print('💧 网络已初始化为【物理先验模式】:')
        print(f'   初始基准 beta_d: {F.softplus(logit_beta_d).tolist()}')
        print(f'   初始基准 beta_b: {F.softplus(logit_beta_b).tolist()}')
        print(f'   初始基准 b_inf : {torch.sigmoid(logit_b_inf).tolist()}')
        print('   策略：FC 层权重已归零，网络将从基准值开始，随训练学习根据深度图进行微调。')

    def forward(self, depth):
        depth = F.interpolate(depth, size=(1200, 800), mode='bilinear', align_corners=True)
        e1 = self.encoder1(depth)
        e2 = self.encoder2(e1)
        d1 = self.decoder1(e2)
        e1_att = self.attention_gate(d1, e1)
        d1 = d1 + e1_att
        d2 = self.decoder2(d1)
        x = self.global_conv(d2)
        x = x.view(x.size(0), -1)
        params = self.fc(x)
        beta_d = F.softplus(params[:, :3])
        beta_b = F.softplus(params[:, 3:6])
        b_inf = torch.sigmoid(params[:, 6:])
        return (beta_d, beta_b, b_inf)
