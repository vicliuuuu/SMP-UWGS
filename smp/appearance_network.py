import torch
import torch.nn as nn
import torch.nn.functional as F

def decouple_appearance(image, gaussians, view_idx):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    H, W = (image.size(1), image.size(2))
    crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode='bilinear', align_corners=True)[0]
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down, H, W).squeeze()
    transformed_image = mapping_image * image
    return (transformed_image, mapping_image)

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.global_avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FiLMGenerator(nn.Module):

    def __init__(self, appearance_dim, num_layers=5):
        super(FiLMGenerator, self).__init__()
        self.num_layers = num_layers
        layer_dims = [64, 128, 256, 512, 1024]
        self.film_generators = nn.ModuleList()
        for dim in layer_dims:
            generator = nn.Sequential(nn.Linear(appearance_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, dim * 2))
            self.film_generators.append(generator)

    def forward(self, appearance_embedding):
        film_params = []
        for generator in self.film_generators:
            params = generator(appearance_embedding)
            gamma, beta = torch.chunk(params, 2, dim=1)
            film_params.append((gamma, beta))
        return film_params

class FiLMLayer(nn.Module):

    def __init__(self):
        super(FiLMLayer, self).__init__()

    def forward(self, x, gamma, beta):
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta

class ImprovedUpsampleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(ImprovedUpsampleBlock, self).__init__()
        self.scale_factor = scale_factor
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x = self.conv_block(x)
        return x

class FiLMEncoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, layer_idx):
        super(FiLMEncoderBlock, self).__init__()
        self.layer_idx = layer_idx
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.film = FiLMLayer()
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, film_params=None):
        x = self.conv1(x)
        x = self.conv2(x)
        if film_params is not None:
            gamma, beta = film_params
            x = self.film(x, gamma, beta)
        x = self.se(x)
        skip = x
        x = self.pool(x)
        return (x, skip)

class FiLMDecoderBlock(nn.Module):

    def __init__(self, in_channels, out_channels, layer_idx):
        super(FiLMDecoderBlock, self).__init__()
        self.layer_idx = layer_idx
        self.upsample = ImprovedUpsampleBlock(in_channels, out_channels)
        self.conv1 = ConvBlock(out_channels * 2, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.se = SEBlock(out_channels)
        self.film = FiLMLayer()

    def forward(self, x, skip, film_params=None):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        if film_params is not None:
            gamma, beta = film_params
            x = self.film(x, gamma, beta)
        x = self.se(x)
        return x

class AppearanceNetwork(nn.Module):

    def __init__(self, num_input_channels, num_output_channels, appearance_dim=64):
        super(AppearanceNetwork, self).__init__()
        self.appearance_dim = appearance_dim
        self.film_generator = FiLMGenerator(appearance_dim)
        self.enc1 = FiLMEncoderBlock(num_input_channels, 64, 0)
        self.enc2 = FiLMEncoderBlock(64, 128, 1)
        self.enc3 = FiLMEncoderBlock(128, 256, 2)
        self.enc4 = FiLMEncoderBlock(256, 512, 3)
        self.bottleneck_conv1 = ConvBlock(512, 1024)
        self.bottleneck_conv2 = ConvBlock(1024, 1024)
        self.bottleneck_film = FiLMLayer()
        self.dec1 = FiLMDecoderBlock(1024, 512, 0)
        self.dec2 = FiLMDecoderBlock(512, 256, 1)
        self.dec3 = FiLMDecoderBlock(256, 128, 2)
        self.dec4 = FiLMDecoderBlock(128, 64, 3)
        self.final_upsample1 = ImprovedUpsampleBlock(64, 32)
        self.final_upsample2 = ImprovedUpsampleBlock(32, 16)
        self.output_conv1 = ConvBlock(16, 16)
        self.output_conv2 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, H, W, appearance_embedding=None):
        input_H, input_W = (x.shape[2], x.shape[3])
        film_params_list = None
        if appearance_embedding is not None:
            film_params_list = self.film_generator(appearance_embedding)
        e1, s1 = self.enc1(x, film_params_list[0] if film_params_list else None)
        e2, s2 = self.enc2(e1, film_params_list[1] if film_params_list else None)
        e3, s3 = self.enc3(e2, film_params_list[2] if film_params_list else None)
        e4, s4 = self.enc4(e3, film_params_list[3] if film_params_list else None)
        b = self.bottleneck_conv1(e4)
        b = self.bottleneck_conv2(b)
        if film_params_list and len(film_params_list) > 4:
            b = self.bottleneck_film(b, film_params_list[4][0], film_params_list[4][1])
        d1 = self.dec1(b, s4, film_params_list[3] if film_params_list else None)
        d2 = self.dec2(d1, s3, film_params_list[2] if film_params_list else None)
        d3 = self.dec3(d2, s2, film_params_list[1] if film_params_list else None)
        d4 = self.dec4(d3, s1, film_params_list[0] if film_params_list else None)
        if d4.shape[2:] != (input_H, input_W):
            d4 = F.interpolate(d4, size=(input_H, input_W), mode='bilinear', align_corners=True)
        x = self.final_upsample1(d4)
        x = self.final_upsample2(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        x = self.output_conv1(x)
        x = self.output_conv2(x)
        x = self.sigmoid(x)
        return x

def decouple_appearance_with_film(image, gaussians, view_idx):
    appearance_embedding = gaussians.get_apperance_embedding(view_idx)
    H, W = (image.size(1), image.size(2))
    crop_image_down = torch.nn.functional.interpolate(image[None], size=(H // 32, W // 32), mode='bilinear', align_corners=True)[0]
    crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H // 32, W // 32, 1).permute(2, 0, 1)], dim=0)[None]
    mapping_image = gaussians.appearance_network(crop_image_down, H, W, appearance_embedding).squeeze()
    transformed_image = mapping_image * image
    return (transformed_image, mapping_image)
if __name__ == '__main__':
    H, W = (1200 // 32, 1600 // 32)
    input_channels = 3 + 64
    output_channels = 3
    appearance_dim = 64
    input = torch.randn(1, input_channels, H, W).cuda()
    appearance_embedding = torch.randn(1, appearance_dim).cuda()
    model = AppearanceNetwork(input_channels, output_channels, appearance_dim).cuda()
    output = model(input, H * 32, W * 32, appearance_embedding)
    print(f'FiLM Version:')
    print(f'Input shape: {input.shape}')
    print(f'Appearance embedding shape: {appearance_embedding.shape}')
    print(f'Output shape: {output.shape}')

    def count_parameters(model):
        return sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'FiLM Model Parameters: {count_parameters(model):,}')
