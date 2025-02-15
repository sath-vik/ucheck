__all__ = ["UKAN", "KANLayer", "KANBlock", "DWConv", "DW_bn_relu", "PatchEmbed", "ConvLayer", "D_ConvLayer"]

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from kan import KANLinear
import math

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        # KAN configuration
        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(in_features, hidden_features,
                               grid_size=grid_size, spline_order=spline_order,
                               scale_noise=scale_noise, scale_base=scale_base,
                               scale_spline=scale_spline, base_activation=base_activation,
                               grid_eps=grid_eps, grid_range=grid_range)
            self.fc2 = KANLinear(hidden_features, out_features,
                               grid_size=grid_size, spline_order=spline_order,
                               scale_noise=scale_noise, scale_base=scale_base,
                               scale_spline=scale_spline, base_activation=base_activation,
                               grid_eps=grid_eps, grid_range=grid_range)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)

        self.dwconv = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B*N, C)).reshape(B, N, -1)
        x = self.dwconv(x, H, W)
        x = self.fc2(x.reshape(B*N, -1)).reshape(B, N, -1)
        return x

class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.norm2 = norm_layer(dim)
        self.layer = KANLayer(dim, drop=drop, no_kan=no_kan)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x, H, W):
        return x + self.drop_path(self.layer(self.norm2(x), H, W))

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.relu(self.bn(self.dwconv(x)))
        return x.flatten(2).transpose(1, 2)

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, patch_size, stride, patch_size//2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), H, W

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UKAN(nn.Module):
    def __init__(self, num_classes=19, input_channels=3, deep_supervision=False,
                 img_size=256, patch_size=4, in_chans=3, embed_dims=[64, 128, 256],
                 drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, 
                 depths=[1, 1, 1], no_kan=False):
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.embed_dims = embed_dims
        self.num_classes = num_classes

        # Encoder
        self.encoder1 = ConvLayer(input_channels, embed_dims[0])
        self.encoder2 = ConvLayer(embed_dims[0], embed_dims[1])
        self.encoder3 = ConvLayer(embed_dims[1], embed_dims[2])

        # Decoder with corrected channels
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[2])  # 256→256
        self.decoder2 = D_ConvLayer(embed_dims[2], embed_dims[1])   # 256→128
        self.decoder3 = D_ConvLayer(embed_dims[1], embed_dims[0])    # 128→64
        self.decoder4 = D_ConvLayer(embed_dims[0], embed_dims[0]//2)  # 64 → 32
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(embed_dims[0]//2, num_classes, kernel_size=1)
        )


        # # Final output
        # self.final_upsample = nn.Sequential(
        #     nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        #     nn.Conv2d(32, num_classes, kernel_size=1)
        # )

        # KAN Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.block1 = nn.ModuleList([KANBlock(embed_dims[1], drop=drop_rate, drop_path=dpr[0])])
        self.block2 = nn.ModuleList([KANBlock(embed_dims[2], drop=drop_rate, drop_path=dpr[1])])

        # Patch Embedding
        self.patch_embed3 = PatchEmbed(
            img_size//4, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[1]
        )
        self.patch_embed4 = PatchEmbed(
            img_size//8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2]
        )

        # Normalization
        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

    def forward(self, x):
        B = x.shape[0]
        
        # Encoder
        e1 = F.relu(F.max_pool2d(self.encoder1(x), 2, 2))  # (B,64,128,128)
        e2 = F.relu(F.max_pool2d(self.encoder2(e1), 2, 2))  # (B,128,64,64)
        e3 = F.relu(F.max_pool2d(self.encoder3(e2), 2, 2))  # (B,256,32,32)

        # KAN Processing
        x, H, W = self.patch_embed3(e3)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed4(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # Decoder with exact dimension matching
        d1 = F.interpolate(self.decoder1(x), size=e3.shape[2:], mode='bilinear', align_corners=True)
        d1 = torch.add(d1, e3)

        d2 = F.interpolate(self.decoder2(d1), size=e2.shape[2:], mode='bilinear', align_corners=True)
        d2 = torch.add(d2, e2)

        d3 = F.interpolate(self.decoder3(d2), size=e1.shape[2:], mode='bilinear', align_corners=True)
        d3 = torch.add(d3, e1)

        d4 = self.decoder4(d3)  # (B,32,128,128)
        output = self.final_upsample(d4)
        print(f"Input shape: {x.shape}")
        print(f"Final output shape: {output.shape}")


        return output
