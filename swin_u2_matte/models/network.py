import torch
import torch.nn as nn
import torch.nn.functional as F

from .rsu_blocks import RSU7, RSU6, RSU5, RSU4, RSU4F, REBNCONV, _upsample_like
from .swin_transformer import SwinTransformer

class SwinU2Matte(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(SwinU2Matte, self).__init__()

        # --- 1. Swin Transformer Branch (Encoder 2) ---
        # Load Swin-Large with pretrained weights
        self.swin_unet = SwinTransformer(
            pretrained_path='./pretrained/swin_large_patch4_window12_384_22k.pth'
        )
        
        # Project channels: 192 (Swin-Large) -> 256 (U2-Net)
        self.swin_projection = nn.Conv2d(192, 256, kernel_size=1)

        # --- 2. U2-Net Encoder (Encoder 1) ---
        self.stage1 = RSU7(in_ch, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Fusion: CNN (256) + Swin (256) = 512 -> 256
        self.fusion_conv = REBNCONV(512, 256, dirate=1) 

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # --- 3. Decoder ---
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # --- 4. Side Outputs ---
        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

    def forward(self, x):
        # x shape: [Batch, 3, 768, 768]

        # --- Swin Branch ---
        # Resize to 384x384 for Swin input
        x_swin_in = F.interpolate(x, size=(384, 384), mode='bilinear', align_corners=True)
        swin_outs = self.swin_unet(x_swin_in)
        
        # Get Stage 0 features
        swin_feat = swin_outs[0]                    # [Batch, 192, 96, 96]
        swin_feat = self.swin_projection(swin_feat) # [Batch, 256, 96, 96]

        # --- CNN Encoder Branch ---
        hx1 = self.stage1(x)         # [Batch, 64, 768, 768]
        hx = self.pool12(hx1)        # [Batch, 64, 384, 384] (Pooling)

        hx2 = self.stage2(hx)        # [Batch, 128, 384, 384]
        hx = self.pool23(hx2)        # [Batch, 128, 192, 192] (Pooling)

        hx3 = self.stage3(hx)        # [Batch, 256, 192, 192]
        hx = self.pool34(hx3)        # [Batch, 256, 96, 96]   (Pooling) -> FUSION POINT

        # --- Feature Fusion ---
        # Concatenate CNN (96x96) and Swin (96x96)
        hx = torch.cat((hx, swin_feat), 1) # [Batch, 512, 96, 96]
        hx = self.fusion_conv(hx)          # [Batch, 256, 96, 96]

        # Continue Encoder
        hx4 = self.stage4(hx)        # [Batch, 512, 96, 96]
        hx = self.pool45(hx4)        # [Batch, 512, 48, 48]   (Pooling)

        hx5 = self.stage5(hx)        # [Batch, 512, 48, 48]
        hx = self.pool56(hx5)        # [Batch, 512, 24, 24]   (Pooling)

        hx6 = self.stage6(hx)        # [Batch, 512, 24, 24]   (Deepest Layer)
        
        # --- Decoder ---
        hx6up = _upsample_like(hx6, hx5)                  # [Batch, 512, 48, 48]
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))   # [Batch, 512, 48, 48]

        hx5dup = _upsample_like(hx5d, hx4)                # [Batch, 512, 96, 96]
        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))  # [Batch, 256, 96, 96]

        hx4dup = _upsample_like(hx4d, hx3)                # [Batch, 256, 192, 192]
        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))  # [Batch, 128, 192, 192]

        hx3dup = _upsample_like(hx3d, hx2)                # [Batch, 128, 384, 384]
        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))  # [Batch, 64, 384, 384]

        hx2dup = _upsample_like(hx2d, hx1)                # [Batch, 64, 768, 768]
        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))  # [Batch, 64, 768, 768]

        # --- Side Outputs (Deep Supervision) ---
        d1 = self.side1(hx1d) # [Batch, 1, 768, 768]
        d2 = self.side2(hx2d) # [Batch, 1, 384, 384]
        d3 = self.side3(hx3d) # [Batch, 1, 192, 192]
        d4 = self.side4(hx4d) # [Batch, 1, 96, 96]
        d5 = self.side5(hx5d) # [Batch, 1, 48, 48]
        d6 = self.side6(hx6)  # [Batch, 1, 24, 24]

        # Return upsampled outputs (All resize back to 768x768 for Loss Calc)
        return [
            _upsample_like(d1, x),
            _upsample_like(d2, x),
            _upsample_like(d3, x),
            _upsample_like(d4, x),
            _upsample_like(d5, x),
            _upsample_like(d6, x)
        ]