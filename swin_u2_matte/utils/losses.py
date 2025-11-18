import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. SSIM Loss---
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size // 2)**2 / float(2 * sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        # img1, img2: [Batch, 1, H, W] (Normalized 0~1)
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel

        return 1 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# --- 2. Integrated Loss (BCE + IoU + SSIM) ---
class SwinU2Loss(nn.Module):
    def __init__(self, bce_w=1.0, iou_w=1.0, ssim_w=1.0):
        super(SwinU2Loss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ssim_loss = SSIM(window_size=11)
        self.bce_w = bce_w
        self.iou_w = iou_w
        self.ssim_w = ssim_w

    def forward(self, logits, targets):
        """
        logits: Model output before sigmoid [B, 1, H, W]
        targets: Ground Truth [B, 1, H, W] (0 or 1)
        """
        
        # 1. BCE Loss (Pixel-wise)
        bce = self.bce_loss(logits, targets)

        # 2. Sigmoid for IoU and SSIM
        preds = torch.sigmoid(logits)

        # 3. IoU Loss (Region-wise)
        # Flatten: [B, H*W]
        preds_flat = preds.view(preds.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (preds_flat * targets_flat).sum(1)
        total = (preds_flat + targets_flat).sum(1)
        union = total - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1 - iou.mean()

        # 4. SSIM Loss (Structure-wise)
        # SSIM needs standard [0,1] input, so we use preds (after sigmoid)
        ssim = self.ssim_loss(preds, targets)

        # Combine
        loss = (self.bce_w * bce) + (self.iou_w * iou_loss) + (self.ssim_w * ssim)
        
        return loss