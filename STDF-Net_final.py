import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


from pytorch_wavelets import DWTForward
class ConvNormAct2d(nn.Module):
    """Grouped 2D convolution followed by batch normalization and SiLU."""

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=d * (k - 1) // 2,
                              dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HFPSA(nn.Module):
    """High-Frequency Preserving Self-Attention (Section III-C, Fig. 8)."""
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 pooled_sizes=[11, 8, 6, 4], q_pooled_size=16, q_conv=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.num_elements = np.array([t * t for t in pooled_sizes]).sum()
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias))
        self.kv = nn.Sequential(nn.Linear(dim, dim * 2, bias=qkv_bias))

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.pooled_sizes = pooled_sizes
        self.pools = nn.ModuleList()
        self.eps = 0.001

        self.norm = nn.LayerNorm(dim)
        self.q_pooled_size = q_pooled_size

        if q_conv and self.q_pooled_size > 1:
            self.q_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
            self.q_norm = nn.LayerNorm(dim)
        else:
            self.q_conv = None
            self.q_norm = None

        self.d_convs = nn.ModuleList(
            [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim) for temp in pooled_sizes])

    def forward(self, x):
        B, C, H, W = x.size()
        x = x.flatten(2).permute(0, 2, 1)

        if self.q_pooled_size > 1:
            q_pooled_size = (self.q_pooled_size, int(round(W * float(self.q_pooled_size) / H + self.eps))) \
                if W >= H else (int(round(H * float(self.q_pooled_size) / W + self.eps)), self.q_pooled_size)

            q = F.adaptive_avg_pool2d(x.transpose(1, 2).reshape(B, C, H, W), q_pooled_size)
            _, _, H1, W1 = q.shape
            if self.q_conv is not None:
                q = q + self.q_conv(q)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
            else:
                q = q.view(B, C, -1).transpose(1, 2)
            q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            H1, W1 = H, W
            if self.q_conv is not None:
                x1 = x.view(B, -1, C).transpose(1, 2).reshape(B, C, H1, W1)
                q = x1 + self.q_conv(x1)
                q = self.q_norm(q.view(B, C, -1).transpose(1, 2))
                q = self.q(q).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
            else:
                q = self.q(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()

        pools = []
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        for (pooled_size, l) in zip(self.pooled_sizes, self.d_convs):
            pooled_size_tuple = (pooled_size, int(round(W * pooled_size / H + self.eps))) if W >= H else (
                int(round(H * pooled_size / W + self.eps)), pooled_size)

            avg_pool = F.adaptive_avg_pool2d(x_, pooled_size_tuple)
            max_pool = F.adaptive_max_pool2d(x_, pooled_size_tuple)
            pool = avg_pool + max_pool

            pool = pool + l(pool)
            pools.append(pool.view(B, C, -1))

        pools = torch.cat(pools, dim=2)
        pools = self.norm(pools.permute(0, 2, 1))

        kv = self.kv(pools).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v)
        x = x.transpose(1, 2).reshape(B, -1, C)

        x = self.proj(x)
        x = x.transpose(1, 2).reshape(B, C, H1, W1)

        if self.q_pooled_size > 1:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)

        return x


class FSSFB(nn.Module):
    """Frequency-Spatial Self-Attention Fusion Block (Section III-C, Fig. 5)."""
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.c1 = c1
        self.conv1 = ConvNormAct2d(c1, 2 * c1, 1, g=c1)
        self.hwt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv2 = ConvNormAct2d(c1, c2, k, 2, g=math.gcd(c1, c2))
        self.conv3 = ConvNormAct2d(c1 * 3, c2, 3, d=1, g=math.gcd(c1 * 3, c2))

        self.hfpsa = HFPSA(dim=c2, num_heads=4)

        self.conv4 = ConvNormAct2d(c1, c2, 3, g=math.gcd(c1, c2))
        self.conv5 = ConvNormAct2d(2 * c2, c2, 1)

    def _wavelet_forward_fp32(self, x):
        with torch.autocast(device_type=x.device.type if x.device.type != 'cpu' else 'cpu', enabled=False):
            self.hwt = self.hwt.float()
            return self.hwt(x.float())

    def forward(self, x):
        x0 = self.conv1(x)
        x1, x2 = torch.split(x0, self.c1, dim=1)
        conv_spatial = self.conv2(x1)

        yL, yH = self._wavelet_forward_fp32(x2)
        yL = yL.to(dtype=x.dtype)

        y_HL = yH[0][:, :, 0, :].to(dtype=x.dtype)
        y_LH = yH[0][:, :, 1, :].to(dtype=x.dtype)
        y_HH = yH[0][:, :, 2, :].to(dtype=x.dtype)

        high_frequency_fused = torch.cat([y_HL, y_LH, y_HH], dim=1)
        high_frequency_fused_output = self.conv3(high_frequency_fused)

        high_frequency_fused_output = self.hfpsa(high_frequency_fused_output)
        low_frequency_fused_output = self.conv4(yL)

        spatial_output = conv_spatial * high_frequency_fused_output

        fused = torch.cat([spatial_output, low_frequency_fused_output], dim=1)
        out = self.conv5(fused)
        return out


class DirectionalGating3D(nn.Module):
    """Directional branches and shared gate inside STDG-Conv (Fig. 3)."""
    def __init__(self, in_channels, out_channels, reduction=4):
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.shrink = nn.Conv3d(in_channels, mid, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)

        self.conv_w = nn.Conv3d(mid, mid, kernel_size=(1, 1, 3), padding=(0, 0, 1), bias=False)
        self.conv_h = nn.Conv3d(mid, mid, kernel_size=(1, 3, 1), padding=(0, 1, 0), bias=False)
        self.conv_t = nn.Conv3d(mid, mid, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)

        self.gate = nn.Conv3d(mid, mid, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.expand = nn.Conv3d(mid * 3, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x_shrink = self.act(self.shrink(x))

        feat_w = self.act(self.conv_w(x_shrink))
        feat_h = self.act(self.conv_h(x_shrink))
        feat_t = self.act(self.conv_t(x_shrink))

        gate = self.sigmoid(self.gate(x_shrink))

        w_gated = feat_w * gate
        h_gated = feat_h * gate
        t_gated = feat_t * gate

        combined = torch.cat([w_gated, h_gated, t_gated], dim=1)
        return self.expand(combined)


class STDGConv(nn.Module):
    """Spatio-Temporal Decoupled Gated Convolution (Section III-B, Fig. 3)."""
    def __init__(self, in_channels, hidden_channels, kernel_size=3, reduction=2):
        super().__init__()


        self.dwconv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False
        )
        self.norm = nn.BatchNorm3d(in_channels)
        self.act = nn.SiLU(inplace=True)


        self.pwconv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            bias=False
        )


        self.directional_gating = DirectionalGating3D(in_channels=hidden_channels, out_channels=hidden_channels, reduction=reduction)

    def forward(self, x):


        x_3d = x.permute(0, 2, 1, 3, 4).contiguous()


        feat = self.dwconv3d(x_3d)
        feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.pwconv3d(feat)


        out = feat + self.directional_gating(feat)


        return out


class TWA(nn.Module):
    """Temporal Weighted Aggregation over the input frames (Fig. 4)."""
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.weight = nn.Parameter(torch.ones(T))

    def forward(self, x):
        w = torch.softmax(self.weight, dim=0).view(1, 1, self.T, 1, 1)
        return (x * w).sum(dim=2)


class SpatialEncoder(nn.Module):
    """Three spatial scales with two FSSFB downsampling stages (Fig. 2)."""
    def __init__(self, in_ch):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.stage2 = FSSFB(c1=32, c2=48, k=3)
        self.stage3 = FSSFB(c1=48, c2=64, k=3)

    def forward(self, x):
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


class SpatialDecoder(nn.Module):
    """Spatial reconstruction using skip features and a pixel prediction head."""
    def __init__(self, out_channels=1):
        super().__init__()

        self.up3 = nn.ConvTranspose2d(64, 48, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(48 + 48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.up2 = nn.ConvTranspose2d(48, 32, kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2)
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, feats, orig_size):
        f1, f2, f3 = feats

        x = self.up3(f3)
        if x.shape[-2:] != f2.shape[-2:]:
            x = F.interpolate(x, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, f2], dim=1)
        x = self.conv2(x)

        x = self.up2(x)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode='bilinear', align_corners=False)

        x = torch.cat([x, f1], dim=1)
        x = self.conv1(x)

        if x.shape[-2:] != orig_size:
            x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)

        out = self.out_conv(x)
        return out


class STDFNet(nn.Module):
    """STDF-Net: STDG-Conv -> TWA -> spatial encoder -> spatial decoder."""
    def __init__(self, in_channels=1, hidden=8, timesteps=5, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.timesteps = timesteps


        self.stdg_conv = STDGConv(in_channels=in_channels, hidden_channels=hidden, kernel_size=3, reduction=2)

        self.twa = TWA(T=timesteps)
        self.encoder = SpatialEncoder(in_ch=hidden)
        self.decoder = SpatialDecoder(out_channels=out_channels)

    def forward(self, x):

        target_T = self.timesteps
        target_C = self.in_channels

        for i in reversed(range(1, x.dim())):
            if x.shape[i] == 1:
                x = x.squeeze(i)

        if target_C == 1 and x.dim() == 4:
            x = x.unsqueeze(2)
        elif target_C > 1 and target_T > 1 and x.dim() == 5:
            if x.shape[-1] == target_C:
                x = x.permute(0, 1, 4, 2, 3)
            elif x.shape[1] == target_C and x.shape[2] == target_T:
                x = x.permute(0, 2, 1, 3, 4)

        orig_size = (x.size(-2), x.size(-1))


        feat3d = self.stdg_conv(x)


        feat2d = self.twa(feat3d)


        feats = self.encoder(feat2d)
        out = self.decoder(feats, orig_size)

        return out


class STDFLoss(nn.Module):
    """Weighted BCE, Dice, Focal Tversky and structural similarity (Section III-D)."""
    def __init__(self, pos_weight=5.0, alpha=0.7, beta=0.3, gamma=0.75):
        super().__init__()
        self.pos_weight = pos_weight
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

    def forward(self, pred, target):
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        if pred.dim() == 2:
            pred = pred.unsqueeze(0).unsqueeze(0)
        if target.dim() == 2:
            target = target.unsqueeze(0).unsqueeze(0)


        target = target.float()

        pred_prob = torch.sigmoid(pred)
        smooth = 1e-5

        bce_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.pos_weight], device=pred.device, dtype=pred.dtype)
        )
        bce = bce_loss_fn(pred, target)

        intersection = (pred_prob * target).sum()
        dice = 1 - (2. * intersection + smooth) / (pred_prob.sum() + target.sum() + smooth)

        tp = (pred_prob * target).sum()
        fp = ((1 - target) * pred_prob).sum()
        fn = (target * (1 - pred_prob)).sum()
        tversky = (tp + smooth) / (tp + self.alpha * fp + self.beta * fn + smooth)
        focal_tversky = (1 - tversky) ** self.gamma

        from pytorch_msssim import ssim

        ssim_loss = 1 - ssim(pred_prob.float(), target.float(), data_range=1.0, size_average=True)

        return 0.5 * bce + 0.3 * dice + 0.15 * focal_tversky + 0.05 * ssim_loss



__all__ = ["STDFNet", "STDGConv", "TWA", "FSSFB", "HFPSA",
           "DirectionalGating3D", "SpatialEncoder", "SpatialDecoder", "STDFLoss"]
