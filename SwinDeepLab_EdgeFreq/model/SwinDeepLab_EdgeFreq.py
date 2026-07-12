import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm


# ===================== 1. 基础模块 (修正版) =====================

class GaussianFilter(nn.Module):
    # 保持不变：用于平滑底层特征，去噪
    def __init__(self, channels, kernel_size=5, sigma=0.5):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        x, y = np.meshgrid(np.linspace(-1, 1, kernel_size), np.linspace(-1, 1, kernel_size))
        d = np.sqrt(x ** 2 + y ** 2)
        g = np.exp(-(d ** 2 / (2 * sigma ** 2)))
        g /= g.sum()
        kernel = torch.from_numpy(g).float().repeat(channels, 1, 1, 1)
        self.kernel = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        return F.conv2d(x, self.kernel, padding=self.kernel_size // 2, groups=self.channels)


class DirectionalFilter(nn.Module):
    # 保持不变：捕捉不同方向的纹理特征
    def __init__(self, in_channels, out_channels, angles=[0, 45, 90, 135]):
        super().__init__()
        self.directional_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels // len(angles), kernel_size=3, padding=1, bias=False)
            for _ in angles
        ])

    def forward(self, x):
        return torch.cat([conv(x) for conv in self.directional_convs], dim=1)


class GuidedFilter(nn.Module):
    # 保持不变：边缘保持滤波
    def __init__(self, r=4, eps=1e-3):
        super().__init__()
        self.r = r
        self.eps = eps
        self.box_filter = nn.AvgPool2d(kernel_size=2 * r + 1, stride=1, padding=r)

    def forward(self, x, guide):
        if x.shape[2:] != guide.shape[2:]:
            x = F.interpolate(x, size=guide.shape[2:], mode='bilinear', align_corners=False)
        if guide.shape[1] != x.shape[1]:
            if not hasattr(self, 'proj'):
                self.proj = nn.Conv2d(guide.shape[1], x.shape[1], 1).to(guide.device)
            guide = self.proj(guide)

        mean_g = self.box_filter(guide)
        mean_x = self.box_filter(x)
        cov_gx = self.box_filter(guide * x) - mean_g * mean_x
        var_g = self.box_filter(guide * guide) - mean_g * mean_g

        a = cov_gx / (var_g + self.eps)
        b = mean_x - a * mean_g
        return self.box_filter(a) * guide + self.box_filter(b)


class CBAM(nn.Module):
    # 保持不变：通道和空间注意力机制
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(F.adaptive_avg_pool2d(x, 1).view(b, -1)).view(b, -1, 1, 1)
        max_out = self.fc(F.adaptive_max_pool2d(x, 1).view(b, -1)).view(b, -1, 1, 1)
        x = x * torch.sigmoid(avg_out + max_out)
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        return x * self.spatial(torch.cat([avg_s, max_s], dim=1))


# 【修改点 1】：将低通滤波器改为高频增强器 (High-Frequency Enhancer)
class HighFreqEnhancer(nn.Module):
    """提取高频(边缘)信息并叠加到原特征上，而不是滤除它们"""

    def __init__(self, enhance_factor=0.5):
        super().__init__()
        self.enhance_factor = enhance_factor

    def forward(self, x):
        f = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
        f_shift = torch.fft.fftshift(f)
        h, w = x.shape[-2:]
        cx, cy = h // 2, w // 2
        r = int(min(cx, cy) * 0.05)  # 只遮罩最中心的低频部分

        y, x_grid = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing='ij')
        mask = ((y - cx) ** 2 + (x_grid - cy) ** 2 > r ** 2).float()  # 高通掩码
        mask = mask.expand_as(f_shift)

        # 只保留高频部分
        f_high = f_shift * mask
        img_high = torch.fft.ifft2(torch.fft.ifftshift(f_high), dim=(-2, -1), norm='ortho')
        img_high = torch.real(img_high)

        # 融入残差：原特征 + 增强的高频边缘特征
        return x + self.enhance_factor * img_high


# ===================== 2. 功能模块 =====================

class FeatureAlignmentModule(nn.Module):
    # 保持不变
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.aligned_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        offset = self.offset_conv(x)
        return self.aligned_conv(x + offset)


class ASPPEnhanced(nn.Module):
    # 保持不变
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super().__init__()
        self.aspp_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels),
                          nn.ReLU(inplace=True))
        ])
        for rate in atrous_rates:
            self.aspp_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=rate, dilation=rate, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
            ))
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        total_channels = out_channels * (len(atrous_rates) + 2)
        self.directional = DirectionalFilter(total_channels, out_channels)
        self.cbam = CBAM(out_channels)
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = [block(x) for block in self.aspp_blocks]
        gp = F.interpolate(self.global_pool(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        res.append(gp)
        x = torch.cat(res, dim=1)
        x = self.directional(x)
        x = self.cbam(x)
        return self.project(x)


# ===================== 3. 核心集成模型 =====================

class SwinDeepLabV3PlusEnhanced(nn.Module):
    def __init__(self, num_classes=6, in_channels=3, input_res=256,
                 use_aspp_enhanced=True,  # 开关1：方向滤波与注意力增强ASPP
                 use_guided_fusion=True,  # 开关2：FAM特征对齐与引导滤波
                 use_edge_branch=True,  # 开关3：显式边缘解耦分支
                 use_fft=True):  # 开关4：频域高频残差增强
        super().__init__()

        self.use_aspp_enhanced = use_aspp_enhanced
        self.use_guided_fusion = use_guided_fusion
        self.use_edge_branch = use_edge_branch
        self.use_fft = use_fft

        # 1. 骨干网络
        self.backbone = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=True,
            in_chans=in_channels,
            features_only=True,
            img_size=input_res
        )

        # 2. 多尺度模块 (ASPP)
        if self.use_aspp_enhanced:
            self.aspp = ASPPEnhanced(768, 256)
        else:
            # 退化为基础的降维卷积（模拟Baseline的简单处理）
            self.aspp = nn.Sequential(
                nn.Conv2d(768, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

        # 3. 底层特征与融合模块
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(96, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        if self.use_guided_fusion:
            self.gaussian = GaussianFilter(96)
            self.fam = FeatureAlignmentModule(256, 256)
            self.guided = GuidedFilter(r=4)

        # 4. 边缘分支
        decoder_in_channels = 256 + 48  # ASPP(256) + Low(48)
        if self.use_edge_branch:
            self.edge_branch = nn.Sequential(
                nn.Conv2d(decoder_in_channels, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 1, 1, bias=False),
                nn.Sigmoid()
            )
            decoder_in_channels += 1  # 如果开启边缘分支，解码器输入通道 +1 (变成305)

        # 5. 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(decoder_in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 6. 频域增强与输出头
        if self.use_fft:
            self.hf_enhancer = HighFreqEnhancer(enhance_factor=0.5)
        self.seg_head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        original_size = x.shape[2:]
        features = self.backbone(x)
        s1, s2, s3, s4 = features

        if s1.dim() == 4 and s1.shape[-1] == 96:
            s1 = s1.permute(0, 3, 1, 2).contiguous()
        if s4.dim() == 4 and s4.shape[-1] == 768:
            s4 = s4.permute(0, 3, 1, 2).contiguous()

        # ASPP 阶段
        aspp_feat = self.aspp(s4)

        # 融合阶段
        if self.use_guided_fusion:
            aspp_feat_aligned = self.fam(aspp_feat, s1.shape[2:])
            s1_enhanced = self.gaussian(s1)
            aspp_feat_refined = self.guided(aspp_feat_aligned, s1_enhanced)
        else:
            # 退化为普通的双线性插值上采样，不使用底层特征平滑
            aspp_feat_refined = F.interpolate(aspp_feat, size=s1.shape[2:], mode='bilinear', align_corners=False)
            s1_enhanced = s1

        # 底层特征降维
        low_feat = self.low_level_proj(s1_enhanced)
        fused = torch.cat([aspp_feat_refined, low_feat], dim=1)

        # 边缘分支阶段
        edge_map = None
        if self.use_edge_branch:
            edge_map = self.edge_branch(fused)
            final_feat = torch.cat([fused, edge_map], dim=1)
        else:
            final_feat = fused  # 不拼接边缘图

        # 解码与输出阶段
        out = self.decoder(final_feat)

        if self.use_fft:
            out = self.hf_enhancer(out)

        out = self.seg_head(out)
        out = F.interpolate(out, size=original_size, mode='bilinear', align_corners=False)

        # 如果关闭了边缘分支，返回 out 和 None，tr_pots.py 会自动适应！
        return out, edge_map


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinDeepLabV3PlusEnhanced(num_classes=6, input_res=256).to(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    mask, edge = model(dummy_input)
    print("\n✅ 测试通过!")
    print(f"Mask: {mask.shape}, Edge: {edge.shape}")
