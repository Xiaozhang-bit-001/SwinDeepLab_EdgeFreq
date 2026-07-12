import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """通道注意力模块"""

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 全局平均池化和全局最大池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)

        # 计算平均池化特征和最大池化特征
        avg_out = self.mlp(self.avg_pool(x))  # 形状: (batch_size, channels, 1, 1)
        max_out = self.mlp(self.max_pool(x))  # 形状: (batch_size, channels, 1, 1)

        # 特征相加并经过sigmoid激活
        attention = self.sigmoid(avg_out + max_out)  # 形状: (batch_size, channels, 1, 1)

        # 与输入特征相乘（通道维度上的注意力加权）
        return x * attention


class SpatialAttention(nn.Module):
    """空间注意力模块"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 确保卷积核大小为奇数
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"

        # 卷积层，用于融合通道信息
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)

        # 在通道维度上计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 形状: (batch_size, 1, height, width)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 形状: (batch_size, 1, height, width)

        # 拼接两个特征图
        attention = torch.cat([avg_out, max_out], dim=1)  # 形状: (batch_size, 2, height, width)

        # 通过卷积层融合特征并激活
        attention = self.sigmoid(self.conv(attention))  # 形状: (batch_size, 1, height, width)

        # 与输入特征相乘（空间维度上的注意力加权）
        return x * attention


class CBAM(nn.Module):
    """CBAM注意力模块：先通道注意力，后空间注意力"""

    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)  # 先应用通道注意力
        x = self.spatial_attention(x)  # 再应用空间注意力
        return x


# 测试CBAM模块
def test_cbam():
    # 创建随机输入张量 (batch_size=2, channels=64, height=32, width=32)
    x = torch.randn(2, 64, 32, 32)

    # 创建CBAM模块
    cbam = CBAM(in_channels=64)

    # 前向传播
    output = cbam(x)

    # 检查输出形状是否与输入一致
    assert x.shape == output.shape, f"输出形状错误: 输入 {x.shape}, 输出 {output.shape}"
    print(f"测试通过! 输入形状: {x.shape}, 输出形状: {output.shape}")

    # 打印模块结构
    print("\nCBAM模块结构:")
    print(cbam)


# 示例：在卷积块中使用CBAM
class ConvBlockWithCBAM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlockWithCBAM, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.cbam = CBAM(out_channels)  # 在卷积后添加CBAM注意力

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.cbam(x)  # 应用注意力机制
        return x


# if __name__ == "__main__":
#     test_cbam()
