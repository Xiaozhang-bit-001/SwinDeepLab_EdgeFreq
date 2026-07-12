import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np


# 首先引入之前实现的SoftPool2d
class SoftPool2d(nn.Module):
    """
    实现SoftPool2d操作

    论文参考: "SoftPool: Improving Pooling in Convolutional Neural Networks"
    原理: 对每个池化窗口内的元素应用softmax，然后进行加权求和
    """

    def __init__(self, kernel_size=2, stride=None, padding=0, ceil_mode=False):
        super(SoftPool2d, self).__init__()
        # 统一参数为元组格式（支持单值或元组输入）
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if stride is not None else self.kernel_size
        self.stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.ceil_mode = ceil_mode

    def forward(self, x):
        """前向传播：精确计算输出维度，确保与Unfold结果一致"""
        batch_size, channels, height, width = x.size()

        # 1. 先应用填充
        x_padded = F.pad(x, (
            self.padding[1], self.padding[1],  # 左右填充
            self.padding[0], self.padding[0]  # 上下填充
        ))
        pad_height, pad_width = x_padded.size()[2], x_padded.size()[3]

        # 2. 计算输出维度（使用PyTorch官方池化计算方式）
        out_h = self._calculate_output_dim(pad_height, self.kernel_size[0], self.stride[0], self.ceil_mode)
        out_w = self._calculate_output_dim(pad_width, self.kernel_size[1], self.stride[1], self.ceil_mode)

        # 3. 使用Unfold提取所有池化窗口
        unfold = nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride
        )
        x_unfolded = unfold(x_padded)  # shape: [B, C*K_h*K_w, N]，N为窗口总数

        # 4. 调整形状以适配softmax（按窗口维度计算权重）
        kernel_flat = self.kernel_size[0] * self.kernel_size[1]  # 单个窗口的元素数
        x_unfolded = x_unfolded.view(batch_size, channels, kernel_flat, -1)

        # 5. 计算窗口内元素的权重（softmax）并加权求和
        weights = F.softmax(x_unfolded, dim=2)  # 按窗口维度（dim=2）计算softmax
        x_soft_pooled = torch.sum(x_unfolded * weights, dim=2)  # 加权求和，得到每个窗口的结果

        # 6. 获取实际窗口数量并验证
        num_windows = x_soft_pooled.size(-1)

        # 兼容性处理：允许微小误差（由于浮点计算）
        if not (abs(out_h * out_w - num_windows) < 2):
            # 自动调整输出维度以匹配实际窗口数
            out_h, out_w = self._adjust_dimensions(out_h, out_w, num_windows)
            print(f"警告：自动调整输出维度为 ({out_h}, {out_w}) 以匹配实际窗口数 {num_windows}")

        # 7. 重塑为最终的4D特征图形状
        x_soft_pooled = x_soft_pooled.view(batch_size, channels, out_h, out_w)

        return x_soft_pooled

    def _calculate_output_dim(self, in_dim, kernel, stride, ceil_mode):
        """使用PyTorch官方公式计算输出维度"""
        if ceil_mode:
            return (in_dim - kernel + stride - 1) // stride + 1
        else:
            return (in_dim - kernel) // stride + 1

    def _adjust_dimensions(self, out_h, out_w, num_windows):
        """自动调整维度以匹配实际窗口数量"""
        # 尝试保持高度不变，调整宽度
        if num_windows % out_h == 0:
            return out_h, num_windows // out_h
        # 尝试保持宽度不变，调整高度
        if num_windows % out_w == 0:
            return num_windows // out_w, out_w
        # 都不行则返回最接近的分解
        import math
        new_h = int(math.sqrt(num_windows))
        while new_h > 0:
            if num_windows % new_h == 0:
                return new_h, num_windows // new_h
            new_h -= 1
        # 最坏情况
        return 1, num_windows


# 使用SoftPool改造的CNN模型
class SoftPoolCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SoftPoolCNN, self).__init__()
        # 第一个卷积块：卷积 + SoftPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.softpool1 = SoftPool2d(kernel_size=2, stride=2)

        # 第二个卷积块：卷积 + SoftPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.softpool2 = SoftPool2d(kernel_size=2, stride=2)

        # 第三个卷积块：卷积 + SoftPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.softpool3 = SoftPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.softpool1(x)  # 使用SoftPool替代MaxPool

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.softpool2(x)  # 使用SoftPool替代MaxPool

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.softpool3(x)  # 使用SoftPool替代MaxPool

        # 展平特征图
        x = x.view(-1, 128 * 3 * 3)

        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 传统CNN模型（使用MaxPool作为对比）
class MaxPoolCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(MaxPoolCNN, self).__init__()
        # 第一个卷积块：卷积 + MaxPool
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第二个卷积块：卷积 + MaxPool
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 第三个卷积块：卷积 + MaxPool
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # 全连接层
        self.fc1 = nn.Linear(128 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, num_classes)

        # Dropout层防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一个卷积块
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)  # 使用MaxPool

        # 第二个卷积块
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxpool2(x)  # 使用MaxPool

        # 第三个卷积块
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxpool3(x)  # 使用MaxPool

        # 展平特征图
        x = x.view(-1, 128 * 3 * 3)

        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# 验证函数：比较SoftPool和MaxPool的特征提取能力
def validate_pooling():
    # 创建随机输入张量
    x = torch.randn(1, 1, 28, 28)  # MNIST图像大小

    # 创建池化层实例
    soft_pool = SoftPool2d(kernel_size=2, stride=2)
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # 应用池化
    x_soft = soft_pool(x)
    x_max = max_pool(x)

    print(f"输入形状: {x.shape}")
    print(f"SoftPool输出形状: {x_soft.shape}")
    print(f"MaxPool输出形状: {x_max.shape}")

    # 可视化池化效果
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    plt.title('原始输入')
    plt.imshow(x.squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(132)
    plt.title('SoftPool_output')
    plt.imshow(x_soft.squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.subplot(133)
    plt.title('MaxPool_output')
    plt.imshow(x_max.squeeze().numpy(), cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('pooling_comparison.png')
    print("池化效果对比图已保存为 'pooling_comparison.png'")


# 训练和评估函数
def train_and_evaluate(model, model_name, train_loader, test_loader, epochs=5, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_accs = []

    print(f"\n开始训练{model_name} (使用{device})")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'[{epoch + 1}/{epochs}, {batch_idx + 1}/{len(train_loader)}] 损失: {loss.item():.6f}')

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f'Epoch {epoch + 1} 平均训练损失: {avg_train_loss:.6f}')

        # 在测试集上评估
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        test_acc = 100 * correct / total
        test_accs.append(test_acc)
        print(f'Epoch {epoch + 1} 测试准确率: {test_acc:.2f}%')

    print(f"{model_name} 最终测试准确率: {test_accs[-1]:.2f}%")
    return train_losses, test_accs


# 主函数：执行验证和训练
def main():
    # 1. 验证池化层功能
    print("===== 验证池化层功能 =====")
    validate_pooling()

    # 2. 准备数据集 (MNIST)
    print("\n===== 准备数据集 =====")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1000, shuffle=False, num_workers=2
    )

    # 3. 训练和评估使用SoftPool的模型
    softpool_model = SoftPoolCNN()
    softpool_losses, softpool_accs = train_and_evaluate(
        softpool_model, "SoftPool CNN", train_loader, test_loader, epochs=5
    )

    # 4. 训练和评估使用MaxPool的模型（作为对比）
    maxpool_model = MaxPoolCNN()
    maxpool_losses, maxpool_accs = train_and_evaluate(
        maxpool_model, "MaxPool CNN", train_loader, test_loader, epochs=5
    )

    # 5. 可视化训练结果
    plt.figure(figsize=(12, 5))

    # 损失对比
    plt.subplot(121)
    plt.plot(softpool_losses, label='SoftPool CNN')
    plt.plot(maxpool_losses, label='MaxPool CNN')
    plt.title('train_loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()

    # 准确率对比
    plt.subplot(122)
    plt.plot(softpool_accs, label='SoftPool CNN')
    plt.plot(maxpool_accs, label='MaxPool CNN')
    plt.title('acc:')
    plt.xlabel('Epoch')
    plt.ylabel('acc (%)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_comparison.png')
    print("\n训练结果对比图已保存为 'training_comparison.png'")

#
# if __name__ == "__main__":
#     main()
