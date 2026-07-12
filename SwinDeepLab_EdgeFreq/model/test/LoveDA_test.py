import argparse
import os
# 强行关闭 Hugging Face 网络请求
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TORCH_HOME'] = '/tmp/torch_cache'
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import cv2
import json
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ================= 导入你的模型 =================
# 确保运行该脚本的路径可以正常 import 这些模块
PROJECT_ROOT = r'.../model2' # 根据你的实际代码路径调整
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from DeepLabREG import SwinDeepLabV3PlusEnhanced as DeepLabReg
# ===============================================

# ======================== 1. 核心配置 ========================
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

DATASET_NAME = 'LoveDA_256'
NUM_CLASSES = 7
IN_CHANNELS = 3
IMG_SIZE = 256
BATCH_SIZE = 16
VIS_NUM = 300  # 可视化数量，建议不用太多，否则生成很慢
DEBUG = False

# ================== ⚠️ 路径配置 (请核对) ==================
# npz 数据所在的文件夹
ROOT_PATH = r'.../Dataset/LoveDA/npz_data_all'
# 测试集列表（包含测试样本的文件名）
TEST_LIST = r'.../Dataset/LoveDA/lists_txt/test.txt'

# 归一化统计文件
STATS_PATH = r".../Dataset/LoveDA/rgb_data_stats_improved.npz"

# 你的最优权重路径 (请根据你终端的实际打印路径进行修改)
MODEL_WEIGHT_PATH = r".../ComparedModels/DeepLabReg_LoveDA_new_networks/DeepLabReg_LoveDA_256_256_Baseline/iter60k_epo100_bs32_lr0.01_s1234/best_model.pth"

# 结果保存路径
OUTPUT_DIR = r'.../TestResults/DeepLabReg_LoveDA'

# 模型消融实验参数 (必须与你训练时保持绝对一致)
USE_ASPP_ENHANCED = False
USE_GUIDED_FUSION = False
USE_EDGE_BRANCH = False
USE_FFT = False
# ==========================================================

# 类别配置 (0~6)
CLASS_NAMES = ['Background', 'Building', 'Road', 'Water', 'Barren', 'Forest', 'Agriculture']
VALID_CLASSES = [0, 1, 2, 3, 4, 5, 6]
CLASS_COLORS = [
    (0, 0, 0),        # 0: 背景 (黑色)
    (255, 0, 0),      # 1: 建筑 (红色)
    (255, 255, 0),    # 2: 道路 (青/黄色)
    (0, 0, 255),      # 3: 水体 (蓝色)
    (159, 129, 183),  # 4: 裸地 (紫色)
    (0, 255, 0),      # 5: 森林 (绿色)
    (255, 195, 128)   # 6: 农业 (橙色)
]

# ======================== 2. 数据加载 ========================
class LoveDATestDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, list_path, img_size=256, transform=None):
        self.root_path = root_path
        self.img_size = img_size
        self.transform = transform

        # 读取 test.txt 列表
        with open(list_path, 'r') as f:
            # list 文件里的内容可能是绝对路径，我们只提取文件名
            self.file_names = [os.path.basename(line.strip()) for line in f.readlines() if line.strip()]

        print(f"📁 成功从列表加载 {len(self.file_names)} 个测试样本")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        npz_path = os.path.join(self.root_path, file_name)

        data = np.load(npz_path)
        img = data['image']  # (3, 256, 256)
        mask = data['label'] # (256, 256)

        # ================= 🚨 LoveDA 标签映射逻辑 =================
        new_label = np.full_like(mask, 255)
        valid_mask = (mask >= 1) & (mask <= 7)
        new_label[valid_mask] = mask[valid_mask] - 1
        mask = new_label
        # =========================================================

        if self.transform is not None:
            img_trans = img.transpose(1, 2, 0)
            img_trans = self.transform(img_trans)
            img = img_trans.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask, file_name.replace('.npz', '')

# ======================== 3. 评价指标计算 ========================
class SegMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, targets):
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        for pred, target in zip(preds, targets):
            # 过滤掉 255 忽略区
            mask = (target >= 0) & (target < self.num_classes)
            pred_valid = pred[mask]
            target_valid = target[mask]

            if len(pred_valid) > 0 and len(target_valid) > 0:
                self.confusion_matrix += confusion_matrix(
                    target_valid, pred_valid,
                    labels=list(range(self.num_classes))
                )

    def compute(self):
        cm = self.confusion_matrix.copy()
        results = {}

        iou, precision, recall, f1 = [], [], [], []
        for cls in range(self.num_classes):
            tp = cm[cls, cls]
            fp = cm[:, cls].sum() - tp
            fn = cm[cls, :].sum() - tp

            iou_cls = tp / (tp + fp + fn + 1e-8)
            precision_cls = tp / (tp + fp + 1e-8)
            recall_cls = tp / (tp + fn + 1e-8)
            f1_cls = 2 * precision_cls * recall_cls / (precision_cls + recall_cls + 1e-8)

            iou.append(iou_cls)
            precision.append(precision_cls)
            recall.append(recall_cls)
            f1.append(f1_cls)

        results['mIoU'] = float(np.mean(iou))
        results['mPrecision'] = float(np.mean(precision))
        results['mRecall'] = float(np.mean(recall))
        results['mF1'] = float(np.mean(f1))

        total_tp = np.diag(cm).sum()
        total_samples = cm.sum()
        results['OA'] = float(total_tp / (total_samples + 1e-8))

        results['per_class_iou'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(iou)}
        results['per_class_precision'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(precision)}
        results['per_class_recall'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(recall)}
        results['per_class_f1'] = {CLASS_NAMES[i]: float(v) for i, v in enumerate(f1)}

        return results

# ======================== 4. 可视化函数 ========================
def vis_result(img, mask, pred, file_name, save_path, mean, std):
    img_np = img.cpu().numpy().transpose(1, 2, 0)
    img_np = img_np * std + mean
    if mean.max() > 1:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    def mask2color(mask_data):
        if isinstance(mask_data, torch.Tensor):
            mask_data = mask_data.cpu().numpy()
        h, w = mask_data.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)
        for cls in range(NUM_CLASSES):
            color_img[mask_data == cls] = CLASS_COLORS[cls]
        return color_img

    mask_color = mask2color(mask)
    pred_color = mask2color(pred)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(mask_color)
    axes[1].set_title('Ground Truth', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(pred_color)
    axes[2].set_title('DeepLabReg Prediction', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(file_name, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ======================== 5. 主函数 ========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUTPUT_DIR, 'visualization')
    pred_map_dir = os.path.join(OUTPUT_DIR, 'pred_maps')
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(pred_map_dir, exist_ok=True)
    print(f"📌 测试结果保存路径: {OUTPUT_DIR}")

    # 读取归一化参数
    if os.path.exists(STATS_PATH):
        stats = np.load(STATS_PATH)
        train_mean = stats["mean"]
        train_std = stats["std"]
        print(f"📊 从npz加载归一化参数 - 均值: {train_mean.round(4)}, 标准差: {train_std.round(4)}")
    else:
        raise FileNotFoundError(f"找不到归一化文件: {STATS_PATH}")

    def transform(img):
        img = img.astype(np.float32)
        if train_mean.max() > 1:
            img = (img - train_mean) / train_std
        else:
            img = img / 255.0
            img = (img - train_mean) / train_std
        return img

    test_dataset = LoveDATestDataset(
        root_path=ROOT_PATH,
        list_path=TEST_LIST,
        img_size=IMG_SIZE,
        transform=transform
    )
    
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print("\n🔧 加载 DeepLabReg 模型...")
    model = DeepLabReg(
        num_classes=NUM_CLASSES,
        input_res=IMG_SIZE,
        use_aspp_enhanced=USE_ASPP_ENHANCED,
        use_guided_fusion=USE_GUIDED_FUSION,
        use_edge_branch=USE_EDGE_BRANCH,
        use_fft=USE_FFT
    ).to(DEVICE)

    # 加载权重
    checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 消除 DDP 的 'module.' 前缀
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)
    print(f"✅ 模型权重加载成功！")
    model.eval()

    metrics = SegMetrics(NUM_CLASSES)
    vis_count = 0
    
    print("\n🚀 开始测试模型...")
    with torch.no_grad():
        for batch_idx, (imgs, masks, file_names) in enumerate(tqdm(test_loader, desc='Testing')):
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            # 模型前向传播，兼容多输出情况
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            
            preds = torch.argmax(outputs, dim=1)
            metrics.update(preds, masks)

            # 可视化输出
            if vis_count < VIS_NUM:
                for i in range(len(file_names)):
                    if vis_count >= VIS_NUM: break
                    # 保存对比图
                    vis_save_path = os.path.join(vis_dir, f"{file_names[i]}.png")
                    vis_result(imgs[i], masks[i], preds[i], file_names[i], vis_save_path, train_mean, train_std)
                    
                    # 仅保存预测结果掩码图
                    pred = preds[i].cpu().numpy()
                    pred_color = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                    for cls in range(NUM_CLASSES):
                        pred_color[pred == cls] = CLASS_COLORS[cls]
                    
                    pred_save_path = os.path.join(pred_map_dir, f"{file_names[i]}_pred.png")
                    # 转为 BGR 供 OpenCV 保存
                    cv2.imwrite(pred_save_path, cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
                    vis_count += 1

    results = metrics.compute()
    print("\n" + "=" * 60)
    print("📊 DeepLabReg 模型测试结果汇总")
    print("=" * 60)
    print(f"总体精度 (OA) : {results['OA']:.4f}")
    print(f"平均 IoU (mIoU) : {results['mIoU']:.4f}")
    
    print("\n📋 每类详细指标:")
    for cls_name in CLASS_NAMES:
        print(f"  {cls_name.ljust(15)} IoU: {results['per_class_iou'][cls_name]:.4f}  |  F1: {results['per_class_f1'][cls_name]:.4f}")

    metrics_path = os.path.join(OUTPUT_DIR, 'test_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n✅ 测试完全结束！所有的测试指标已经保存在: {metrics_path}")

if __name__ == "__main__":
    main()
