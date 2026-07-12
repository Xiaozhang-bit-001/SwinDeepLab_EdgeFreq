import os
import numpy as np
from PIL import Image
from tqdm import tqdm

# -------------------------- 1. 配置参数 --------------------------
LOVEDA_ROOT = r".../dataset/LoveDA"
OUTPUT_NPZ_DIR = os.path.join(LOVEDA_ROOT, "npz_data_all")
OUTPUT_LIST_DIR = os.path.join(LOVEDA_ROOT, "lists_txt")

PATCH_SIZE = 256
OVERLAP = 64
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SAVE_STATS = True
MIN_VALID_PIXELS = 10
INVALID_LABEL = 255  # 统一无效值为 255
RANDOM_SEED = 42


# -------------------------- 2. 工具函数 --------------------------
def read_image(img_path):
    """读取 PNG 图像并转换为 CHW 格式"""
    img = Image.open(img_path).convert('RGB')
    return np.array(img).transpose(2, 0, 1).astype(np.float32)


def read_label(label_path):
    """直接读取索引标签图"""
    # LoveDA 原生就是单通道 PNG 索引图，直接读取即可
    label = np.array(Image.open(label_path), dtype=np.uint8)
    # 根据数据集要求，0 是无效区域，统一设为 255 以便在 Loss 中 ignore
    label[label == 0] = INVALID_LABEL
    return label


def crop_to_patches(data, patch_size, overlap):
    patches = []
    # 适配图像 (3, H, W) 和标签 (H, W) 的维度
    if data.ndim == 2: data = data[np.newaxis, :, :]
    c, h, w = data.shape
    stride = patch_size - overlap

    row_steps = max(1, (h - patch_size + stride) // stride)
    col_steps = max(1, (w - patch_size + stride) // stride)

    for i in range(row_steps):
        for j in range(col_steps):
            h_start = min(i * stride, h - patch_size)
            w_start = min(j * stride, w - patch_size)
            patch = data[:, h_start:h_start + patch_size, w_start:w_start + patch_size]
            patches.append(patch)
    return patches


def calculate_data_stats(npz_dir):
    """计算数据集全局统计信息"""
    npz_paths = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]
    channel_means, channel_vars = np.zeros(3), np.zeros(3)
    total_pixels = 0
    for npz_path in tqdm(npz_paths, desc="计算统计信息"):
        data = np.load(npz_path)
        img = data["image"]
        total_pixels += img.size // 3
        for c in range(3):
            channel_means[c] += np.sum(img[c])
            channel_vars[c] += np.sum(img[c] ** 2)

    means = channel_means / total_pixels
    stds = np.sqrt(channel_vars / total_pixels - means ** 2)
    np.savez(os.path.join(os.path.dirname(npz_dir), "rgb_data_stats_improved.npz"), mean=means, std=stds)
    print(f"\n统计完成: 均值={means}, 标准差={stds}")


# -------------------------- 3. 主逻辑 --------------------------
def main():
    os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LIST_DIR, exist_ok=True)
    npz_names = []

    for root, dirs, files in os.walk(LOVEDA_ROOT):
        if "images_png" in root:
            mask_root = root.replace("images_png", "masks_png")
            if not os.path.exists(mask_root): continue

            for img_name in tqdm(files, desc=f"处理: {os.path.basename(root)}"):
                if not img_name.endswith(".png"): continue

                img_path = os.path.join(root, img_name)
                label_path = os.path.join(mask_root, img_name)

                img_data = read_image(img_path)
                label_data = read_label(label_path)

                img_patches = crop_to_patches(img_data, PATCH_SIZE, OVERLAP)
                label_patches = crop_to_patches(label_data, PATCH_SIZE, OVERLAP)

                sub_folder = os.path.basename(os.path.dirname(os.path.dirname(root)))
                for idx, (img_patch, label_patch) in enumerate(zip(img_patches, label_patches)):
                    if np.sum(label_patch != INVALID_LABEL) < MIN_VALID_PIXELS: continue

                    save_name = f"{sub_folder}_{os.path.splitext(img_name)[0]}_{idx}.npz"
                    np.savez_compressed(os.path.join(OUTPUT_NPZ_DIR, save_name),
                                        image=img_patch, label=np.squeeze(label_patch))
                    npz_names.append(save_name)

    # 划分数据集
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(npz_names)
    total = len(npz_names)
    test_num, val_num = int(total * TEST_RATIO), int(total * VAL_RATIO)

    # 保存列表
    for name, list_data in zip(["train.txt", "val.txt", "test.txt"],
                               [npz_names[test_num + val_num:], npz_names[test_num:test_num + val_num],
                                npz_names[:test_num]]):
        with open(os.path.join(OUTPUT_LIST_DIR, name), "w") as f:
            f.write("\n".join([os.path.join(OUTPUT_NPZ_DIR, item) for item in list_data]))

    if SAVE_STATS: calculate_data_stats(OUTPUT_NPZ_DIR)
    print("✅ 数据集处理完成！")


if __name__ == "__main__":
    main()
