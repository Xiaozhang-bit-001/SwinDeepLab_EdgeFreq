import os
import numpy as np
import tifffile
from tqdm import tqdm

# -------------------------- 1. 配置参数（核心修改：新增测试集比例） --------------------------
RAW_DATA_ROOT = r"...\datasets\Potsdam"
IMAGE_DIR = os.path.join(RAW_DATA_ROOT, ".../RGB")
LABEL_DIR = os.path.join(RAW_DATA_ROOT, "...")

IMAGE_SUFFIX = "_RGB"
OUTPUT_NPZ_DIR = os.path.join(RAW_DATA_ROOT, "npz_data_RGB_improved")
OUTPUT_LIST_DIR = os.path.join(RAW_DATA_ROOT, "lists_txt_RGB_improved")
PATCH_SIZE = 256  # 若仍内存不足，可改为128
OVERLAP = 64     # 若仍内存不足，可改为32
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
SAVE_STATS = True
MIN_VALID_PIXELS = 10
KNOWN_INVALID_VALUES = {-1, 127, 255}
RANDOM_SEED = 42  # 新增：固定随机种子，确保划分结果可复现


# -------------------------- 2. 标签映射（保持不变） --------------------------
RGB_LABEL_MAPPING = {
    (0, 255, 255): 0,  # 低灌木（青色）
    (255, 255, 255): 1,  # 不透水面（白色）
    (0, 0, 255): 2,  # 建筑（纯蓝色）
    (255, 0, 0): 3,  # 背景（纯红色）
    (0, 255, 0): 4,  # 植被（纯绿色）
    (255, 255, 0): 5  # 车辆（纯黄色）
}
INVALID_LABEL = 255
CLASS_NAMES = ["低灌木-青色", "硬化路-白色", "建筑-蓝色", "裸土-红色", "大树-绿色", "车辆-黄色"]


# -------------------------- 3. 工具函数（保持不变） --------------------------
def read_rgb_label(label_path):
    rgb_label = tifffile.imread(label_path)
    if rgb_label.ndim == 3:
        if rgb_label.shape[-1] != 3 and rgb_label.shape[0] == 3:
            rgb_label = np.transpose(rgb_label, (1, 2, 0))
    elif rgb_label.ndim == 2:
        rgb_label = np.stack([rgb_label] * 3, axis=-1)
    else:
        raise ValueError(f"标签维度错误：{rgb_label.ndim}D，路径：{label_path}")

    h, w = rgb_label.shape[:2]
    class_label = np.full((h, w), INVALID_LABEL, dtype=np.uint8)
    for (r, g, b), idx in RGB_LABEL_MAPPING.items():
        mask = np.all(rgb_label == [r, g, b], axis=-1)
        class_label[mask] = idx

    invalid_ratio = np.sum(class_label == INVALID_LABEL) / (h * w) * 100
    if invalid_ratio > 10:
        print(f"警告：{os.path.basename(label_path)} 无效像素占比{invalid_ratio:.2f}%")
    return class_label


def read_rgb_image(img_path):
    img = tifffile.imread(img_path)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"RGB图像格式错误：{img.shape}，路径：{img_path}")
    img = np.transpose(img, (2, 0, 1)).astype(np.float32)
    if np.any(img < 0) or np.any(img > 255):
        print(f"注意：{os.path.basename(img_path)} 像素值超出0-255范围，建议归一化")
    return img


def crop_to_patches(data, patch_size, overlap):
    patches = []
    c, h, w = data.shape
    stride = patch_size - overlap
    row_steps = max(1, (h - patch_size + stride) // stride)
    col_steps = max(1, (w - patch_size + stride) // stride)
    if row_steps * stride + patch_size < h:
        row_steps += 1
    if col_steps * stride + patch_size < w:
        col_steps += 1

    for i in range(row_steps):
        for j in range(col_steps):
            h_start = min(i * stride, h - patch_size)
            w_start = min(j * stride, w - patch_size)
            patch = data[:, h_start:h_start + patch_size, w_start:w_start + patch_size]
            patches.append(patch)
    return patches


def post_process_label(label_patch):
    for val in KNOWN_INVALID_VALUES:
        label_patch[label_patch == val] = INVALID_LABEL
    valid_mask = (label_patch >= 0) & (label_patch < len(RGB_LABEL_MAPPING))
    label_patch[~valid_mask] = INVALID_LABEL
    return label_patch


def calculate_class_distribution(npz_dir):
    class_counts = np.zeros(len(RGB_LABEL_MAPPING), dtype=np.uint64)
    npz_paths = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]
    for npz_path in tqdm(npz_paths, desc="分析RGB数据集类别分布"):
        data = np.load(npz_path)
        label = data["label"]
        for cls in range(len(RGB_LABEL_MAPPING)):
            class_counts[cls] += np.sum(label == cls)
        # 核心优化：读取后立即释放npz数据内存
        del data, label

    total = np.sum(class_counts)
    print("\n" + "=" * 60)
    print("RGB数据集类别分布统计：")
    for i, (name, count) in enumerate(zip(CLASS_NAMES, class_counts)):
        ratio = count / total * 100 if total > 0 else 0
        print(f"{name}({i})：{count}像素，占比{ratio:.2f}%")
    print("=" * 60 + "\n")

    if total > 0 and class_counts[5] / total < 0.01:
        print("警告：车辆类别占比极低，建议：")
        print("1. 训练时增加车辆patch采样权重；2. 使用Focal Loss；3. 车辆样本增强\n")
    return class_counts


def calculate_data_stats(npz_dir):
    npz_paths = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]
    if not npz_paths:
        raise ValueError("无RGB NPZ文件，无法计算统计信息！")

    channel_means = np.zeros(3, dtype=np.float64)
    channel_vars = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for npz_path in tqdm(npz_paths, desc="计算RGB数据统计（均值/标准差）"):
        data = np.load(npz_path)
        img = data["image"]
        pixels = img.size // 3
        total_pixels += pixels

        for c in range(3):
            channel_data = img[c].ravel()
            channel_means[c] += np.sum(channel_data)
            channel_vars[c] += np.sum(channel_data **2 - 2 * channel_data * channel_means[c] / total_pixels)
        # 核心优化：读取后立即释放npz数据内存
        del data, img, channel_data

    channel_means /= total_pixels
    channel_vars = (channel_vars / total_pixels) + channel_means** 2
    channel_stds = np.sqrt(channel_vars)

    stats_path = os.path.join(os.path.dirname(npz_dir), "rgb_data_stats_improved.npz")
    np.savez(stats_path, mean=channel_means, std=channel_stds)
    print(f"\nRGB数据统计保存至：{stats_path}")
    print(f"R通道均值：{channel_means[0]:.4f}，标准差：{channel_stds[0]:.4f}")
    print(f"G通道均值：{channel_means[1]:.4f}，标准差：{channel_stds[1]:.4f}")
    print(f"B通道均值：{channel_means[2]:.4f}，标准差：{channel_stds[2]:.4f}\n")
    return channel_means, channel_stds


# -------------------------- 4. 核心逻辑（核心优化：内存管理） --------------------------
def main():
    os.makedirs(OUTPUT_NPZ_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LIST_DIR, exist_ok=True)
    npz_names = []
    invalid_samples = []
    class_patch_counts = np.zeros(len(RGB_LABEL_MAPPING), dtype=int)

    img_files = [f for f in os.listdir(IMAGE_DIR)
                 if f.endswith(".tif") and
                 not f.endswith(".tfw") and
                 IMAGE_SUFFIX in f]

    if not img_files:
        raise ValueError(f"RGB图像目录{IMAGE_DIR}中无有效文件！请检查：1. 路径是否正确；2. 文件名是否含{IMAGE_SUFFIX}")

    for img_name in tqdm(img_files, desc="处理RGB图像-标签对"):
        if not img_name.endswith(f"{IMAGE_SUFFIX}.tif"):
            continue
        core_name = img_name.replace(f"{IMAGE_SUFFIX}.tif", "")
        img_path = os.path.join(IMAGE_DIR, img_name)
        label_path = os.path.join(LABEL_DIR, f"{core_name}_label.tif")

        if not os.path.exists(label_path):
            print(f"警告：标签不存在，跳过 → {os.path.basename(label_path)}")
            continue

        try:
            # 读取图像和标签
            img_data = read_rgb_image(img_path)
            label_data = read_rgb_label(label_path)[np.newaxis, :, :]

            # 生成补丁
            img_patches = crop_to_patches(img_data, PATCH_SIZE, OVERLAP)
            label_patches = crop_to_patches(label_data, PATCH_SIZE, OVERLAP)

            # 处理单个补丁并保存
            for idx, (img_patch, label_patch) in enumerate(zip(img_patches, label_patches)):
                clean_label = post_process_label(np.squeeze(label_patch))
                valid_pixels = np.sum(clean_label != INVALID_LABEL)
                has_vehicle = np.any(clean_label == 5)

                if valid_pixels < MIN_VALID_PIXELS and not has_vehicle:
                    continue
                if has_vehicle and valid_pixels < MIN_VALID_PIXELS:
                    print(f"保留含车辆的低有效像素patch：{core_name}_RGB_patch{idx}")

                # 更新类别patch计数
                for cls in range(len(RGB_LABEL_MAPPING)):
                    if np.any(clean_label == cls):
                        class_patch_counts[cls] += 1

                # 保存NPZ并记录文件名
                npz_name = f"{core_name}_RGB_patch{idx}.npz"
                npz_path = os.path.join(OUTPUT_NPZ_DIR, npz_name)
                np.savez_compressed(npz_path, image=img_patch, label=clean_label)
                npz_names.append(npz_name)

                # 核心优化：处理完单个补丁后，立即释放该补丁内存
                del img_patch, label_patch, clean_label

            # 核心优化：处理完单张图像的所有补丁后，立即释放整图数据内存
            del img_data, label_data, img_patches, label_patches

        except Exception as e:
            err_msg = f"处理{img_name}出错：{str(e)}"
            print(f"⚠️ {err_msg}，跳过")
            invalid_samples.append(img_name)
            # 异常时也释放内存，避免内存泄漏
            if 'img_data' in locals():
                del img_data
            if 'label_data' in locals():
                del label_data
            continue

    if not npz_names:
        raise ValueError("未生成任何RGB NPZ文件！请检查：1. 图像/标签路径；2. 标签映射是否正确")

    print("\nRGB数据集各类别patch数量：")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name}：{class_patch_counts[i]}个patch")

    # 数据集划分（新增固定随机种子）
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(npz_names)
    total = len(npz_names)
    test_num = int(total * TEST_RATIO)
    val_num = int(total * VAL_RATIO)
    train_num = total - test_num - val_num

    test_list = npz_names[:test_num]
    val_list = npz_names[test_num:test_num+val_num]
    train_list = npz_names[test_num+val_num:]

    # 保存TXT文件
    train_txt_path = os.path.join(OUTPUT_LIST_DIR, "train.txt")
    val_txt_path = os.path.join(OUTPUT_LIST_DIR, "val.txt")
    test_txt_path = os.path.join(OUTPUT_LIST_DIR, "test.txt")

    with open(train_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.join(OUTPUT_NPZ_DIR, name) for name in train_list]))
    with open(val_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.join(OUTPUT_NPZ_DIR, name) for name in val_list]))
    with open(test_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join([os.path.join(OUTPUT_NPZ_DIR, name) for name in test_list]))

    # 计算统计信息（优化后内存占用降低）
    if SAVE_STATS:
        calculate_data_stats(OUTPUT_NPZ_DIR)
    calculate_class_distribution(OUTPUT_NPZ_DIR)

    # 输出汇总信息
    print("\n" + "=" * 80)
    print("✅ RGB数据集预处理完成！")
    print(f"总RGB NPZ文件数：{total}")
    print(f"训练集（70%）：{len(train_list)}个 → 路径：{train_txt_path}")
    print(f"验证集（20%）：{len(val_list)}个 → 路径：{val_txt_path}")
    print(f"测试集（10%）：{len(test_list)}个 → 路径：{test_txt_path}")
    if invalid_samples:
        print(f"处理失败的样本：{len(invalid_samples)}个 → 示例：{invalid_samples[:3]}...")
    print("=" * 80)


if __name__ == "__main__":
    main()
