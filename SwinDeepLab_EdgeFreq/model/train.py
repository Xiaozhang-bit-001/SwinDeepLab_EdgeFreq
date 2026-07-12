import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

# 强制指定使用国内镜像，且禁用验证，这往往能避开 401 报错
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# 导入自定义模型（你的基础模型）
from DeepLabREG import SwinDeepLabV3PlusEnhanced as DeepLabReg

# =====================================================

from tr_new2 import trainer_synapse

parser = argparse.ArgumentParser()
# 基础配置（所有模型共用，确保对比公平）
parser.add_argument('--root_path', type=str,
                    default=r'.../dataset/LoveDA/npz_data_RGB_improved',
                    help='数据根目录')
parser.add_argument('--dataset', type=str, default='遥感图像语义分割对比实验', help='实验名称')
parser.add_argument('--list_dir', type=str,
                    default=r'.../dataset/LoveDA/lists_txt_RGB_improved',
                    help='数据列表目录')
parser.add_argument('--num_classes', type=int, default=6, help='输出类别数')
parser.add_argument('--max_iterations', type=int, default=60000, help='最大迭代数')
parser.add_argument('--resume_path', type=str,
                    default=r'',
                    help='指定要恢复的最优权重文件路径')
parser.add_argument('--max_epochs', type=int, default=150, help='最大训练轮数')
parser.add_argument('--batch_size', type=int, default=8, help='单卡batch size（命令行指定时生效）')
parser.add_argument('--n_gpu', type=int, default=1, help='GPU数量（固定单卡）')
parser.add_argument('--deterministic', type=int, default=1, help='是否启用确定性训练')
parser.add_argument('--base_lr', type=float, default=0.01, help='学习率（所有模型统一）')
parser.add_argument('--img_size', type=int, default=256, help='输入图像尺寸（所有模型统一）')
parser.add_argument('--seed', type=int, default=1234, help='随机种子（确保可复现）')
parser.add_argument('--att-type', type=str, choices=['BAM', 'CBAM'], default=None, help='注意力类型（特定模型用）')
# ================= 消融实验控制开关 =================
parser.add_argument('--use_aspp_enhanced', action='store_true', help='开启增强ASPP')
parser.add_argument('--use_guided_fusion', action='store_true', help='开启引导滤波融合')
parser.add_argument('--use_edge_branch', action='store_true', help='开启边缘监督分支')
parser.add_argument('--use_fft', action='store_true', help='开启频域高频增强')
# ====================================================
parser.add_argument('--dataset_name', type=str, default='LoveDA_256',
                    choices=['Vai_256', 'Pots_256', 'LoveDA_256'], help='选择数据集')

# 新增：模型选择参数（核心改造）
parser.add_argument('--model_name', type=str,
                    default='DeepLabReg',help='your model name')

# 数据统计文件路径
parser.add_argument('--data_stats_path', type=str,
                    default=r"../dataset/LoveDA/rgb_data_stats_improved.npz",
                    help='数据均值/标准差文件路径')

args = parser.parse_args()

if __name__ == "__main__":
    # 确定性训练配置（所有模型保持一致）
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    # 固定随机种子（确保实验可复现）
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # 数据集配置（统一路径和参数，所有模型共用）
    args.img_size = 256  # same size
    args.batch_size = 8  # same batch size
    dataset_name = args.dataset_name  # your dataset_name
    dataset_config = {
        'Vai_256': {
            'root_path': r'../Dataset/Vaihingen/npz_data_RGB_improved',
            'list_dir': r'../Dataset/Vaihingen/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3  # Vaihingen RGB 3 channels
        },
        'Pots_256': {
            'root_path': r'../Dataset/Potsdam/npz_data_RGB_improved',
            'list_dir': r'../Dataset/Potsdam/lists_txt_RGB_improved',
            'num_classes': 6,
            'in_channels': 3  # Potsdam RGB 3 channels
        },
        'LoveDA_256': {
            'root_path': r'../dataset/LoveDA/npz_data_all',  # your true npz_path
            'list_dir': r'../dataset/LoveDA/lists_txt',      # your true txt_path
            'num_classes': 7,
            'in_channels': 3
        }
    }
    # 更新数据集参数（覆盖命令行输入，确保与数据集匹配）
    args.num_classes = dataset_config[dataset_name]['num_classes']
    args.root_path = dataset_config[dataset_name]['root_path']
    args.list_dir = dataset_config[dataset_name]['list_dir']
    args.in_channels = dataset_config[dataset_name]['in_channels']
    args.is_pretrain = False  # 统一不使用预训练（如需开启，改为True并确保模型支持）

    # 加载数据统计（所有模型共用同一归一化参数）
    if not os.path.exists(args.data_stats_path):
        raise FileNotFoundError(f"数据统计文件不存在：{args.data_stats_path}")
    stats = np.load(args.data_stats_path)
    train_mean = stats["mean"].tolist()
    train_std = stats["std"].tolist()
    print(f"✅ 加载数据统计完成（所有模型共用）：")
    print(f"   均值（R/G/B）：{[round(x, 4) for x in train_mean]}")
    print(f"   标准差（R/G/B）：{[round(x, 4) for x in train_std]}")

    ablation_tag = ""

    # 初始化模型（根据model_name选择，保持原逻辑）
    if args.model_name == 'DeepLabReg':
        net = DeepLabReg(
            num_classes=args.num_classes,
            input_res=256,
            use_aspp_enhanced=args.use_aspp_enhanced,
            use_guided_fusion=args.use_guided_fusion,
            use_edge_branch=args.use_edge_branch,
            use_fft=args.use_fft
        ).cuda()
        # 为了区分不同消融实验保存的文件夹，我们在 core_tag 里加上标记
        ablation_tag = ""
        if args.use_aspp_enhanced: ablation_tag += "_ASPP"
        if args.use_guided_fusion: ablation_tag += "_GF"
        if args.use_edge_branch: ablation_tag += "_Edge"
        if args.use_fft: ablation_tag += "_FFT"
        if ablation_tag == "": ablation_tag = "_Baseline"

    else:
        raise ValueError(f"未支持的模型: {args.model_name}")
    # =====================================================

    # -------------------------- 核心修复：快照路径构建（简洁、动态、无重复）--------------------------
    # 1. 动态模型文件夹（根据model_name自动生成，避免硬编码FCNnetworks）
    model_folder = f"{args.model_name}_{dataset_name.split('_')[0]}_new_networks"

    # 【关键修改】：在这里把 ablation_tag 拼接到 core_tag 的尾部！
    core_tag = f"{args.model_name}_{dataset_name}_{args.img_size}{ablation_tag}"

    vit_tag = ""
    if args.model_name == 'ViT_seg':
        vit_tag = f"vit{args.vit_name}_skip{args.n_skip}_patch{args.vit_patches_size}"

    iter_tag = f"iter{args.max_iterations // 1000}k"
    epoch_tag = f"epo{args.max_epochs}"
    batch_tag = f"bs{args.batch_size}"
    lr_tag = f"lr{args.base_lr}"
    seed_tag = f"s{args.seed}"
    param_tag = "_".join([iter_tag, epoch_tag, batch_tag, lr_tag, seed_tag])

    snapshot_path = os.path.join(
        r"your_path",  
        model_folder,
        core_tag,
        vit_tag,
        param_tag
    )
    snapshot_path = os.path.normpath(snapshot_path)
    # ---------------------------------------------------------------------------------------------

    print('-------------------------------------------')
    print(f"当前模型: {args.model_name}")
    print(f"快照保存路径: {snapshot_path}")
    print('-------------------------------------------')
    # create path
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path, exist_ok=True)

    # ================= 断点续训逻辑 =================
    start_epoch = 0

    # 【新增】自动检测当前快照目录下最新的权重文件
    if args.resume_path == '' and os.path.exists(snapshot_path):
        import glob

        ckpts = glob.glob(os.path.join(snapshot_path, '*epoch*.pth'))
        if not ckpts:
           
            ckpts = glob.glob(os.path.join(snapshot_path, 'best_model.pth'))

        if ckpts:
            
            ckpts.sort(key=os.path.getmtime, reverse=True)
            args.resume_path = ckpts[0]

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"🚀")
        checkpoint = torch.load(args.resume_path, map_location=torch.device('cuda'))

        print("🔧")
        dummy_input = torch.randn(2, args.in_channels, args.img_size, args.img_size).cuda()
        net.train()
        with torch.no_grad():
            _ = net(dummy_input)
        # ==========================================================

        
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            net.load_state_dict(checkpoint['model'])
            
            start_epoch = checkpoint.get('epoch', 0) + 1
        else:
            net.load_state_dict(checkpoint)
            
            start_epoch = 0

        print(f"✅")
    else:
        print("⚠️")
    # =================================================================

    # 启动训练
    trainer_synapse(
        args,
        net,
        snapshot_path,
        start_epoch=start_epoch,
        train_mean=train_mean,
        train_std=train_std
    )
