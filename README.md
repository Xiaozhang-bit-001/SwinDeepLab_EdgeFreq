# FGH-DeepLab: Remote Sensing Image Semantic Segmentation with Swin Transformer and Frequency Domain Edge Enhancement

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **FGH-DeepLab**, a hybrid semantic segmentation network for high-resolution remote sensing imagery. Building upon the Swin Transformer backbone with enhanced atrous spatial pyramid pooling, spatially aligned guided decoding, and edge-frequency joint refinement, the proposed method effectively alleviates internal semantic fracture of large-scale ground objects and edge blurring of artificial facilities, achieving state-of-the-art performance on multiple public remote sensing benchmarks.

> 📄 **Paper**: *Combining Swin Transformer with Multi-Dimensional Feature Enhancement for Remote Sensing Image Semantic Segmentation* (under review)
> 👨‍💻 **Author**: Junming Zhang
> 🏫 **Affiliation**: School of Software, Liaoning Technical University, Huludao, China

---

## 📊 Performance Highlights

| Dataset   | mIoU (%) | mF1 (%) | OA (%) |
|-----------|----------|---------|--------|
| ISPRS Potsdam   | **90.55**| **95.01** | 94.72  |
| ISPRS Vaihingen | **86.36**| **92.56** | 92.18  |
| LoveDA          | **85.27**| **92.01** | 91.45  |

FGH-DeepLab achieves state-of-the-art segmentation performance on ISPRS Potsdam, ISPRS Vaihingen and LoveDA benchmarks, consistently outperforming DC-Swin, LOGCAN++, SwinUNet, DeepLabV3+, UMFormer, CMTFNet, DANet and BiCoR-Seg.

---

## 🧠 Key Innovations

1. **Multi-Scale Directional Feature Encoder**
   Adopts Swin-Tiny hierarchical shifted window backbone to maintain global spatial continuity and relieve internal semantic fracture of large-scale ground objects. An enhanced ASPP module with parallel dilated convolutions and 0°/45°/90°/135° directional filters is embedded to extract geometric texture of artificial facilities, combined with CBAM dual attention to suppress background noise.

2. **Spatial Alignment Edge-Preserving Decoder**
   Introduces Feature Alignment Module (FAM) with offset convolution to adaptively calculate local spatial deformation field and correct pixel-level semantic offset during upsampling. Gaussian-smoothed low-level features serve as guidance map, and guided filtering injects bottom edge details into high-level features without loss, avoiding jagged noise from direct concatenation.

3. **Edge-Frequency Joint Refinement Module**
   Designs a decoupled explicit morphological edge branch with BCE boundary loss to impose geometric prior constraints. An FFT-based high-frequency residual enhancer is proposed to compensate edge blurring caused by continuous convolution from the physical signal level, which significantly improves boundary accuracy of tiny targets such as vehicles.

4. **Multi-Task Joint Loss Function**
   Combines cross-entropy loss, Dice loss and edge loss with weighted coefficients, balancing overall segmentation accuracy and fine boundary fidelity simultaneously.

---

## 📁 Project Structure

---

## ⚙️ Installation

### Requirements

- Python 3.9+
- PyTorch 2.5.1
- CUDA 12.4 (recommended for GPU acceleration)
- Key dependencies:
  - `einops` for tensor rearrangement
  - `numpy` for numerical computation
  - `opencv-python` for image processing
  - `matplotlib` for result visualization
  - `scikit-learn` for metric calculation
  - `tqdm` for progress bar
  - `tensorboardX` for training logging

### Setup Steps

```bash
# 1. Clone the repository
git clone https://github.com/Xiaozhang-bit-001/SwinDeepLab_EdgeFreq.git
cd SwinDeepLab_EdgeFreq

# 2. Create conda environment (optional but recommended)
conda create -n swindeeplab python=3.9
conda activate swindeeplab

# 3. Install PyTorch with CUDA support
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

# 4. Install other dependencies
pip install einops numpy opencv-python matplotlib scikit-learn tqdm tensorboardX
