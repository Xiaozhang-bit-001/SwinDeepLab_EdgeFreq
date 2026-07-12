# SwinDeepLab-EdgeFreq: Remote Sensing Image Semantic Segmentation with Swin Transformer and Frequency Domain Edge Enhancement

[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.9-3776AB.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official implementation of **SwinDeepLab-EdgeFreq**, a hybrid semantic segmentation network for high-resolution remote sensing imagery. Building upon the Swin Transformer backbone with enhanced atrous spatial pyramid pooling, spatially aligned guided decoding, and edge-frequency joint refinement, the proposed method effectively alleviates internal semantic fracture of large-scale ground objects and edge blurring of artificial facilities, achieving state-of-the-art performance on multiple public remote sensing benchmarks.

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

SwinDeepLab-EdgeFreq achieves state-of-the-art segmentation performance on ISPRS Potsdam, ISPRS Vaihingen and LoveDA benchmarks, consistently outperforming DC-Swin, LOGCAN++, SwinUNet, DeepLabV3+, UMFormer, CMTFNet, DANet and BiCoR-Seg.

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
