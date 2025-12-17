# IsoGS-SLAM: 基于等势面约束的高斯泼溅实时SLAM系统

<p align="center">
  <h1 align="center">IsoGS-SLAM</h1>
  <h3 align="center">Real-Time Consistent Mesh Reconstruction via Surface-Aligned Gaussian Splatting SLAM</h3>
  <p align="center">
    <strong>基于等势面约束的高斯泼溅实时SLAM系统</strong>
  </p>
</p>

## 项目简介

**IsoGS-SLAM** 是一个创新的实时SLAM系统，它将经典的 **Metaballs（元球/隐式曲面）** 思想与现代的 **3D Gaussian Splatting** 技术相结合，实现了既保持照片级渲染质量，又能实时输出高质量、水密（Watertight）网格地图的突破性成果。

### 核心创新

- **等势面约束（Isosurface Constraint）**：将显式的3D高斯球视为隐式密度场的采样粒子，通过正则化项强迫它们排列成连续的等势面
- **实时网格提取**：在SLAM过程中实时生成几何准确的Mesh，而非传统3DGS的"云雾状"点云
- **解析法线计算**：利用微分几何直接从密度场推导出表面法线，实现精确的几何对齐

### 解决的问题

现有的3DGS SLAM系统（如SplaTAM、MonoGS）虽然渲染效果出色，但存在以下问题：

- ❌ 地图是离散的、非结构化的高斯球集合，像一团云雾
- ❌ 无法进行机器人导航（无法判断哪里是墙、哪里是路）
- ❌ 无法做物理交互（碰撞检测失效）
- ❌ 无法直接进行传统的Mesh编辑

**IsoGS-SLAM** 完美解决了这些问题，在保持实时渲染速度的同时，输出可用的几何实体。

---

## 目录

- [系统要求](#系统要求)
- [环境配置](#环境配置)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [核心算法](#核心算法)
- [项目结构](#项目结构)
- [实验与评估](#实验与评估)
- [网格提取指南](#网格提取指南)
- [基线对比](#基线对比)
- [开发路线图](#开发路线图)
- [引用](#引用)

---

## 系统要求

### 硬件配置

- **GPU**: NVIDIA RTX 3090/4090 (推荐24GB+ VRAM)
  - 本项目在 **RTX 4090 48GB** 上测试通过
- **CPU**: Intel i7/i9 (12代以上) 或 AMD Ryzen 9
  - 本项目使用 **Intel 14900K**
- **内存**: 32GB+ (推荐64GB)

### 软件环境

- **操作系统**: Ubuntu 22.04 LTS (强烈推荐)
- **CUDA**: 11.8 或 12.1
- **Python**: 3.8 或 3.9 (通过Conda管理)
- **C++**: C++14 或 C++17 (编译CUDA扩展需要)
- **ROS**: ROS2 Humble (Ubuntu 22.04)

---

## 环境配置

### 1. 安装CUDA和驱动

```bash
# 安装NVIDIA驱动 (推荐535或550版本)
sudo apt update
sudo apt install nvidia-driver-535

# 安装CUDA Toolkit 12.1
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# 配置环境变量
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### 2. 安装ROS2 Humble

```bash
# 设置locale
sudo apt update && sudo apt install locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# 添加ROS2源
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# 安装ROS2 Humble
sudo apt update
sudo apt install ros-humble-desktop -y

# 配置环境
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 3. 创建Conda环境

```bash
# 安装Miniconda (如果还没有)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# 创建项目环境
conda create -n isogs python=3.9
conda activate isogs

# 安装PyTorch (CUDA 12.1)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

---

## 安装指南

### 1. 克隆仓库

```bash
git clone <repository-url>
cd IsoGS-SLAM
```

### 2. 安装Python依赖

```bash
conda activate isogs
pip install -r requirements.txt
```

### 3. 编译CUDA扩展

本项目依赖以下CUDA扩展库：

```bash
# 编译 diff-gaussian-rasterization (3DGS核心渲染器)
cd diff-gaussian-rasterization
pip install -e .

# 编译 simple-knn (K近邻搜索)
cd ../simple-knn
pip install -e .
```

### 4. 验证安装

```bash
python scripts/test_installation.py
```

如果看到 "Installation successful!" 说明环境配置正确。

---

## 快速开始

### 运行SLAM

```bash
# 激活环境
conda activate isogs
source /opt/ros/humble/setup.bash

# 运行SLAM (使用Replica数据集)
python scripts/isogs_slam.py configs/replica/isogs_slam.py
```

### 可视化结果

```bash
# 实时可视化渲染和网格
python viz_scripts/online_recon.py configs/replica/isogs_slam.py

# 查看最终重建结果
python viz_scripts/final_recon.py configs/replica/isogs_slam.py
```

### 导出Mesh

从训练好的 checkpoint 中提取网格：

```bash
# 基本使用（自动查找最新checkpoint）
python scripts/extract_mesh_isogs.py configs/replica/splatam.py

# 指定checkpoint和输出路径
python scripts/extract_mesh_isogs.py configs/replica/splatam.py \
    --checkpoint params100.npz \
    --output mesh.ply

# 高质量网格（更小的体素）
python scripts/extract_mesh_isogs.py configs/replica/splatam.py \
    --voxel-size 0.01 \
    --iso-level 1.0
```

📖 **详细使用指南**：请参阅 [网格提取完整指南](docs/MESH_EXTRACTION_GUIDE.md)

> **注意**：`scripts/export_ply.py` 用于导出高斯点云，而 `extract_mesh_isogs.py` 用于提取网格面片。

---

## 核心算法

### 1. 密度场定义 (Density Field)

我们将场景表示为连续可微的标量场：

$$F(x) = \sum_{k \in K} \alpha_k \exp\left(-\frac{1}{2} (x - \mu_k)^T \Sigma_k^{-1} (x - \mu_k)\right)$$

其中物体的物理表面 $S$ 定义为密度场的等势面：

$$S = \{ x \in \mathbb{R}^3 \mid F(x) = \lambda \}$$

### 2. 解析法线计算 (Analytical Normals)

表面法线通过密度场梯度计算：

$$n(x) = -\frac{\nabla_x F(x)}{\| \nabla_x F(x) \|}$$

其中梯度解析解为：

$$\nabla_x F(x) = \sum_{k \in K} -\alpha_k \cdot E_k(x) \cdot \Sigma_k^{-1} (x - \mu_k)$$

### 3. 优化目标函数 (Optimization Objective)

总损失函数包含多个几何约束项：

$$L = L_{rgb} + \beta_1 L_{geo} + \beta_2 L_{reg}$$

#### 几何一致性项 $L_{geo}$

- **等势面约束**：$L_{iso} = \sum_{p \in P_{obs}} |F(p) - \lambda|^2$
- **法线对齐约束**：$L_{normal} = \sum_{p \in P_{obs}} \left( 1 - \frac{\langle \nabla_x F(p), n_{obs} \rangle}{\| \nabla_x F(p) \|} \right)$

#### 正则化项 $L_{reg}$

- **扁平化约束**：$L_{flat} = \sum_{k} \frac{s_3}{s_1 + s_2}$ (将高斯球压扁成"瓦片")
- **熵正则化**：$L_{entropy} = \sum_{k} -(\alpha_k \log \alpha_k + (1 - \alpha_k) \log (1 - \alpha_k))$

### 4. 自适应致密化策略 (Gradient-aware Densification)

基于密度梯度的分裂策略，自动在几何复杂区域（如锐利桌角）增加高斯球密度。

详细数学推导请参考项目文档 `docs/mathematics.md`。

---

## 项目结构

```
IsoGS-SLAM/
├── configs/              # 配置文件
│   ├── replica/          # Replica数据集配置
│   ├── tum/              # TUM RGB-D配置
│   └── scannet/          # ScanNet配置
├── scripts/              # 主要脚本
│   ├── isogs_slam.py     # SLAM主程序
│   ├── export_mesh.py    # 网格导出
│   └── test_installation.py
├── viz_scripts/           # 可视化脚本
├── src/                   # 源代码
│   ├── core/             # 核心算法
│   │   ├── density_field.py    # 密度场计算
│   │   ├── geometry_loss.py    # 几何损失
│   │   └── meshing.py          # 网格提取
│   ├── tracking/         # 跟踪模块
│   ├── mapping/          # 建图模块
│   └── utils/            # 工具函数
├── docs/                  # 文档
│   ├── mathematics.md    # 数学推导
│   └── implementation.md # 实现细节
└── requirements.txt      # Python依赖
```

---

## 实验与评估

### 数据集

- **Replica Dataset**: 合成数据集，用于调试算法几何准确性
- **TUM RGB-D**: 真实世界数据，测试噪声深度图下的鲁棒性
- **ScanNet**: 大规模室内场景数据集

### 评估指标

- **渲染质量**: PSNR, SSIM, LPIPS
- **几何质量**: Chamfer Distance (CD), F-score
- **实时性能**: FPS, 内存占用

### 预期结果

- **渲染质量**: 与SplaTAM持平或略低（为几何质量做出轻微牺牲）
- **几何质量**: Mesh精度显著优于原始3DGS，接近TSDF Fusion效果
- **实时性能**: 关键帧优化时30+ FPS，跟踪线程100+ FPS

---

## 网格提取指南

训练完成后，可以使用 `extract_mesh_isogs.py` 脚本从 checkpoint 中提取 3D 网格。

### 快速开始

```bash
# 使用默认参数提取网格
python scripts/extract_mesh_isogs.py configs/replica/splatam.py
```

提取的网格会保存在结果目录下的 `mesh.ply` 文件中。

### 主要参数

- `--voxel-size`: 体素大小（默认 0.02 米），更小的值产生更精细的网格
- `--iso-level`: 等值面阈值（默认 1.0），根据训练日志中的 Mean Density 调整
- `--chunk-size`: 批处理大小（默认 64），显存不足时可减小
- `--checkpoint`: 指定 checkpoint 文件，默认为最新

### 详细文档

完整的使用指南、参数说明、常见问题等，请参阅：

📖 **[网格提取完整指南](docs/MESH_EXTRACTION_GUIDE.md)**

该指南包含：
- 详细参数说明和推荐值
- 多个使用示例
- 算法原理解释
- 常见问题解决方案
- 性能参考

---

## 基线对比

本项目与以下SOTA方法进行对比：

1. **SplaTAM (CVPR 2024)**: 经典3DGS SLAM，渲染质量高但几何质量差
2. **SuGaR (CVPR 2024)**: 表面对齐高斯泼溅，但为离线处理
3. **MonoGS / GS-SLAM**: 早期3DGS SLAM基础方法
4. **Co-SLAM / NICE-SLAM**: 隐式场SLAM代表，渲染速度慢但Mesh质量好

**IsoGS-SLAM的优势**：
- ✅ 实时处理（SuGaR为离线）
- ✅ 高质量Mesh（SplaTAM无法提供）
- ✅ 快速渲染（Co-SLAM/NICE-SLAM较慢）

---

## 开发路线图

### Phase 1: 环境搭建与基线复现 (Week 1-2)
- [x] 配置Ubuntu 22.04 + ROS2 Humble
- [x] 安装CUDA和PyTorch
- [ ] 跑通SplaTAM基线代码
- [ ] 跑通SuGaR离线Demo

### Phase 2: 核心算法实现 (Week 3-8)
- [ ] 实现密度场计算模块
- [ ] 实现几何正则化Loss
- [ ] 集成到SplaTAM的Mapping循环
- [ ] 实现实时Mesh提取

### Phase 3: 优化与实验 (Week 9-12)
- [ ] 性能优化（CUDA加速）
- [ ] 在多个数据集上测试
- [ ] 生成对比实验和Demo视频

### Phase 4: 论文撰写 (Week 13-16)
- [ ] 撰写IROS 2026论文
- [ ] 准备补充材料
- [ ] 提交投稿

---

## 关键技术细节

### 实时性保证

- **关键帧优化策略**: 只在关键帧插入时进行高强度几何优化
- **并行线程**: Tracking线程只优化颜色和位姿，Mapping线程处理几何约束
- **CUDA加速**: 核心计算使用CUDA Kernel实现

### 显存优化

- **局部更新**: 只对当前视锥内的活跃高斯球进行优化
- **剪枝机制**: 定期移除对表面贡献小的高斯球
- **批量处理**: 利用4090的大显存进行批量矩阵运算

### 数值稳定性

- **Log-Space计算**: 使用`torch.logsumexp`避免下溢
- **梯度裁剪**: 防止训练过程中的数值爆炸
- **法线归一化保护**: 添加小量epsilon防止除零

---

## 常见问题

### Q: 为什么选择Ubuntu 22.04而不是24.04？

A: Ubuntu 22.04是目前学术界代码兼容性最好的版本，很多CUDA扩展库在24.04上还未完全适配。

### Q: 必须使用ROS2吗？ROS1可以吗？

A: ROS1已在2025年停止维护，且本项目主要使用Python/PyTorch，ROS2的`rclpy`接口更现代化，处理多线程和大数据传输更高效。

### Q: 显存不够怎么办？

A: 可以调整以下参数：
- 减少高斯球数量（降低`max_gaussians`）
- 增加剪枝频率
- 使用更小的图像分辨率

### Q: 如何调试几何Loss？

A: 建议先单独测试每个Loss项，确认数值范围合理后再组合。可以使用TensorBoard可视化Loss曲线。

---

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

## 引用

如果本项目对您的研究有帮助，请引用：

```bibtex
@article{isogs2026,
  title={IsoGS-SLAM: Real-Time Consistent Mesh Reconstruction via Surface-Aligned Gaussian Splatting},
  author={Your Name},
  journal={IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2026}
}
```

---

## 致谢

本项目基于以下优秀开源项目：

- [SplaTAM](https://github.com/spla-tam/SplaTAM): 3DGS SLAM基础框架
- [SuGaR](https://github.com/antao97/SuGaR): 表面对齐高斯泼溅
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting): 核心渲染技术

---

## 许可证

本项目采用 [MIT License](LICENSE) 许可证。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交Issue: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your.email@example.com

---

**最后更新**: 2025年12月
