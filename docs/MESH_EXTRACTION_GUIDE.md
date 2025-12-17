# IsoGS 网格提取指南

本指南介绍如何使用 `scripts/extract_mesh_isogs.py` 脚本从 IsoGS (Iso-Surface Gaussian Splatting) 训练好的模型中提取 3D 网格。

## 目录

- [概述](#概述)
- [安装依赖](#安装依赖)
- [基本使用](#基本使用)
- [参数说明](#参数说明)
- [使用示例](#使用示例)
- [算法原理](#算法原理)
- [常见问题](#常见问题)

## 概述

`extract_mesh_isogs.py` 脚本实现了基于密度的 Marching Cubes 网格提取算法。它从训练好的 SplaTAM checkpoint 中：

1. **加载高斯参数**：读取保存的高斯分布参数（均值、协方差、不透明度等）
2. **计算密度场**：在体素网格上计算所有高斯的密度总和
3. **提取等值面**：使用 Marching Cubes 算法提取指定密度阈值的网格
4. **清理网格**：移除孤立碎片，保留最大连通分量

## 安装依赖

### 在 Conda 环境中安装

```bash
# 激活 conda 环境
conda activate isogs

# 安装依赖（使用 pip）
pip install scikit-image trimesh

# 或者使用 conda（推荐）
conda install -c conda-forge scikit-image trimesh
```

**注意**：包名是 `scikit-image`（不是 `skimage`）！

### 完整依赖列表

脚本需要的 Python 包：
- `numpy` - 通常已包含在科学计算环境中
- `torch` - PyTorch（应该已在 isogs 环境中）
- `scikit-image` - 用于 Marching Cubes（`skimage.measure`）
- `trimesh` - 用于网格处理和清理
- `tqdm` - 进度条（通常已安装）

### 快速安装（一键）

```bash
conda activate isogs
pip install scikit-image trimesh
```

### 加速安装（使用国内镜像源）

如果下载速度慢，可以使用清华镜像源：

```bash
# 使用清华镜像源（推荐）
pip install scikit-image trimesh -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或者永久配置（推荐）
mkdir -p ~/.pip
cat > ~/.pip/pip.conf << EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
[install]
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF
```

### 使用代理

如果有代理（如端口 7890）：

```bash
# 设置代理
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890

# 然后安装
pip install scikit-image trimesh
```

如果需要使用 GPU 加速（推荐），请确保已安装支持 CUDA 的 PyTorch。

## 基本使用

### ⚠️ 重要：运行位置

**请从项目根目录运行脚本**（推荐方式）：

```bash
# 确保在项目根目录（IsoGS-SLAM/SplaTAM）
cd ~/IsoGS-SLAM/SplaTAM

# 然后运行
python scripts/extract_mesh_isogs.py configs/replica/splatam.py
```

如果当前在 `scripts/` 目录下，有两种选择：

**选项 1**：回到根目录（推荐）
```bash
cd ..  # 回到项目根目录
python scripts/extract_mesh_isogs.py configs/replica/splatam.py
```

**选项 2**：在当前目录运行
```bash
# 在 scripts/ 目录下，路径需要调整
python extract_mesh_isogs.py ../configs/replica/splatam.py
```

### 最简单的用法（自动选择 checkpoint）

```bash
python scripts/extract_mesh_isogs.py configs/replica/splatam.py
```

脚本会自动选择 checkpoint，优先级为：
1. **优先**：`params.npz`（最终结果，如果存在）
2. **否则**：自动查找所有 `params*.npz`，选择编号最大的（如 `params1250.npz` > `params100.npz`）

这会：
- 自动选择最合适的 checkpoint 文件
- 使用默认参数提取网格
- 在结果目录下生成 `mesh.ply` 文件

### 指定 checkpoint

```bash
python scripts/extract_mesh_isogs.py configs/replica/splatam.py \
    --checkpoint params100.npz
```

### 自定义输出路径

```bash
python scripts/extract_mesh_isogs.py configs/replica/splatam.py \
    --output my_mesh.ply
```

## 参数说明

### 必需参数

- `config`: 实验配置文件路径（如 `configs/replica/splatam.py`）

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--checkpoint` | str | None | 指定 checkpoint 文件（如 `params100.npz`）。如果为 None，自动使用最新的 |
| `--output` | str | None | 输出网格文件路径。如果为 None，默认在结果目录下生成 `mesh.ply` |
| `--voxel-size` | float | 0.02 | 体素大小（米）。更小的值产生更精细的网格，但计算时间更长 |
| `--iso-level` | float | 1.0 | 等值面阈值。根据训练日志中的 "Mean Density" 值调整（建议设为 1.0） |
| `--padding` | float | 0.5 | 边界框周围的填充（米）。增大此值可以确保场景边界被包含 |
| `--chunk-size` | int | 64 | 批处理块大小。如果遇到 OOM，减小此值 |
| `--truncate-sigma` | float | 3.0 | 截断距离（σ倍数）。只计算距离 voxel 中心 3σ 范围内的高斯，用于加速 |
| `--device` | str | "cuda" | 计算设备（"cuda" 或 "cpu"） |

### 参数选择建议

#### 体素大小 (`--voxel-size`)

- **精细网格**：`0.01` - 适用于小型场景或需要高精度的场景
- **标准网格**：`0.02` - 平衡质量和速度的默认选择
- **粗糙网格**：`0.05` - 快速预览或大型场景

#### 等值面阈值 (`--iso-level`)

- 根据训练日志中的 **Mean Density** 值设置
- 如果 Mean Density 约为 1.15，建议使用 `1.0`
- 如果提取的网格有空洞，尝试降低阈值（如 `0.8`）
- 如果网格太厚，尝试提高阈值（如 `1.2`）

#### 批处理大小 (`--chunk-size`)

- 如果遇到 GPU 显存不足（OOM），减小此值（如 `32` 或 `16`）
- 如果显存充足，可以增大此值（如 `128`）以加速计算

## 使用示例

### 示例 1：快速预览网格

```bash
python scripts/extract_mesh_isogs.py configs/replica/splatam.py \
    --voxel-size 0.05 \
    --output preview_mesh.ply
```

使用较大的体素快速生成粗糙网格用于预览。

### 示例 2：高质量网格提取

```bash
python scripts/extract_mesh_isogs.py configs/replica/splatam.py \
    --voxel-size 0.01 \
    --iso-level 1.0 \
    --chunk-size 32 \
    --output high_quality_mesh.ply
```

使用小体素生成高精度网格，适合最终输出。

### 示例 3：处理大场景

```bash
python scripts/extract_mesh_isogs.py configs/scannet/splatam.py \
    --voxel-size 0.02 \
    --chunk-size 32 \
    --truncate-sigma 2.5 \
    --padding 1.0
```

对于大型场景：
- 减小 `chunk-size` 避免 OOM
- 可以稍微减小 `truncate-sigma` 加速
- 增大 `padding` 确保边界完整

### 示例 4：从特定 checkpoint 提取

```bash
python scripts/extract_mesh_isogs.py configs/replica/splatam.py \
    --checkpoint params50.npz \
    --output mesh_frame50.ply
```

从训练过程中的特定 checkpoint 提取网格，用于观察训练进度。

## 算法原理

### 密度函数

对于空间中的任意一点 $x$，其密度值 $D(x)$ 定义为所有高斯核在该点的累加：

$$D(x) = \sum_{i=1}^{N} o_i \cdot \exp\left(-\frac{1}{2} (x - \mu_i)^T \Sigma_i^{-1} (x - \mu_i)\right)$$

其中：
- $\mu_i$: 第 $i$ 个高斯的均值（中心位置）
- $\Sigma_i$: 第 $i$ 个高斯的协方差矩阵
- $o_i$: 第 $i$ 个高斯的不透明度（opacity）

### 协方差矩阵构建

协方差矩阵由缩放（Scaling）和旋转（Rotation）构建：

$$\Sigma = R S S^T R^T$$

其中：
- $S = \text{diag}([s_x, s_y, s_z])$ 是对角缩放矩阵
- $R$ 是从四元数构建的旋转矩阵

逆协方差矩阵为：

$$\Sigma^{-1} = R S^{-2} R^T$$

### 网格提取流程

1. **确定范围**：根据高斯中心位置计算场景的边界框（Bounding Box）
2. **创建体素网格**：在边界框内创建均匀的 3D 网格
3. **计算密度**：
   - 对每个体素中心点计算密度值
   - 使用分块处理（Chunking）避免显存溢出
   - 使用 3σ 截断加速：只考虑距离体素中心 3 倍标准差范围内的高斯
4. **Marching Cubes**：在密度场上提取等值面
5. **清理**：保留最大连通分量，移除孤立碎片

## 常见问题

### Q0: 运行时报错 "can't open file '.../scripts/scripts/extract_mesh_isogs.py'"

**原因**：当前在 `scripts/` 目录下，但命令中又写了 `scripts/`，导致路径重复。

**解决方案**：
```bash
# 方法1：回到项目根目录（推荐）
cd ~/IsoGS-SLAM/SplaTAM
python scripts/extract_mesh_isogs.py configs/replica/splatam.py

# 方法2：在 scripts 目录下直接运行（调整路径）
python extract_mesh_isogs.py ../configs/replica/splatam.py
```

### Q1: 运行时报错 "out of memory"

**解决方案**：
- 减小 `--chunk-size`（如改为 `32` 或 `16`）
- 增大 `--voxel-size`（如改为 `0.05`）
- 减小 `--truncate-sigma`（如改为 `2.5`）

### Q2: 提取的网格有空洞

**可能原因**：
- 等值面阈值设置过高
- 训练不充分，某些区域密度不足

**解决方案**：
- 降低 `--iso-level`（如改为 `0.8` 或 `0.9`）
- 检查训练日志，确保 Mean Density 已收敛
- 减小 `--voxel-size` 提高精度

### Q3: 网格太厚或包含噪声

**解决方案**：
- 提高 `--iso-level`（如改为 `1.2` 或 `1.3`）
- 检查训练过程中的密度损失是否正常

### Q4: 提取速度太慢

**优化建议**：
- 增大 `--chunk-size`（如果显存允许）
- 增大 `--voxel-size`
- 使用 GPU（确保 `--device cuda`）

### Q5: 找不到 checkpoint 文件

**检查**：
- 确保实验已经运行并保存了 checkpoint
- 检查 `config['workdir']` 和 `config['run_name']` 是否正确
- 手动指定 checkpoint 路径：`--checkpoint params100.npz`

### Q6: 网格质量不理想

**可能原因**：
- 训练不足或参数设置不当
- 高斯分布不够扁平（Ratio 太小）

**解决方案**：
- 检查训练日志中的 Ratio 是否足够大（建议 > 5.0）
- 确保 Iso-Surface Loss 正常收敛
- 尝试不同的 `--iso-level` 值

### Q7: 如何使用提取的网格？

提取的 `.ply` 文件可以用以下工具打开：
- **MeshLab**：`meshlab mesh.ply`
- **Blender**：导入 PLY 文件
- **CloudCompare**
- **Open3D** (Python)：
  ```python
  import open3d as o3d
  mesh = o3d.io.read_triangle_mesh("mesh.ply")
  o3d.visualization.draw_geometries([mesh])
  ```

## 输出说明

脚本会输出以下信息：

```
Loading checkpoint: /path/to/params.npz
Loaded 245678 Gaussians
  Means shape: torch.Size([245678, 3])
  Scales shape: torch.Size([245678, 3])
  Opacities range: [0.0123, 0.9876]
Bounding box: min=[-2.5, -1.8, -0.5], max=[2.3, 1.9, 3.2]
Voxel grid dimensions: [240, 185, 185] (voxel_size=0.02)
Total voxels: 8,214,000
Processing 8214000 query points in 128363 chunks...
Computing density: 100%|████████████| 128363/128363 [15:32<00:00, 137.85it/s]
Density statistics:
  Min: 0.0000
  Max: 4.5678
  Mean: 0.8234
  Std: 0.3456
Extracting mesh at iso-level 1.0...
Extracted mesh: 45678 vertices, 91234 faces
Cleaning mesh...
Found 3 connected components
Keeping largest component with 45123 vertices
Final mesh: 45123 vertices, 90145 faces

Mesh saved to: /path/to/mesh.ply

Mesh Statistics:
  Vertices: 45,123
  Faces: 90,145
  Bounds: [[-2.5, -1.8, -0.5], [2.3, 1.9, 3.2]]
  Volume: 12.345678
```

## 性能参考

以下是在 NVIDIA RTX 4090 (24GB) 上的性能参考：

| 场景规模 | 高斯数量 | 体素大小 | 网格分辨率 | 计算时间 |
|---------|---------|---------|-----------|---------|
| 小型 | ~50K | 0.02 | ~200³ | ~2-3 分钟 |
| 中型 | ~250K | 0.02 | ~250³ | ~10-15 分钟 |
| 大型 | ~500K | 0.02 | ~300³ | ~30-45 分钟 |

*注：实际时间取决于场景复杂度和参数设置*

## 相关文件

- `scripts/extract_mesh_isogs.py` - 网格提取脚本
- `scripts/export_ply.py` - 导出高斯点云为 PLY（不同于网格提取）
- `scripts/splatam.py` - 主训练脚本

## 许可证

本脚本遵循项目主许可证。

