# SplaTAM 可视化教程

本教程介绍如何导出和可视化 SplaTAM 生成的高斯点云数据。

## 目录

1. [导出点云为 PLY 文件](#1-导出点云为-ply-文件)
2. [使用项目代码可视化](#2-使用项目代码可视化)
3. [使用网页工具可视化](#3-使用网页工具可视化)

---

## 1. 导出点云为 PLY 文件

### 自动查找最新参数文件

`export_ply.py` 脚本会自动查找最新的参数文件：
- 如果存在 `params.npz`（最终结果），则使用它
- 如果不存在，则查找最新的检查点文件（如 `params100.npz`、`params600.npz`、`params1250.npz` 等），使用数字最大的那个

### 使用方法

```bash
python scripts/export_ply.py configs/replica/splatam.py
```

### 输出文件

- 如果使用 `params.npz`，输出文件为：`experiments/Replica/room0_0/splat.ply`
- 如果使用检查点文件（如 `params100.npz`），输出文件为：`experiments/Replica/room0_0/splat_100.ply`

**💡 提示**：`export_ply.py` 是唯一支持检查点文件的工具，因为导出点云不需要相机参数。如果您的实验还在运行中，可以使用此工具导出中间检查点的结果，然后用网页工具（如 SuperSplat）查看。

### 示例

假设实验目录中有以下文件：
```
experiments/Replica/room0_0/
├── params0.npz
├── params50.npz
├── params100.npz
└── params150.npz  (最新的检查点)
```

运行导出命令后，会自动使用 `params150.npz`，并生成 `splat_150.ply`。

---

## 2. 使用项目代码可视化

### 2.1 最终重建可视化（推荐）

使用 `viz_scripts/final_recon.py` 可以以交互式方式可视化最终的 SLAM 重建结果。

#### 自动查找最新参数文件

与导出脚本类似，`final_recon.py` 会自动查找最新的参数文件：
- 如果存在 `params.npz`（最终结果），则使用它
- 如果不存在，则查找最新的检查点文件（如 `params100.npz`、`params600.npz` 等），使用数字最大的那个

**⚠️ 重要提示：检查点文件的限制**

**检查点文件（`params100.npz` 等）不能用于可视化！** 虽然脚本会自动查找这些文件，但检查点文件只包含高斯点参数，缺少可视化所需的相机参数（`org_width`、`org_height`、`w2c`、`intrinsics` 等），因此会导致错误。

**解决方案：**
1. **推荐方式**：等待实验完成，使用最终的 `params.npz` 文件进行可视化
2. **查看中间结果**：如果需要查看中间检查点的结果，使用 `export_ply.py` 导出点云，然后用网页工具（如 SuperSplat）查看

#### 基本使用

```bash
python viz_scripts/final_recon.py configs/replica/splatam.py
```

运行时会显示使用的参数文件：
```
Found final params file: ./experiments/Replica/room0_0/params.npz
```

#### 可视化功能

- **交互式查看**：使用鼠标进行旋转、缩放、平移
- **渲染模式**：可在配置文件中设置不同的渲染模式
  - `color`：彩色渲染
  - `depth`：深度渲染
  - `centers`：仅显示高斯点中心
- **相机轨迹**：可选择显示相机轨迹和视锥体

#### 配置说明

可视化选项在配置文件的 `viz` 部分设置，例如 `configs/replica/splatam.py`：

```python
viz=dict(
    render_mode='color',  # ['color', 'depth' 或 'centers']
    offset_first_viz_cam=True,  # 将视图相机向后偏移 0.5 单位
    show_sil=False,  # 显示轮廓而不是 RGB
    visualize_cams=True,  # 可视化相机视锥和轨迹
    viz_w=600, viz_h=340,
    viz_near=0.01, viz_far=100.0,
    view_scale=2,
),
```

#### 查看特定检查点

**⚠️ 注意**：检查点文件（`params100.npz` 等）**不能用于可视化**，因为它们缺少相机参数。如果需要查看中间结果：

1. **推荐方式**：使用 `export_ply.py` 导出检查点的点云，然后用网页工具查看：
   ```bash
   # 导出点云（会自动找到最新的检查点）
   python scripts/export_ply.py configs/replica/splatam.py
   # 然后在 SuperSplat 或 PolyCam 中打开生成的 PLY 文件
   ```

2. **等待最终结果**：等待实验完成，使用最终的 `params.npz` 文件进行可视化

如果需要强制指定参数文件路径（仅适用于最终的 `params.npz`），可以临时修改配置文件，添加：

```python
config['scene_path'] = './experiments/Replica/room0_0/params.npz'
```

然后运行可视化命令。

### 2.2 在线重建可视化

使用 `viz_scripts/online_recon.py` 可以实时查看重建过程：

```bash
python viz_scripts/online_recon.py configs/replica/splatam.py
```

**注意**：
- `online_recon.py` 也会自动查找最新的参数文件，逻辑与 `final_recon.py` 相同
- **同样需要最终的 `params.npz` 文件**，检查点文件无法用于此可视化

#### 功能说明

`online_recon.py` 与 `final_recon.py` 的区别：
- **`final_recon.py`**：显示最终重建结果，可交互式查看完整的场景
- **`online_recon.py`**：按时间顺序播放重建过程，展示场景如何逐步构建的动态过程
  - 按时间顺序播放每一帧的重建结果
  - 实时显示相机轨迹和视锥体
  - 展示场景如何逐步完善
  - 可以观察到 SLAM 的实时重建过程

这个工具会按时间顺序播放重建过程，展示 SLAM 如何逐步构建场景。

---

## 3. 使用网页工具可视化

导出的 PLY 文件（3D Gaussian Splatting 格式）可以在多个网页查看器中查看，无需安装额外软件。

### 3.1 SuperSplat（强烈推荐）

**网址**：https://playcanvas.com/supersplat/editor

**使用方法**：
1. 访问上述网址
2. 直接将 PLY 文件拖拽到浏览器窗口
3. 等待加载完成（可能需要几秒到几分钟，取决于文件大小）
4. 使用鼠标进行交互：
   - **左键拖动**：旋转视角
   - **右键拖动**：平移场景
   - **滚轮**：缩放

**优点**：
- 界面简洁，易于使用
- 支持高质量的渲染
- 无需注册或登录
- 完全基于浏览器，跨平台

### 3.2 PolyCam

**网址**：https://poly.cam/tools/gaussian-splatting

**使用方法**：
1. 访问上述网址
2. 点击上传按钮或拖拽 PLY 文件
3. 等待处理完成
4. 在浏览器中查看和分享

**优点**：
- 支持分享链接
- 提供额外的编辑工具
- 支持多种格式导出

### 3.3 其他网页查看器

#### Antimatter15's 3D Gaussian Splatting Viewer

**网址**：https://antimatter15.com/splat/

- 轻量级查看器
- 快速加载
- 适合快速预览

#### Marko Mihajlovic's Viewer

**网址**：https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

- 由 INRIA 提供
- 功能完整
- 适合研究和学术用途

---

## 常见问题

### Q1: 导出的 PLY 文件无法在传统点云查看器中打开？

**A**: 导出的 PLY 文件是 3D Gaussian Splatting 格式，不是传统的点云格式。请使用上述推荐的网页查看器（SuperSplat、PolyCam 等）进行查看。这些查看器专门为高斯点渲染设计，能够正确显示效果。

### Q2: 如何提取传统点云格式（如仅包含位置和颜色）？

如果需要传统点云格式，可以编写脚本从高斯点中提取中心位置，并使用 Open3D 转换为标准点云格式。但这样会丢失高斯点的渲染特性。

### Q3: 可视化时显示空白或错误？

**A**: 请检查：
1. **参数文件类型**：确保使用最终的 `params.npz` 文件，而不是检查点文件（`params100.npz` 等）
   - 检查点文件缺少相机参数，无法用于可视化
   - 如果看到 `KeyError: 'org_width'` 错误，说明使用了检查点文件
   - 解决方案：等待实验完成生成 `params.npz`，或使用 `export_ply.py` 导出点云用网页工具查看
2. 参数文件是否存在且完整
3. 配置文件路径是否正确
4. 是否安装了所有依赖（特别是 Open3D、PyTorch 等）

### Q4: 网页查看器加载很慢？

**A**: 这是正常的，因为：
- 高斯点文件可能很大（几十到几百 MB）
- 需要通过网络上传文件
- 浏览器需要解析和渲染大量数据

建议使用 SuperSplat，它通常具有更好的性能。

---

## 快速参考

### 导出命令

```bash
# 基本导出（自动查找最新参数文件）
python scripts/export_ply.py configs/replica/splatam.py

# 其他数据集
python scripts/export_ply.py configs/tum/splatam.py
python scripts/export_ply.py configs/scannet/splatam.py
```

### 可视化命令

```bash
# 最终重建可视化
python viz_scripts/final_recon.py configs/replica/splatam.py

# 在线重建可视化
python viz_scripts/online_recon.py configs/replica/splatam.py
```

### 网页查看器链接

- **SuperSplat**: https://playcanvas.com/supersplat/editor
- **PolyCam**: https://poly.cam/tools/gaussian-splatting
- **Antimatter15**: https://antimatter15.com/splat/

---

## 更多资源

- [SplaTAM 项目页面](https://spla-tam.github.io/)
- [论文](https://arxiv.org/pdf/2312.02126.pdf)
- [GitHub 仓库](https://github.com/spla-tam/SplaTAM)

