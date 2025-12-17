## 快速网格提取命令速查（`extract_mesh_fast.py`）

本页是 **纯命令速查表**，方便复制粘贴使用 `scripts/extract_mesh_fast.py` 从 IsoGS/SplaTAM checkpoint 中快速提取网格（Tile-based/Block-based 快速算法版本）。

**建议运行位置：**

```bash
cd ~/IsoGS-SLAM/SplaTAM
```

---

## 一、最常用命令

- **1. 默认提取（自动选最新 checkpoint，默认清理，只保留最大连通块）**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py
```

- **2. 调整 iso-level 和输出文件名（仍然会清理网格）**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --iso-level 0.8 --output mesh_iso0.8.ply
```

- **3. 关闭清理功能，保留所有组件（墙壁、小碎片都保留）**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --no-cleaning
```

- **4. 同时指定 iso-level、输出文件名，并关闭清理**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --iso-level 0.8 --output mesh_iso0.8_full.ply --no-cleaning
```

- **5. 从指定 checkpoint 提取，并关闭清理**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --checkpoint params500.npz --iso-level 0.8 --output mesh_frame500_full.ply --no-cleaning
```

---

## 二、参数简要说明

- **必需参数**
  - **配置文件**：`configs/replica/splatam.py`  
    你也可以换成自己的配置，例如 `configs/scannet/splatam.py`。

- **常用可选参数（只列最常用的几个）**

| 参数 | 示例 | 说明 |
|------|------|------|
| `--checkpoint` | `--checkpoint params500.npz` | 指定某个 checkpoint 文件；如果不写，会自动选择 `params.npz` 或编号最大的 `paramsXXXX.npz` |
| `--output` | `--output mesh_iso0.8_full.ply` | 指定输出网格文件名（相对于当前实验结果目录） |
| `--iso-level` | `--iso-level 0.8` | 等值面阈值：数值越小，网格越“鼓”、可能更多噪声；越大，网格越“瘦”、可能有空洞 |
| `--voxel-size` | `--voxel-size 0.02` | 体素大小：越小越精细，越大越快 |
| `--padding` | `--padding 0.5` | 在高斯边界外扩展的范围（米） |
| `--block-size` | `--block-size 16` | Block 尺寸，影响 tile-based 算法的分块大小 |
| `--truncate-sigma` | `--truncate-sigma 3.0` | 截断距离（σ），越小越快，但可能丢细节 |
| `--device` | `--device cuda` / `--device cpu` | 指定使用 GPU 还是 CPU |
| `--no-cleaning` | `--no-cleaning` | **关闭 mesh 清理**，不再只保留最大连通块 |

---

## 三、典型场景示例

- **房间完整性调试：看墙是否被误删**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --iso-level 0.8 --output room0_0_full_iso0.8.ply --no-cleaning
```

- **快速预览（粗网格 + 仍然清理）**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --voxel-size 0.05 --output preview_fast.ply
```

- **高质量结果（精细体素 + 清理）**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --voxel-size 0.01 --iso-level 1.0 --output high_quality_fast.ply
```

- **固定某一帧 checkpoint，对比训练过程**

```bash
python scripts/extract_mesh_fast.py configs/replica/splatam.py --checkpoint params200.npz --iso-level 1.0 --output mesh_frame200_fast.ply
```

---

## 四、输出与查看网格

- **脚本会自动输出两个文件：**
  - 一个 `.ply`：例如 `mesh_fast.ply` 或你通过 `--output` 指定的名字
  - 一个同名 `.obj`：例如 `mesh_fast.obj`，与 `.ply` 在同一目录

- **默认会自动弹出 3D Viewer 窗口**（使用 `trimesh` 内置交互界面），用于快速查看结果。
  - 如果你不想每次都弹窗，可以在命令中加上：

    ```bash
    --no-show
    ```

### 用外部工具查看

你可以直接用常见 3D 工具打开 `.obj` 或 `.ply`：

- **MeshLab**

```bash
meshlab experiments/Replica/room0_0/mesh_iso0.8_full.obj
```

- **Blender / CloudCompare**：直接导入 `.obj` 或 `.ply` 文件即可。


