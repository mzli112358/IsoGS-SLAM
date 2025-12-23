#!/usr/bin/env bash

# 自动批量运行 Replica 所有 8 个场景：
# 1) 运行 SLAM 到 800 帧
# 2) 导出高斯场景（PLY）
# 3) 从高斯场景提取 mesh（PLY/OBJ/TXT）
#
# 使用方式（推荐在已经激活 isogs 的终端里运行）：
#   cd /home/pw_is_6/IsoGS-SLAM/SplaTAM
#   bash bash_scripts/run_replica_all_scenes.sh
#
# 如果你希望脚本自己激活 conda，请根据你本机路径修改 CONDA_BASE，再取消注释相关几行。

set -e

PROJECT_ROOT="/home/pw_is_6/IsoGS-SLAM/SplaTAM"

########################################
# 自动激活 conda 环境
########################################
# 检测 conda 路径（尝试多个常见路径）
if [ -z "$CONDA_BASE" ]; then
    if [ -d "$HOME/anaconda3" ]; then
        CONDA_BASE="$HOME/anaconda3"
    elif [ -d "$HOME/miniconda3" ]; then
        CONDA_BASE="$HOME/miniconda3"
    elif [ -d "/opt/conda" ]; then
        CONDA_BASE="/opt/conda"
    else
        echo "[Error] 无法找到 conda 安装路径，请手动激活 isogs 环境后运行"
        exit 1
    fi
fi

# 激活 conda 环境
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1090
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    # 检查是否已在 isogs 环境中
    if [ "$CONDA_DEFAULT_ENV" != "isogs" ]; then
        echo "[Info] 激活 conda 环境: isogs"
        conda activate isogs
    else
        echo "[Info] 已在 isogs 环境中"
    fi
else
    echo "[Error] conda.sh 未找到: $CONDA_BASE/etc/profile.d/conda.sh"
    echo "[Error] 请手动激活 isogs 环境后运行"
    exit 1
fi

cd "$PROJECT_ROOT"

SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")

for IDX in "${!SCENES[@]}"; do
    SCENE_NAME="${SCENES[$IDX]}"
    echo "================================================================"
    echo "[Scene] Index: $IDX, Name: $SCENE_NAME"
    echo "================================================================"

    # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
    export SPLATAM_SCENE_INDEX="$IDX"

    echo "[Step 1] Run SLAM until frame 300 ..."
    python scripts/splatam.py configs/replica/splatam.py --end-at 300

    echo "[Step 2] Export Gaussian PLY ..."
    python scripts/export_ply.py configs/replica/splatam.py

    echo "[Step 3] Extract mesh from Gaussian field ..."
    python scripts/extract_mesh_fast.py configs/replica/splatam.py \
        --voxel-size 0.01 \
        --iso-level 0.3 \
        --no-cleaning \
        --no-show

    echo "[Done] Scene $SCENE_NAME (index $IDX) finished."
    echo
done

echo "================================================================"
echo "All Replica scenes finished."
echo "================================================================"


