#!/usr/bin/env bash

# 自动批量运行 Replica 所有 8 个场景：
# 1) 运行 SLAM 到 800 帧
# 2) 导出高斯场景（PLY）
# 3) 从高斯场景提取 mesh（PLY/OBJ/TXT）
#
# 使用方式（推荐在已经激活 isogs 的终端里运行）：
#   cd /media/pw_is_6/Disk2/IsoGS-SLAM/SplaTAM
#   bash bash_scripts/run_replica_all_scenes.sh
#
# 如果你希望脚本自己激活 conda，请根据你本机路径修改 CONDA_BASE，再取消注释相关几行。

set -e

# 自动获取项目根目录（脚本所在目录的上一级目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

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
NUM_SCENES=${#SCENES[@]}

########################################
# 交互式选择要执行的步骤
########################################
echo "================================================================"
echo "请选择要执行的步骤（输入数字组合，例如: 1, 2, 3, 12, 13, 23, 123）"
echo "  1) 运行 SLAM 到 2000 帧"
echo "  2) 导出高斯场景（PLY）"
echo "  3) 从高斯场景提取 mesh"
echo "================================================================"
read -p "请输入选择: " USER_INPUT

# 解析用户输入，检查是否包含步骤1、2、3
DO_STEP1=false
DO_STEP2=false
DO_STEP3=false

if [[ "$USER_INPUT" == *"1"* ]]; then
    DO_STEP1=true
fi
if [[ "$USER_INPUT" == *"2"* ]]; then
    DO_STEP2=true
fi
if [[ "$USER_INPUT" == *"3"* ]]; then
    DO_STEP3=true
fi

# 验证输入是否有效
if [ "$DO_STEP1" = false ] && [ "$DO_STEP2" = false ] && [ "$DO_STEP3" = false ]; then
    echo "[Error] 无效的输入，请至少选择一个步骤（1、2或3）"
    exit 1
fi

echo
echo "将执行以下步骤:"
[ "$DO_STEP1" = true ] && echo "  ✓ 步骤1: 运行 SLAM"
[ "$DO_STEP2" = true ] && echo "  ✓ 步骤2: 导出高斯场景（PLY）"
[ "$DO_STEP3" = true ] && echo "  ✓ 步骤3: 提取 mesh"
echo

########################################
# 步骤1: 对所有场景运行 SLAM
########################################
if [ "$DO_STEP1" = true ]; then
    echo "================================================================"
    echo "[Step 1] Run SLAM until frame 2000 for all scenes"
    echo "================================================================"
    echo

    for IDX in "${!SCENES[@]}"; do
        SCENE_NAME="${SCENES[$IDX]}"
        echo "------------------------------------------------------------"
        echo "[Scene $((IDX + 1))/$NUM_SCENES] Index: $IDX, Name: $SCENE_NAME"
        echo "------------------------------------------------------------"
        
        # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
        export SPLATAM_SCENE_INDEX="$IDX"
        
        python scripts/splatam.py configs/replica/splatam.py --end-at 2000
        
        echo "[Done] Scene $SCENE_NAME SLAM finished."
        echo
    done
fi

########################################
# 步骤2: 对所有场景导出高斯场景（PLY）
########################################
if [ "$DO_STEP2" = true ]; then
    echo "================================================================"
    echo "[Step 2] Export Gaussian PLY for all scenes"
    echo "================================================================"
    echo

    for IDX in "${!SCENES[@]}"; do
        SCENE_NAME="${SCENES[$IDX]}"
        echo "------------------------------------------------------------"
        echo "[Scene $((IDX + 1))/$NUM_SCENES] Index: $IDX, Name: $SCENE_NAME"
        echo "------------------------------------------------------------"
        
        # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
        export SPLATAM_SCENE_INDEX="$IDX"
        
        python scripts/export_ply.py configs/replica/splatam.py
        
        echo "[Done] Scene $SCENE_NAME PLY export finished."
        echo
    done
fi

########################################
# 步骤3: 对所有场景提取 mesh
########################################
if [ "$DO_STEP3" = true ]; then
    echo "================================================================"
    echo "[Step 3] Extract mesh from Gaussian field for all scenes"
    echo "================================================================"
    echo

    for IDX in "${!SCENES[@]}"; do
        SCENE_NAME="${SCENES[$IDX]}"
        echo "------------------------------------------------------------"
        echo "[Scene $((IDX + 1))/$NUM_SCENES] Index: $IDX, Name: $SCENE_NAME"
        echo "------------------------------------------------------------"
        
        # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
        export SPLATAM_SCENE_INDEX="$IDX"
        
        python scripts/extract_mesh_fast.py configs/replica/splatam.py \
            --voxel-size 0.015 \
            --iso-level 0.3 \
            --no-cleaning \
            --no-show
        
        echo "[Done] Scene $SCENE_NAME mesh extraction finished."
        echo
    done
fi

echo "================================================================"
echo "All Replica scenes finished."
echo "================================================================"


