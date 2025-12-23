#!/usr/bin/env bash

# 自动批量运行 Replica (V1) 所有 8 个场景的网格几何质量评估：
# 评估预测网格与真实网格的几何质量（多种指标：Accuracy, Completion, Chamfer Distance, F-score等）
#
# 使用方式（推荐在已经激活 isogs 的终端里运行）：
#   cd /media/pw_is_6/Disk2/IsoGS-SLAM/SplaTAM
#   bash bash_scripts/run_replica_mesh_eval_all.sh
#
# 可选参数：
#   --render-views: 启用mesh渲染可视化（从数据集位姿渲染对比图）
#   --render-every N: 渲染每N帧（默认：10）
#
# 如果你希望脚本自己激活 conda，请根据你本机路径修改 CONDA_BASE，再取消注释相关几行。
#
# ========================================================================
# 输入输出文件目录结构说明：
# ========================================================================
#
# 【输入文件】
# 1. 真实网格（Ground Truth Mesh）：
#    路径: data/Replica/{scene_name}_mesh.ply
#    示例: data/Replica/office0_mesh.ply
#    说明: Replica 数据集提供的真实场景网格文件
#
# 2. 预测网格（Predicted Mesh）：
#    路径: experiments/Replica/{scene_name}_0/（自动查找最新的mesh_thickened_*.ply）
#    说明: 脚本会自动查找最新的mesh文件（按帧号排序）
#
# 【输出】
# 结果保存在: experiments/Replica/{scene_name}_0/mesh_compare_eval/
#   - mesh_eval_results.csv: 评估指标和mesh统计信息（CSV格式）
#   - mesh_eval_results.txt: 详细的评估结果（文本格式）
#   - view/frame_xxxx.png: 预测mesh渲染视图（如果启用--render-views）
#   - rendered_view/frame_xxxx.png: GT mesh渲染视图（如果启用--render-views）
#   - compare_view_plots/frame_xxxx.png: 对比图（如果启用--render-views）
#
# ========================================================================

# 不使用 set -e，允许单个任务失败时继续执行其他任务
# set -e

PROJECT_ROOT="/media/pw_is_6/Disk2/IsoGS-SLAM/SplaTAM"

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

# 8个场景 (Replica V1 命名：room0, office0 等)
SCENES=("room0" "room1" "room2" "office0" "office1" "office2" "office3" "office4")
NUM_SCENES=${#SCENES[@]}

# 数据目录和实验结果目录
DATA_DIR="$PROJECT_ROOT/data/Replica"
EXPERIMENTS_DIR="$PROJECT_ROOT/experiments/Replica"

# 解析命令行参数
RENDER_VIEWS=false
RENDER_EVERY=10
ALIGN_MESH=""
for arg in "$@"; do
    case $arg in
        --render-views)
            RENDER_VIEWS=true
            shift
            ;;
        --render-every=*)
            RENDER_EVERY="${arg#*=}"
            shift
            ;;
        --align-mesh=*)
            ALIGN_MESH="${arg#*=}"
            shift
            ;;
        *)
            ;;
    esac
done

# 交互式选择mesh对齐方式
if [ -z "$ALIGN_MESH" ]; then
    echo "================================================================"
    echo "请选择mesh对齐方式："
    echo "  1) center - 中心对齐（推荐，快速）"
    echo "  2) icp    - ICP对齐（更准确，需要open3d，较慢）"
    echo "  3) none   - 不对齐（不推荐，除非确定坐标系一致）"
    echo "================================================================"
    read -p "请输入选项 (1/2/3，默认: 1): " align_choice
    
    case $align_choice in
        1|"")
            ALIGN_MESH="center"
            ;;
        2)
            ALIGN_MESH="icp"
            ;;
        3)
            ALIGN_MESH="none"
            ;;
        *)
            echo "无效选项，使用默认值: center"
            ALIGN_MESH="center"
            ;;
    esac
fi

echo "已选择mesh对齐方式: $ALIGN_MESH"
echo

echo "================================================================"
echo "开始批量评估 Replica (V1) 数据集 - 网格几何质量"
echo "场景数量: $NUM_SCENES"
echo "================================================================"
echo
echo "输入文件目录:"
echo "  真实网格: $DATA_DIR/{scene_name}_mesh.ply"
echo "  预测网格: $EXPERIMENTS_DIR/{scene_name}_0/（自动查找最新mesh文件）"
echo
echo "输出目录:"
echo "  experiments/Replica/{scene_name}_0/mesh_compare_eval/"
echo "  Mesh对齐方式: $ALIGN_MESH"
if [ "$RENDER_VIEWS" = true ]; then
    echo "  渲染视图: 启用（每 $RENDER_EVERY 帧）"
else
    echo "  渲染视图: 禁用（使用 --render-views 启用）"
fi
echo "================================================================"
echo

TOTAL_TASKS=$NUM_SCENES
CURRENT_TASK=0
SUCCESS_COUNT=0
FAIL_COUNT=0

# 遍历所有场景
for SCENE_IDX in "${!SCENES[@]}"; do
    SCENE_NAME="${SCENES[$SCENE_IDX]}"
    CURRENT_TASK=$((CURRENT_TASK + 1))
    
    echo "================================================================"
    echo "[任务 $CURRENT_TASK/$TOTAL_TASKS] $SCENE_NAME"
    echo "================================================================"
    
    # 构建文件路径
    GT_MESH="$DATA_DIR/${SCENE_NAME}_mesh.ply"
    PRED_DIR="$EXPERIMENTS_DIR/${SCENE_NAME}_0"
    
    # 检查文件是否存在
    if [ ! -f "$GT_MESH" ]; then
        echo "✗ 错误: 真实网格文件不存在: $GT_MESH"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "继续执行下一个任务..."
        echo
        continue
    fi
    
    if [ ! -d "$PRED_DIR" ]; then
        echo "✗ 错误: 预测网格目录不存在: $PRED_DIR"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "继续执行下一个任务..."
        echo
        continue
    fi
    
    # 构建评估命令
    EVAL_CMD="python scripts/eval_mesh_geometry.py --pred-dir \"$PRED_DIR\" --gt \"$GT_MESH\""
    
    # 添加mesh对齐参数
    EVAL_CMD="$EVAL_CMD --align-mesh $ALIGN_MESH"
    
    # 如果启用渲染视图，添加相关参数
    if [ "$RENDER_VIEWS" = true ]; then
        EVAL_CMD="$EVAL_CMD --render-views"
        EVAL_CMD="$EVAL_CMD --render-every $RENDER_EVERY"
        EVAL_CMD="$EVAL_CMD --dataset-config configs/data/replica.yaml"
        EVAL_CMD="$EVAL_CMD --dataset-basedir $DATA_DIR"
        EVAL_CMD="$EVAL_CMD --dataset-sequence $SCENE_NAME"
    fi
    
    # 运行评估
    echo "真实网格: $GT_MESH"
    echo "预测网格目录: $PRED_DIR（自动查找最新mesh文件）"
    echo "执行命令: $EVAL_CMD"
    echo "------------------------------------------------------------"
    START_TIME=$(date +%s)
    if eval $EVAL_CMD; then
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "✓ 成功完成: $SCENE_NAME (耗时: ${ELAPSED}秒)"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "✗ 失败: $SCENE_NAME (耗时: ${ELAPSED}秒)"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        echo "继续执行下一个任务..."
    fi
    
    echo
done

echo "================================================================"
echo "所有 Replica (V1) 场景的网格几何质量评估已完成！"
echo "================================================================"
echo
echo "统计信息:"
echo "  总任务数: $TOTAL_TASKS"
echo "  成功: $SUCCESS_COUNT"
echo "  失败: $FAIL_COUNT"
echo
echo "评估指标说明:"
echo "  - Accuracy (L1): 预测网格点到最近真实网格点的平均距离（单位：cm）"
echo "  - Completion (L1): 真实网格点到最近预测网格点的平均距离（单位：cm）"
echo "  - Chamfer Distance: Accuracy和Completion的平均值（单位：cm）"
echo "  - Completion Ratio: 真实网格点在阈值范围内的比例（单位：%，默认阈值：5cm）"
echo "  - F-score: Precision和Recall的调和平均数"
echo "  - Hausdorff Distance: 最大最近邻距离（单位：cm）"
echo
echo "结果文件位置:"
echo "  experiments/Replica/<scene_name>_0/mesh_compare_eval/"
echo "    - mesh_eval_results.csv: 所有指标和mesh统计信息（CSV格式）"
echo "    - mesh_eval_results.txt: 详细评估结果（文本格式）"
if [ "$RENDER_VIEWS" = true ]; then
    echo "    - view/: 预测mesh渲染视图"
    echo "    - rendered_view/: GT mesh渲染视图"
    echo "    - compare_view_plots/: 对比图"
fi
