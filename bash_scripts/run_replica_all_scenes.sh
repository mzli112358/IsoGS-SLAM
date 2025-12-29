#!/usr/bin/env bash

# 自动批量运行 Replica 场景任务：
# 1) 运行 SLAM 到 2000 帧
# 2) 导出高斯场景（PLY）
# 3) 从高斯场景提取 mesh（密度场方法，PLY/OBJ/TXT）
# 4) 从高斯场景提取 mesh（TSDF融合方法，快速）
#
# 使用方式（推荐在已经激活 isogs 的终端里运行）：
#   cd /media/pw_is_6/Disk2/IsoGS-SLAM/IsoGS-SLAM-main
#   bash bash_scripts/run_replica_all_scenes.sh
#
# 输入格式：
#   两位数字，空格分隔（十位=场景序号0-7，个位=步骤序号1-4）
#   示例：31 32 33 41 43  → 场景3执行步骤1,2,3；场景4执行步骤1,3
#   示例：01 11 21 31     → 场景0,1,2,3都执行步骤1
#   示例：14 24 34        → 场景1,2,3都执行步骤4（TSDF方法）

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
# 任务选择：两位数字编码（十位=场景，个位=步骤）
########################################
echo "================================================================"
echo "请选择要执行的任务（空格分隔，可多选）"
echo ""
echo "输入格式：两位数字"
echo "  - 十位数（0-7）：场景序号"
echo "  - 个位数（1-4）：步骤序号"
echo ""
echo "步骤说明："
echo "  1) 运行 SLAM 到 2000 帧"
echo "  2) 导出高斯场景（PLY）"
echo "  3) 从高斯场景提取 mesh（密度场方法，较慢但精确）"
echo "  4) 从高斯场景提取 mesh（TSDF融合方法，快速）"
echo ""
echo "场景列表："
for IDX in "${!SCENES[@]}"; do
    echo "  $IDX) ${SCENES[$IDX]}"
done
echo ""
echo "示例："
echo "  31 32 33 41 43  → 场景3执行步骤1,2,3；场景4执行步骤1,3"
echo "  01 11 21 31     → 场景0,1,2,3都执行步骤1"
echo "  14 24 34        → 场景1,2,3都执行步骤4（TSDF方法）"
echo "================================================================"
read -p "请输入任务（空格分隔）: " TASK_INPUT

# 解析输入的任务列表
TASKS=()

# 按空格分割输入
IFS=' ' read -ra TASK_PARTS <<< "$TASK_INPUT"
for task_code in "${TASK_PARTS[@]}"; do
    # 去除前后空格
    task_code=$(echo "$task_code" | xargs)
    
    # 跳过空字符串
    if [ -z "$task_code" ]; then
        continue
    fi
    
    # 验证格式：必须是两位数字
    if [[ ! "$task_code" =~ ^[0-9]{2}$ ]]; then
        echo "[Warning] 跳过无效的任务代码: $task_code (必须是两位数字)"
        continue
    fi
    
    # 提取场景索引和步骤
    scene_idx=$((10#${task_code:0:1}))
    step_num=$((10#${task_code:1:1}))
    
    # 验证场景索引
    if [ "$scene_idx" -lt 0 ] || [ "$scene_idx" -ge "$NUM_SCENES" ]; then
        echo "[Warning] 跳过无效的场景索引: $scene_idx (有效范围: 0-$((NUM_SCENES-1)))"
        continue
    fi
    
    # 验证步骤编号
    if [ "$step_num" -lt 1 ] || [ "$step_num" -gt 4 ]; then
        echo "[Warning] 跳过无效的步骤编号: $step_num (有效范围: 1-4)"
        continue
    fi
    
    # 添加到任务列表（保持输入顺序）
    TASKS+=("$scene_idx $step_num")
done

# 验证是否有有效任务
if [ ${#TASKS[@]} -eq 0 ]; then
    echo "[Error] 没有有效的任务，程序退出"
    exit 1
fi

# 显示将要执行的任务
echo
echo "将按顺序执行以下任务:"
TASK_NUM=1
for task in "${TASKS[@]}"; do
    read -r scene_idx step_num <<< "$task"
    scene_name="${SCENES[$scene_idx]}"
    case $step_num in
        1) step_name="运行 SLAM" ;;
        2) step_name="导出高斯场景（PLY）" ;;
        3) step_name="提取 mesh（密度场方法）" ;;
        4) step_name="提取 mesh（TSDF方法）" ;;
        *) step_name="未知步骤" ;;
    esac
    echo "  $TASK_NUM) 场景 $scene_idx ($scene_name) - $step_name"
    TASK_NUM=$((TASK_NUM + 1))
done
echo

########################################
# 按顺序执行任务
########################################
TOTAL_TASKS=${#TASKS[@]}
TASK_COUNTER=0

for task in "${TASKS[@]}"; do
    TASK_COUNTER=$((TASK_COUNTER + 1))
    read -r scene_idx step_num <<< "$task"
    scene_name="${SCENES[$scene_idx]}"
    
    echo "================================================================"
    echo "[Task $TASK_COUNTER/$TOTAL_TASKS] Scene $scene_idx ($scene_name) - Step $step_num"
    echo "================================================================"
    echo
    
    # 通过环境变量控制 configs/replica/splatam.py 中的 scene_name
    export SPLATAM_SCENE_INDEX="$scene_idx"
    
    case $step_num in
        1)
            echo "执行步骤1: 运行 SLAM 到 2000 帧"
            python scripts/splatam.py configs/replica/splatam.py --end-at 2000
            echo "[Done] Scene $scene_name - SLAM finished."
            ;;
        2)
            echo "执行步骤2: 导出高斯场景（PLY）"
            python scripts/export_ply.py configs/replica/splatam.py
            echo "[Done] Scene $scene_name - PLY export finished."
            ;;
        3)
            echo "执行步骤3: 提取 mesh（密度场方法）"
            python scripts/extract_mesh_fast.py configs/replica/splatam.py \
                --voxel-size 0.015 \
                --iso-level 0.3 \
                --no-cleaning \
                --no-show \
                --block-size 128
            echo "[Done] Scene $scene_name - mesh extraction (density field) finished."
            ;;
        4)
            echo "执行步骤4: 提取 mesh（TSDF融合方法）"
            python scripts/extract_mesh_tsdf.py configs/replica/splatam.py \
                --voxel_size 0.015 \
                --skip 5 \
                --depth_trunc 4.0 \
                --sdf_trunc 0.06

            # ============================================================
            # 参数说明注释（保留作为参考）
            # ============================================================
            # --voxel_size: 决定 Mesh 的精细度。0.01~0.02 是比较好的范围。
            # --skip 5: 每 5 帧融合一次。如果想更快，设成 10；如果想更密，设成 2。对于 2000 帧，设 5 或 10 足够了。
            # --depth_trunc: 深度截断。室内场景一般设 3.0 或 4.0 米，太远的数据噪声大，不要融合。
            # --sdf_trunc 0.04 —— 强烈建议加，甚至可以微调
            #   这是什么？：SDF 截断距离（Truncation Distance）。它是 TSDF 算法中定义"表面有多厚"的参数。只有在这个距离内的深度数据才会被融合。
            #   黄金法则：通常设置为 Voxel Size 的 3 到 5 倍。
            #   如果你设得太小（< 2倍）：表面会很脆，稍微一点定位误差就会导致表面破洞，或者融合不起来。
            #   如果你设得太大（> 5倍）：表面会变厚，细节会糊掉，甚至不同物体会粘连。
            #   你的情况：voxel_size = 0.015 (1.5cm) sdf_trunc 默认是 0.04。0.04/0.015≈2.6 倍。这个值略微偏紧，可能会导致一些扫描不充分的地方有空洞。
            #   我的建议：建议改为 0.06 (约 4 倍)。这会让 Mesh 更连贯、更平滑，这对发论文贴图来说视觉效果更好。
            # --no-cleaning —— 千万别加！
            #   这是什么？：这个开关是用来关闭"去噪清理"功能的。
            #   默认行为：脚本会自动运行"连通域分析"，只保留最大的那个连通块（也就是房间主体），自动删掉悬浮在空中的噪点（Flying Pixels）。
            #   为什么不加？：为了论文图好看！
            #   如果你加了 --no-cleaning，导出的 Mesh 周围会有很多悬浮的碎渣（高斯伪影）。在做 Visualization 的时候，这些碎渣非常难看，显得算法不鲁棒。
            #   不加这个参数（即允许 Cleaning），脚本会自动帮你把这些垃圾扫掉，留下干净的房间模型。
            # ============================================================

            echo "[Done] Scene $scene_name - mesh extraction (TSDF fusion) finished."
            ;;
        *)
            echo "[Error] 未知的步骤编号: $step_num (有效范围: 1-4)"
            exit 1
            ;;
    esac
    
    echo
done

echo "================================================================"
echo "All tasks finished."
echo "================================================================"
