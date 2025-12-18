#!/bin/bash

# 全自动 git 提交脚本
# 用法: ./auto_commit.sh [提交消息]

cd "$(dirname "$0")"

# 检查是否有更改
if [ -z "$(git status --porcelain)" ]; then
    echo "没有需要提交的更改"
    exit 0
fi

# 生成提交消息
if [ -z "$1" ]; then
    # 如果没有提供提交消息，自动生成
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    CHANGES=$(git status --short | head -5 | sed 's/^/  - /')
    COMMIT_MSG="自动提交 - $TIMESTAMP

更改的文件:
$CHANGES"
else
    COMMIT_MSG="$1"
fi

# 添加所有更改（包括删除的文件）
git add -A

# 提交
git commit -m "$COMMIT_MSG"

# 显示提交结果
echo "✅ 提交成功!"
echo "📝 提交消息:"
echo "$COMMIT_MSG"
echo ""
git log -1 --stat

