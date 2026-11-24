#!/bin/bash
# 下载单目深度模型（Depth Anything V2 Small，CoreML F16，Apache 2.0，约 50 MB）。
# 来源：Hugging Face 上 Apple 官方仓库。模型不进 git，换机器后跑一次这个脚本。
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://huggingface.co/apple/coreml-depth-anything-v2-small/resolve/main
PKG=DepthAnythingV2SmallF16.mlpackage
DEST=Models/$PKG
mkdir -p "$DEST/Data/com.apple.CoreML/weights"
for f in Manifest.json Data/com.apple.CoreML/model.mlmodel Data/com.apple.CoreML/weights/weight.bin; do
  if [ ! -s "$DEST/$f" ]; then
    echo "下载 $f"
    curl -fL --retry 3 -o "$DEST/$f" "$BASE/$PKG/$f"
  fi
done
ls -l "$DEST/Manifest.json" "$DEST/Data/com.apple.CoreML/model.mlmodel" "$DEST/Data/com.apple.CoreML/weights/weight.bin"
