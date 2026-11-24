#!/bin/bash
# 在服务器上（~/thirdeye-neural）下载并转换一个 ARKitScenes 扫描：fetch_arkit.sh <split> <video_id>
# 下完转成 arkit/scans/<id>.bin 就把压缩包删掉。苹果的条款：仅限个人、非商用。
cd "$(dirname "$0")/.."
SPLIT=$1; ID=$2
[ -s arkit/scans/$ID.bin ] && { echo "$ID 已有"; exit 0; }
B=https://docs-assets.developer.apple.com/ml-research/datasets/arkitscenes/v1/raw/$SPLIT/$ID
mkdir -p arkit/zips/$ID
for a in vga_wide.zip vga_wide_intrinsics.zip lowres_depth.zip lowres_wide.traj; do
  [ -s arkit/zips/$ID/$a ] || curl -s -f -m 1200 -o arkit/zips/$ID/$a $B/$a || { echo "$ID 下载 $a 失败"; rm -rf arkit/zips/$ID; exit 1; }
done
${PYTHON:-python3} code/prep_arkit.py arkit/zips/$ID arkit/scans/$ID.tmp > arkit/scans/$ID.log 2>&1 \
  && mv arkit/scans/$ID.tmp arkit/scans/$ID.bin && rm -rf arkit/zips/$ID && echo "$ID 完成：$(tail -1 arkit/scans/$ID.log)" \
  || { echo "$ID 转换失败：$(tail -1 arkit/scans/$ID.log)"; }
