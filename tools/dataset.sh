#!/bin/bash
# 用法: ./tools/dataset.sh <录像文件> <标签> <房间种子> [扣帧相位，默认 "0 1 2 3 -"]
# 把一段录像变成训练素材：每个相位回放一遍（扣下四分之一的帧不喂给地图，回放完从这些帧的机位
# 把地图渲出来，配上这一帧的相机照片），外加 40 个随机的第三人称机位（标准答案由合成场景算）。
# 相位写 "-" 表示不扣帧、只出第三人称机位。结果放在 data/raw/。
. "$(dirname "$0")/env.sh"; need_sim
# 真实录像没有合成场景可算标准答案：种子写 real，就只出扣帧机位（相位默认 "0 1 2 3"）。
REC=$1; TAG=$2; SEED=$3
if [ "$SEED" = "real" ]; then SEED=0; TP=0; PHASES=${4:-"0 1 2 3"}; else TP=40; PHASES=${4:-"0 1 2 3 -"}; fi
# 相位写 "c" 表示不扣帧，只按 app 里跟拍镜头的算法出 ${CHASE:-80} 个机位的粗渲染（没有标准答案，蒸馏用）。
cd "$(dirname "$0")/.."
C=$(xcrun simctl get_app_container $SIM $BUNDLE data)/Documents
mkdir -p "$C/rec" data/raw
cmp -s "$REC" "$C/rec/capture.bin" || cp "$REC" "$C/rec/capture.bin"
for P in $PHASES; do
  rm -rf "$C/dataset" "$C/log.txt"
  if [ "$P" = "-" ]; then HOLD="-holdEvery 0"; NAME="${TAG}pf"; else HOLD="-holdEvery 4 -holdPhase $P -holdBlock 8"; NAME="${TAG}p$P"; fi
  VIEWS="-tpViews $TP"
  if [ "$P" = "c" ]; then HOLD="-holdEvery 0"; NAME="${TAG}pc"; VIEWS="-tpViews 0 -chaseViews ${CHASE:-80}"; fi
  xcrun simctl launch --terminate-running-process $SIM $BUNDLE \
    -replay YES -replayFast YES -noPredict YES -dataset YES $HOLD $VIEWS \
    -roomSeed $SEED -datasetTag $NAME >/dev/null
  for i in $(seq 1 600); do
    grep -q "DATASET 完成" "$C/log.txt" 2>/dev/null && break
    python3 -c "import time; time.sleep(0.5)"
  done
  grep "DATASET 完成" "$C/log.txt" | tail -1 | sed "s/^.*DATASET/$NAME DATASET/"
  cp "$C/dataset/"* data/raw/ 2>/dev/null
done
xcrun simctl terminate $SIM $BUNDLE >/dev/null 2>&1
ls data/raw | wc -l
