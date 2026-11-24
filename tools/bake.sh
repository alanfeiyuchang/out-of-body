#!/bin/bash
# 用法: ./tools/bake.sh <原始录像（prep_arkit.py 的输出）> <输出录像>
# 外部数据集转来的原始录像没有单目深度：在模拟器里用真的深度模型补上，存成和真机录的一样的录像。
. "$(dirname "$0")/env.sh"; need_sim
RAW=$1; OUT=$2
cd "$(dirname "$0")/.."
C=$(xcrun simctl get_app_container $SIM $BUNDLE data)/Documents
mkdir -p "$C/rec"
cp "$RAW" "$C/rec/import.bin"
rm -f "$C/log.txt" "$C/rec/capture.bin"
xcrun simctl launch --terminate-running-process $SIM $BUNDLE \
  -replay YES -replayFast YES -replayFile import.bin -record YES -bakeOnly YES -noPredict YES >/dev/null
for i in $(seq 1 1200); do
  grep -q "REC 结束" "$C/log.txt" 2>/dev/null && break
  python3 -c "import time; time.sleep(0.5)"
done
grep -E "REPLAY 结束|REC 结束" "$C/log.txt" | sed 's/^\[[^]]*\] //' | tr '\n' ' '; echo
xcrun simctl terminate $SIM $BUNDLE >/dev/null 2>&1
cp "$C/rec/capture.bin" "$OUT" && rm -f "$C/rec/import.bin"
