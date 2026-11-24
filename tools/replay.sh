#!/bin/bash
# 用法: ./tools/replay.sh <录像文件> [截图名] [启动参数...]
# 在模拟器里快放一段录像，打出结果那一行（合成场景的录像带误差分数）；
# 给了截图名就再截一张俯视图和一张跟拍图到 shots/。
. "$(dirname "$0")/env.sh"; need_sim
REC=$1; NAME=$2; shift; shift
cd "$(dirname "$0")/.."
C=$(xcrun simctl get_app_container $SIM $BUNDLE data)/Documents
mkdir -p "$C/rec" shots
cmp -s "$REC" "$C/rec/capture.bin" || cp "$REC" "$C/rec/capture.bin"
run() {
  rm -f "$C/log.txt"
  xcrun simctl launch --terminate-running-process $SIM $BUNDLE \
    -replay YES -replayFast YES -noPredict YES "$@" >/dev/null
  for i in $(seq 1 240); do
    grep -q "REPLAY 结束" "$C/log.txt" 2>/dev/null && break
    python3 -c "import time; time.sleep(0.5)"
  done
}
run -evalSynthetic YES "$@"
grep "REPLAY 结束" "$C/log.txt" | tail -1 | sed 's/^.*REPLAY/REPLAY/'
if [ -n "$NAME" ] && [ "$NAME" != "-" ]; then
  python3 -c "import time; time.sleep(1.5)"
  xcrun simctl io $SIM screenshot shots/$NAME-chase.png >/dev/null 2>&1
  run -pitch 1.34 -dist 9 "$@"
  python3 -c "import time; time.sleep(1.5)"
  xcrun simctl io $SIM screenshot shots/$NAME-top.png >/dev/null 2>&1
  sips -Z 1100 shots/$NAME-chase.png shots/$NAME-top.png >/dev/null
  echo shots/$NAME-chase.png shots/$NAME-top.png
fi
