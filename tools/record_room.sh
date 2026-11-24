#!/bin/bash
# 用法: ./tools/record_room.sh <房间种子> [秒数，默认 75]
# 在模拟器里用真的深度模型扫一间随机生成的合成房间，录像存到 rec/synth-room<种子>.bin。
. "$(dirname "$0")/env.sh"; need_sim
SEED=$1; SECS=${2:-75}
cd "$(dirname "$0")/.."
C=$(xcrun simctl get_app_container $SIM $BUNDLE data)/Documents
rm -f "$C/rec/capture.bin"
xcrun simctl launch --terminate-running-process $SIM $BUNDLE \
  -monoReal YES -record YES -noPredict YES -noMover YES -roomSeed $SEED >/dev/null
python3 -c "import time; time.sleep($SECS)"
xcrun simctl terminate $SIM $BUNDLE >/dev/null 2>&1
python3 -c "import time; time.sleep(1)"
cp "$C/rec/capture.bin" rec/synth-room$SEED.bin && ls -la rec/synth-room$SEED.bin
