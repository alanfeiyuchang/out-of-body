#!/bin/bash
# 用法: ./tools-shot.sh <名字> <等几秒> [启动参数...]
# 在已启动的 Duo 模拟器里跑合成场景，等一会儿截一张图到 shots/。
set -e
. "$(dirname "$0")/env.sh"; need_sim
NAME=$1; WAIT=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p shots
xcrun simctl launch --terminate-running-process $SIM $BUNDLE "$@" >/dev/null
python3 -c "import time; time.sleep($WAIT)"
xcrun simctl io $SIM screenshot shots/$NAME.png >/dev/null 2>&1
sips -Z 1100 shots/$NAME.png >/dev/null
echo shots/$NAME.png
