#!/bin/bash
# 用法: ./tools/video.sh <名字> <录几秒> [启动参数...]
# 在模拟器里按原速回放 rec 里装好的录像（先用 replay.sh 或手动拷进去），同时录屏到 shots/vid/<名字>.mp4。
# 用来看「动起来」的效果：帧与帧之间跳不跳、拖动时糊不糊。录完打出一个流畅度数字（见 tools/smooth.sh）。
. "$(dirname "$0")/env.sh"; need_sim
NAME=$1; SECS=$2; shift 2
cd "$(dirname "$0")/.."
mkdir -p shots/vid
rm -f shots/vid/$NAME.mp4
xcrun simctl launch --terminate-running-process $SIM $BUNDLE -replay YES -noPredict YES "$@" >/dev/null
xcrun simctl io $SIM recordVideo --codec h264 --force shots/vid/$NAME.mp4 >/dev/null 2>&1 &
REC=$!
python3 -c "import time; time.sleep($SECS)"
kill -INT $REC; wait $REC 2>/dev/null
xcrun simctl terminate $SIM $BUNDLE >/dev/null 2>&1
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,avg_frame_rate,nb_frames -of csv=p=0 shots/vid/$NAME.mp4
