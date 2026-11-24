#!/bin/bash
# 重新生成工程、给模拟器编一份并装进模拟器。编不过就打出错误并停。
. "$(dirname "$0")/env.sh"; need_sim
cd "$(dirname "$0")/.."
xcodegen generate >/dev/null || exit 1
LOG=$(mktemp)
xcodebuild -project OutOfBody.xcodeproj -scheme OutOfBody -configuration Debug \
  -destination "id=$SIM" -derivedDataPath build build >"$LOG" 2>&1
if ! grep -q "BUILD SUCCEEDED" "$LOG"; then
  grep -E "error" "$LOG" | sort -u | head -40
  echo "BUILD FAILED"
  rm -f "$LOG"
  exit 1
fi
grep -E "warning: " "$LOG" | grep -v -E "appintents|Metadata extraction" | sort -u | head -10
rm -f "$LOG"
xcrun simctl install $SIM build/Build/Products/Debug-iphonesimulator/OutOfBody.app && echo installed
