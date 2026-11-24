#!/bin/bash
# 给真机编一份并装上（要先在 tools/env.local.sh 里写 OOB_DEVICE，在 Config/Local.xcconfig 里写签名）。手机锁着也能装，但启动要先解锁。
. "$(dirname "$0")/env.sh"
UDID=$DEVICE
[ -n "$UDID" ] || { echo "没指定真机：设 OOB_DEVICE=<UDID>（xcrun devicectl list devices 能查到）"; exit 1; }
cd "$(dirname "$0")/.."
xcodegen generate >/dev/null || exit 1
LOG=$(mktemp)
xcodebuild -project OutOfBody.xcodeproj -scheme OutOfBody -configuration Debug \
  -destination "id=$UDID" -derivedDataPath build -allowProvisioningUpdates build >"$LOG" 2>&1
if ! grep -q "BUILD SUCCEEDED" "$LOG"; then
  grep -E "error" "$LOG" | sort -u | head -40
  echo "BUILD FAILED"
  rm -f "$LOG"
  exit 1
fi
rm -f "$LOG"
xcrun devicectl device install app --device $UDID \
  build/Build/Products/Debug-iphoneos/OutOfBody.app 2>&1 | tail -3
