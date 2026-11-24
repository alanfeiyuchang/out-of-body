# 各脚本共用的设置，由别的脚本 source 进去，不单独运行。
# 自己的模拟器、真机、包名写在 tools/env.local.sh（不进 git），或者用环境变量盖掉：
#   OOB_SIM=<模拟器 UDID>  OOB_DEVICE=<真机 UDID>  OOB_BUNDLE=<包名>
[ -f "$(dirname "${BASH_SOURCE[0]}")/env.local.sh" ] && . "$(dirname "${BASH_SOURCE[0]}")/env.local.sh"
# 没指定模拟器就用现在开着的第一个。
SIM=${OOB_SIM:-$(xcrun simctl list devices booted 2>/dev/null | grep -oE '[0-9A-F]{8}-[0-9A-F-]{27}' | head -1)}
BUNDLE=${OOB_BUNDLE:-com.example.outofbody}
DEVICE=${OOB_DEVICE:-}
need_sim() {
  [ -n "$SIM" ] || { echo "没有开着的模拟器：先开一个，或者设 OOB_SIM=<UDID>"; exit 1; }
}
