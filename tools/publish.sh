#!/bin/bash
# 用法: tools/publish.sh <目标目录>
# 把这个仓库导出成一份可以公开的副本：只带已提交的文件，去掉不能再分发的第三方素材，
# 再查一遍有没有漏掉的私人信息（本机配置文件里写的那些值、用户名）。
# 目标目录是另一个独立的 git 仓库（历史是干净的），这个脚本只更新它的文件，不提交、不推送。
set -euo pipefail
cd "$(dirname "$0")/.."
DEST=${1:?用法: tools/publish.sh <目标目录>}
[ -z "$(git status --porcelain)" ] || { echo "有没提交的改动，先提交再导出"; exit 1; }
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
[ "$DEST" != "$(pwd)" ] || { echo "目标不能是这个仓库自己"; exit 1; }
# 只往空目录或者上次导出过的目录里写，免得删错东西。
if [ -n "$(ls -A "$DEST")" ] && [ ! -f "$DEST/.public-export" ]; then
  echo "$DEST 不是空的，也不是上次导出的目录，不动它"; exit 1
fi
find "$DEST" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
git archive HEAD | tar -x -C "$DEST"
touch "$DEST/.public-export"

# 不公开的：第三方模型和从它们烘焙出来的文件（许可见 THIRD_PARTY.md），只有作者自己用得上的脚本。
for p in assets OutOfBody/Resources tools/monodbg.sh tools/bake_character.py tools/bake_static.py; do
  rm -rf "$DEST/$p"
done

# 查私人信息：本机配置里的每个值，加上用户名。
PATTERNS=$( { grep -hoE '=[^#/]*' Config/Local.xcconfig tools/env.local.sh 2>/dev/null \
    | tr ',=' '\n\n' | sed -E 's/[${}:-]+[A-Z_]*:-//; s/^[ ]+//; s/[ }]+$//' | grep -E '.{6,}' ; whoami; } | sort -u)
BAD=0
while IFS= read -r pat; do
  [ -n "$pat" ] || continue
  if grep -rnIF --exclude-dir=.git -- "$pat" "$DEST" | head -3 | grep .; then BAD=1; fi
done <<< "$PATTERNS"
[ $BAD = 0 ] || { echo "↑ 导出的文件里还有私人信息，先处理"; exit 1; }
echo "导出到 $DEST：$(cd "$DEST" && find . -path ./.git -prune -o -type f -print | wc -l | tr -d ' ') 个文件，$(du -sh "$DEST" | cut -f1)"
