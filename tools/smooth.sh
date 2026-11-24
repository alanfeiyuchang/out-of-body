#!/bin/bash
# 用法: ./tools/smooth.sh <视频> [从第几秒=8] [量几秒=24]
# 量一段录屏「动起来顺不顺」。只看画面右上角一块（没有人物、没有按钮和文字）。打出两个数（0~255，越小越好）：
#   相邻帧平均差：包含镜头本身在动带来的变化，只能在镜头运动一样的录屏之间比；
#   顿挫：这一帧相对前后两帧的「加速度」（前一帧 − 2×这一帧 + 后一帧）。镜头匀速或缓慢变速时，画面平滑地动，
#         这个数很小；画面突然错一下位（换了一张对不齐的生成图、切换画面来源）它就大。看重投影对不对主要看它。
# 同一设置两次录屏会差一两成，差得不多别下结论。
V=$1; FROM=${2:-8}; LEN=${3:-24}
CROP="crop=iw*0.55:ih*0.38:iw*0.42:ih*0.10,scale=160:-2,format=gray"
A=$(ffmpeg -loglevel error -ss $FROM -t $LEN -i "$V" -vf "$CROP,tblend=all_mode=difference,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-" -f null - 2>/dev/null | grep YAVG | cut -d= -f2 | python3 -c "
import sys, statistics as st
v=[float(x) for x in sys.stdin if x.strip()][1:]
s=sorted(v); print(f'{len(v)} 帧：相邻帧平均差 {st.mean(v):.2f}（95 分位 {s[int(len(s)*0.95)]:.2f}）', end='')")
B=$(ffmpeg -loglevel error -ss $FROM -t $LEN -i "$V" -vf "$CROP,tmix=frames=3:weights='1 -2 1':scale=1,signalstats,metadata=print:key=lavfi.signalstats.YAVG:file=-" -f null - 2>/dev/null | grep YAVG | cut -d= -f2 | python3 -c "
import sys, statistics as st
v=[float(x) for x in sys.stdin if x.strip()][2:]
s=sorted(v); print(f'顿挫 {st.mean(v):.3f}（95 分位 {s[int(len(s)*0.95)]:.2f}）', end='')")
echo "$A，$B"
