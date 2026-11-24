#!/bin/bash
ffmpeg -loglevel error -ss $2 -i "$1" -vf "fps=10,scale=70:100,format=rgb24" -f rawvideo - | python3 -I -c "
import sys
d=sys.stdin.buffer.read(); n=70*100*3; fr=[]
for k in range(len(d)//n):
    f=d[k*n:(k+1)*n]; c=0
    for i in range(0,n,3):
        c+=max(0,min(f[i],f[i+2])-f[i+1])
    fr.append(c/7000/255)
print(' '.join(f'{int(x*100):2d}' for x in fr))
print(f'  点云露出的量：平均 {100*sum(fr)/len(fr):.1f}%  最大 {100*max(fr):.0f}%  超过 5% 的帧 {100*sum(x>0.05 for x in fr)/len(fr):.0f}%')
"
