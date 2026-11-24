"""「粗渲染 → 干净画面」的小网络：一个带跳连的 U-Net，输出的是对输入颜色的修正量。

要在手机上实时跑，所以很小：默认约 100 万参数，全是 3×3 卷积和 ReLU，没有归一化层
（CoreML 上最省事，也不会在整屏都是洞的时候算出奇怪的统计量）。
往下缩 4 次，最深一层再叠两个空洞卷积块，让它看得见洞周围一大圈的内容。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.ReLU(inplace=True))


class Refiner(nn.Module):
    """depth = 往下缩几次（4 → 最深一层是 1/16，5 → 1/32）。
    context = 最深一层再加「整张图的平均特征」：大片没扫到的地方离有内容的地方很远，
    只靠卷积一格一格传不过去，补出来是一道道竖纹；有了整张图的平均，它至少知道这间屋子大体是什么颜色。
    真机上 base 32 / depth 4 一张 14 毫秒，有余量换更大的。"""

    def __init__(self, cin=5, base=24, depth=4, context=False):
        super().__init__()
        mult = [1, 2, 4, 6, 8, 8][: depth + 1]
        c = [base * m for m in mult]
        self.depth = depth
        self.context = context
        self.enc = nn.ModuleList([block(cin, c[0])] + [block(c[i], c[i + 1]) for i in range(depth)])
        self.mid = nn.Sequential(
            nn.Conv2d(c[-1], c[-1], 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(c[-1], c[-1], 3, padding=4, dilation=4), nn.ReLU(inplace=True))
        if context:
            self.ctx = nn.Sequential(nn.Conv2d(c[-1], c[-1], 1), nn.ReLU(inplace=True))
            self.mid2 = block(c[-1], c[-1])
        self.dec = nn.ModuleList([block(c[i + 1] + c[i], c[i]) for i in range(depth - 1, -1, -1)])
        self.out = nn.Conv2d(c[0], 3, 3, padding=1)

    def forward(self, x):
        skips = []
        h = x
        for i, e in enumerate(self.enc):
            if i > 0:
                h = F.avg_pool2d(h, 2)
            h = e(h)
            skips.append(h)
        h = self.mid(h)
        if self.context:
            h = self.mid2(h + self.ctx(h.mean(dim=(2, 3), keepdim=True)))
        for i, d in enumerate(self.dec):
            h = F.interpolate(h, scale_factor=2, mode="bilinear", align_corners=False)
            h = d(torch.cat([h, skips[self.depth - 1 - i]], dim=1))
        return (x[:, :3] + self.out(h)).clamp(0, 1)
