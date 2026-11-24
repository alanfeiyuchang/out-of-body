"""读 DatasetExporter 存下来的素材：<名字>_in.png / _d.bin / _gt.png。"""
import glob
import os

import numpy as np
import torch
from PIL import Image


def load_raw(base, target="gt"):
    """读一组素材并摆正，原样返回（省内存的格式）：颜色 uint8、标准答案 uint8、距离 float16。

    target="gt" 用相机照片当答案；"t" 用老师模型的输出（teacher_label.py 存的 _t.png，已经是摆正的）；
    None 不读答案（给老师模型准备输入时用）。

    相机照片是按传感器方向存的。导出时算好了要逆时针转几个 90°（_up.txt）；
    早先导出的合成素材没有这个文件，它们一律是「图像右边 = 世界下方」。
    竖着拍的摆正后是 384×512，横着拍的是 512×384。
    """
    rgb = np.asarray(Image.open(base + "_in.png").convert("RGB"))
    h, w = rgb.shape[:2]
    gt = np.asarray(Image.open(base + "_gt.png").convert("RGB")) if target == "gt" else rgb
    dist = np.fromfile(base + "_d.bin", dtype=np.float32).reshape(h, w)
    turns = 3 if w > h else 0
    if os.path.exists(base + "_up.txt"):
        turns = int(open(base + "_up.txt").read().strip())
    if turns:
        rgb, gt, dist = (np.rot90(a, k=turns) for a in (rgb, gt, dist))
    if target == "t":
        gt = np.asarray(Image.open(base + "_t.png").convert("RGB"))
    return (np.ascontiguousarray(rgb), np.ascontiguousarray(gt),
            np.ascontiguousarray(dist).astype(np.float16))


def to_tensors(raw):
    """(输入 5 通道, 标准答案 3 通道)，float32 的 CHW。输入 = 粗渲染的 RGB + 有没有东西 + 逆距离。"""
    rgb, gt, dist = raw
    dist = dist.astype(np.float32)
    mask = (dist > 0).astype(np.float32)
    inv = np.where(dist > 0, 1 / np.maximum(dist, 0.3), 0).astype(np.float32)
    x = np.concatenate(
        [rgb.astype(np.float32) / 255 * mask[..., None], mask[..., None], inv[..., None]], axis=2)
    y = gt.astype(np.float32) / 255
    return torch.from_numpy(x).permute(2, 0, 1), torch.from_numpy(y).permute(2, 0, 1)


def load_pair(base):
    return to_tensors(load_raw(base))


def list_bases(root, rooms=None, kinds="ht"):
    """rooms: 比如 ["r0", "r1"]；kinds: "h" = 扣帧机位，"t" = 合成房间的第三人称机位，
    "c" = 按 app 跟拍镜头取的机位（没有相机照片当答案，只能配老师模型的输出）。"""
    out = []
    for p in sorted(glob.glob(os.path.join(root, "*_in.png"))):
        name = os.path.basename(p)[: -len("_in.png")]
        tag, view = name.rsplit("_", 1)
        room = tag.split("p")[0]
        if rooms is not None and room not in rooms:
            continue
        if view[0] not in kinds:
            continue
        out.append(p[: -len("_in.png")])
    return out
