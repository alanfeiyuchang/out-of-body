"""给一个训练好的 Refiner 在指定房间的扣帧机位上打分。用法: python eval_refiner.py refiner.pt <素材目录> v01,v02"""
import sys

import torch

from data import list_bases, load_raw
from model import Refiner
from train import evaluate

ck = torch.load(sys.argv[1], map_location="cuda")
model = Refiner(base=ck["base"], depth=ck.get("depth", 4), context=ck.get("context", False)).cuda()
model.load_state_dict(ck["state"])
items = [load_raw(b) for b in list_bases(sys.argv[2], sys.argv[3].split(","), "h")][::3]
a, b = evaluate(model, items, "cuda")
print(f"{sys.argv[3]}（{len(items)} 张）粗渲染 {a:.2f} dB → 网络 {b:.2f} dB")
