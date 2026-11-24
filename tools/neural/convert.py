"""refiner.pt → CoreML（fp16，固定 384×512 竖屏输入）。用法: python convert.py refiner.pt Refiner.mlpackage"""
import sys

import coremltools as ct
import torch

from model import Refiner

ck = torch.load(sys.argv[1], map_location="cpu")
m = Refiner(base=ck["base"], depth=ck.get("depth", 4), context=ck.get("context", False))
m.load_state_dict(ck["state"])
m.eval()
ex = torch.zeros(1, 5, 512, 384)
traced = torch.jit.trace(m, ex)
ml = ct.convert(
    traced, inputs=[ct.TensorType(name="x", shape=ex.shape)],
    outputs=[ct.TensorType(name="y")], minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16)
ml.save(sys.argv[2])
print("saved", sys.argv[2])
