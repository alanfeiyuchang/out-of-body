"""把 ARKitScenes 的一个扫描变成 app 能回放的「原始录像」（还没有单目深度，那一步在 app 里补）。
用法: python prep_arkit.py <扫描的压缩包目录> <输出 .bin> [最多多少帧，默认 300]

数据集里有的：640×480 彩色（30 fps）、LiDAR 深度 256×192（60 fps）、位姿（10 Hz）、内参。
app 在真机上拿到的是：相机图、ARKit 位姿、几十到一两百个带深度的特征点、ARKit 认出的平面。
所以这里把 LiDAR 深度只当「特征点的深度」和「找平面」用，稠密深度仍然由 app 里的单目模型估——
这样回放走的是和真机一样的流程。注意这比真机乐观：特征点的深度更准，平面一开始就全有。

输出格式跟 Recorder 一样是一条条记录（类型 1 字节 + 长度 4 字节 + 内容，小端）：
  类型 1 = 位姿（和 Recorder 的完全一样）
  类型 3 = 一帧原始材料：时间 f64，相机到世界 16×f32（列优先，ARKit 约定：相机看 -z、+y 朝上），
           fx fy cx cy（相机图分辨率下），对照点个数 + 每个 (u, v 归一化到 0~1, z 米)，
           平面个数 + 每个 (中心 xyz, 法线 xyz, 半径)，图宽、图高、JPEG 长度、JPEG
世界坐标转成 ARKit 的习惯：y 朝上，地面在 y = 0。
"""
import io
import struct
import sys
import zipfile

import numpy as np
from PIL import Image

src, out_path = sys.argv[1:3]
max_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 300
rng = np.random.default_rng(0)


def stamp(name):
    return float(name.rsplit("_", 1)[1].rsplit(".", 1)[0])


def rodrigues(v):
    theta = np.linalg.norm(v)
    if theta < 1e-9:
        return np.eye(3)
    k = v / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K


# ---- 位姿：文件里是「世界到相机」（OpenCV 约定：x 右、y 下、z 前），取逆得到相机到世界 ----
poses = []
for line in open(f"{src}/lowres_wide.traj"):
    t = [float(x) for x in line.split()]
    m = np.eye(4)
    m[:3, :3] = rodrigues(np.array(t[1:4]))
    m[:3, 3] = t[4:7]
    poses.append((t[0], np.linalg.inv(m)))
pose_times = np.array([p[0] for p in poses])

zv = zipfile.ZipFile(f"{src}/vga_wide.zip")
zd = zipfile.ZipFile(f"{src}/lowres_depth.zip")
zi = zipfile.ZipFile(f"{src}/vga_wide_intrinsics.zip")
rgb_names = sorted((n for n in zv.namelist() if n.endswith(".png")), key=stamp)
depth_names = sorted((n for n in zd.namelist() if n.endswith(".png")), key=stamp)
intr_names = sorted((n for n in zi.namelist() if n.endswith(".pincam")), key=stamp)
rgb_times = np.array([stamp(n) for n in rgb_names])
depth_times = np.array([stamp(n) for n in depth_names])
intr_times = np.array([stamp(n) for n in intr_names])


def nearest(times, t):
    i = int(np.clip(np.searchsorted(times, t), 1, len(times) - 1))
    return i - 1 if t - times[i - 1] < times[i] - t else i


# ---- 选帧：位姿是 10 Hz，隔一个取一个（5 fps）；扫描太长就再拉大间隔 ----
stride = max(2, int(np.ceil(len(poses) / max_frames)))
frames = []
for k in range(0, len(poses), stride):
    t, c2w = poses[k]
    i, j = nearest(rgb_times, t), nearest(depth_times, t)
    if abs(rgb_times[i] - t) > 0.03 or abs(depth_times[j] - t) > 0.03:
        continue
    frames.append((t, c2w, rgb_names[i], depth_names[j], intr_names[nearest(intr_times, rgb_times[i])]))


def load_depth(name):
    return np.asarray(Image.open(io.BytesIO(zd.read(name))), dtype=np.float32) / 1000


def intrinsics(name):
    w, h, fx, fy, cx, cy = [float(x) for x in zi.read(name).decode().split()]
    return w, h, fx, fy, cx, cy


def unproject(depth, K, c2w, step=4):
    """深度图 → 世界坐标的点和法线（朝着相机）。K 是深度图分辨率下的 fx fy cx cy。"""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (u + 0.5 - K[2]) / K[0] * depth
    y = (v + 0.5 - K[3]) / K[1] * depth
    cam = np.stack([x, y, depth], axis=-1)
    dx = cam[1:-1, 2:] - cam[1:-1, :-2]
    dy = cam[2:, 1:-1] - cam[:-2, 1:-1]
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    p = cam[1:-1, 1:-1]
    n = np.where((n * p).sum(-1, keepdims=True) > 0, -n, n)
    ok = (depth[1:-1, 1:-1] > 0.3) & (depth[1:-1, 1:-1] < 4.5)
    # 深度跳变的地方法线不可信。
    ok &= (np.abs(dx[..., 2]) < 0.06) & (np.abs(dy[..., 2]) < 0.06)
    p, n, ok = p[::step, ::step], n[::step, ::step], ok[::step, ::step]
    p, n = p[ok], n[ok]
    return p @ c2w[:3, :3].T + c2w[:3, 3], n @ c2w[:3, :3].T


def depth_K(intr):
    w, h, fx, fy, cx, cy = intr
    s = 256 / w
    return fx * s, fy * s, cx * s, cy * s


# ---- 一致性检查：把一帧的点投到几帧之后的深度图里，深度应该对得上 ----
errs = []
for a in range(0, len(frames) - 5, max(1, len(frames) // 12)):
    fa, fb = frames[a], frames[a + 4]
    pa, _ = unproject(load_depth(fa[3]), depth_K(intrinsics(fa[4])), fa[1])
    Kb = depth_K(intrinsics(fb[4]))
    db = load_depth(fb[3])
    w2c = np.linalg.inv(fb[1])
    pc = pa @ w2c[:3, :3].T + w2c[:3, 3]
    z = pc[:, 2]
    u = np.round(pc[:, 0] / z * Kb[0] + Kb[2] - 0.5).astype(int)
    v = np.round(pc[:, 1] / z * Kb[1] + Kb[3] - 0.5).astype(int)
    ok = (z > 0.3) & (u >= 0) & (u < 256) & (v >= 0) & (v < 192)
    d = db[v[ok], u[ok]]
    good = d > 0.3
    if good.sum() > 100:
        errs.append(np.median(np.abs(d[good] - z[ok][good])))
print(f"帧间深度一致性：中位误差 {100 * np.median(errs):.1f} cm（位姿约定对的话应该是一两厘米）")

# ---- 找「上」和地面高度 ----
pts, nrm = [], []
for f in frames[:: max(1, len(frames) // 60)]:
    p, n = unproject(load_depth(f[3]), depth_K(intrinsics(f[4])), f[1])
    pts.append(p)
    nrm.append(n)
pts, nrm = np.concatenate(pts), np.concatenate(nrm)
cams = np.array([f[1][:3, 3] for f in frames])
# 判断依据有两条，要同时满足才像「上」：
#   1. 很多面的法线沿着这条轴（地面、天花板、桌面）；但墙要是正好和坐标轴对齐也会这样，所以还要
#   2. 相机沿这条轴几乎不动（人拿着设备走，高度变化只有一二十厘米，水平方向是好几米）。
share = np.array([(np.abs(nrm[:, a]) > 0.95).mean() for a in range(3)])
spread = cams.std(axis=0)
guess = int(np.argmax(share / (spread + 0.1)))
# ARKitScenes 的世界坐标统一是 z 朝上。上面那个判断在墙和坐标轴对齐的房间里分不太清
# （第一个扫描里 y 和 z 的得分只差一成），所以只拿来核对，不一致就提醒一声。
axis = 2
if guess != axis:
    print(f"注意：按面和相机起伏猜的「上」是 {'xyz'[guess]}，和数据集约定的 z 不一致")
flat = np.abs(nrm[:, axis]) > 0.95
# 正负号：地面是「在相机下面、法线朝上」的面。两个方向各数一数，哪边多哪边是上。
cam_mid = np.median(cams[:, axis])
plus = (nrm[:, axis] > 0.95) & (pts[:, axis] < cam_mid - 0.4)
minus = (nrm[:, axis] < -0.95) & (pts[:, axis] > cam_mid + 0.4)
# 数据集约定是 +z 朝上。「下面朝上的面多」这个判断在几乎只拍墙和天花板的扫描里会反
# （有一个卫生间的扫描就被判成了 -z，整个世界倒了过来），所以也只拿来提醒。
sign = 1.0
if minus.sum() > plus.sum():
    print("注意：按「相机下方朝上的面」数出来像是 -z 朝上，仍按数据集约定的 +z")
up = np.zeros(3)
up[axis] = sign
facing_up = flat & (nrm[:, axis] * sign > 0.95) & (pts[:, axis] * sign < cam_mid * sign - 0.4)
# 没怎么拍到地面的话，按「手持高度大约 1.4 m」估一个。
floor = np.median(pts[facing_up, axis] * sign) if facing_up.sum() > 200 else cam_mid * sign - 1.4
print(f"上 = {'+' if sign > 0 else '-'}{'xyz'[axis]}（各轴上的面占比 {np.round(share, 2)}，相机位置的起伏 {np.round(spread, 2)} m），"
      f"相机离地 {cam_mid * sign - floor:.2f} m")

# 世界 → ARKit 习惯（y 朝上，地面 y = 0）。
a = np.eye(3)[(axis + 1) % 3]
b = np.cross(up, a)
R = np.stack([a, up, b])  # 行向量：新 x、新 y、新 z
if np.linalg.det(R) < 0:
    R[2] *= -1
T = np.eye(4)
T[:3, :3] = R
T[1, 3] = -floor
flip = np.diag([1.0, -1.0, -1.0, 1.0])  # OpenCV 相机轴 → ARKit 相机轴
pts = pts @ R.T + T[:3, 3]
nrm = nrm @ R.T

# ---- 平面：地面 + 支持点最多的几面墙（ARKit 的平面检测给的也是这些） ----
planes = [((float(cams[:, 0].mean()), 0.0, float(cams[:, 2].mean())) if False else None)]
planes = []
cx_, cz_ = (pts[:, 0].mean(), pts[:, 2].mean())
planes.append(((cx_, 0.0, cz_), (0.0, 1.0, 0.0), 5.0))
wall = np.abs(nrm[:, 1]) < 0.15
wp, wn = pts[wall], nrm[wall]
az = np.arctan2(wn[:, 2], wn[:, 0])
hist, edges = np.histogram(az, bins=72, range=(-np.pi, np.pi))
found = []
for _ in range(4):
    k = int(np.argmax(hist))
    if hist[k] < 0.04 * len(az):
        break
    theta = (edges[k] + edges[k + 1]) / 2
    d = np.array([np.cos(theta), 0, np.sin(theta)])
    sel = (wn @ d) > np.cos(np.radians(12))
    off = wp[sel] @ d
    oh, oe = np.histogram(off, bins=np.arange(off.min() - 0.05, off.max() + 0.1, 0.05))
    for j in np.argsort(oh)[::-1][:2]:
        if oh[j] < 0.25 * oh.max() or oh[j] < 300:
            continue
        o = (oe[j] + oe[j + 1]) / 2
        on = sel & (np.abs(wp @ d - o) < 0.06)
        q = wp[on]
        c = q.mean(0)
        c = c - d * (c @ d - o)
        radius = float(min(4.0, np.percentile(np.linalg.norm(q - c, axis=1), 90)))
        if radius > 0.6:
            found.append((int(on.sum()), (tuple(c), tuple(d), radius)))
    for dk in (-2, -1, 0, 1, 2):
        hist[(k + dk) % 72] = 0
found.sort(reverse=True)
planes += [f[1] for f in found[:6]]
print(f"平面：地面 + {len(planes) - 1} 面墙")


def record(tag, payload):
    return struct.pack("<BI", tag, len(payload)) + payload


def matrix_bytes(m):
    return struct.pack("<16f", *m.T.reshape(-1))  # 列优先


out = open(out_path, "wb")
count = 0
frame_times = [f[0] for f in frames]
for k, (t, c2w, rgb_name, depth_name, intr_name) in enumerate(frames):
    # 这一帧之前的那些位姿（10 Hz）先写出去，人偶动起来顺一点。
    t_prev = frame_times[k - 1] if k > 0 else -1
    for pt, pm in poses:
        if t_prev < pt <= t:
            cam = T @ pm @ flip
            out.write(record(1, struct.pack("<d", pt) + matrix_bytes(cam) + b"\x00"
                             + matrix_bytes(np.eye(4)) + b"\x01" + struct.pack("<f", 0.0)))
    cam = T @ c2w @ flip
    w, h, fx, fy, cx, cy = intrinsics(intr_name)
    image = Image.open(io.BytesIO(zv.read(rgb_name))).convert("RGB")
    depth = load_depth(depth_name)
    # 对照点：ARKit 的特征点只长在有纹理的地方。挑明暗变化大的像素，深度取 LiDAR 的。
    gray = np.asarray(image.convert("L").resize((256, 192), Image.BILINEAR), dtype=np.float32)
    gx = np.abs(gray[2:-2, 4:] - gray[2:-2, :-4])
    gy = np.abs(gray[4:, 2:-2] - gray[:-4, 2:-2])
    g = np.maximum(gx, gy)
    d = depth[2:-2, 2:-2]
    cand = np.argwhere((g > 22) & (d > 0.3) & (d < 5.0))
    want = int(rng.integers(40, 150))
    samples = []
    if len(cand) > 0:
        for v, u in cand[rng.choice(len(cand), size=min(want, len(cand)), replace=False)]:
            z = float(d[v, u]) * (1 + 0.03 * (rng.random() - 0.5))
            if rng.random() < 1 / 12:
                z *= 1.4  # 真的特征点里也有错的
            samples.append(((u + 2.5) / 256, (v + 2.5) / 192, z))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    jpeg = buf.getvalue()
    p = struct.pack("<d", t) + matrix_bytes(cam) + struct.pack("<4f", fx, fy, cx, cy)
    p += struct.pack("<I", len(samples)) + b"".join(struct.pack("<3f", *s) for s in samples)
    p += struct.pack("<I", len(planes))
    for c, n, r in planes:
        p += struct.pack("<7f", *c, *n, r)
    p += struct.pack("<III", int(w), int(h), len(jpeg)) + jpeg
    out.write(record(3, p))
    count += 1
out.close()
print(f"写了 {count} 帧（每 {stride} 个位姿取一个，约 {10 / stride:.1f} fps）→ {out_path}")
