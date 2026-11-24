"""给带骨骼的人物模型做三段动作（待机、走路、坐下），逐帧烘焙成 app 直接播放的网格。
用法: /Applications/Blender.app/Contents/MacOS/Blender -b --python tools/bake_animated.py -- \
        <带人形骨骼名的模型.glb> <输出.bin> <输出贴图.png> [预览图目录]

模型：服务器上 SkinTokens 那套流程的产物（SkinTokens 自动绑骨骼，humanoid_walk.py 按人形给骨骼起名：
Hips、LeftUpperLeg、RightLowerArm……）。

动作不是从别的模型搬来的（试过把 KayKit 骑士的走路套上来：那是大头短腿的卡通角色，套到真人比例上像在高抬腿跑），
而是按真人步态直接算：
  - 腿：先定脚在地上怎么走（踩地的那段相对身体匀速往后，所以脚不打滑；离地的那段抬起来往前摆；
    落地脚跟先着地，蹬地时脚跟先抬起、绕前脚掌转），再定胯多高（双脚都着地时最低，单脚撑到正中时最高），
    大腿小腿用两段骨头的反解算出来；
  - 左臂跟对侧的腿反着摆；右臂固定成举着手机的姿势；头微微低着看手机；
  - 待机：双脚与胯同宽站稳，轻微的呼吸起伏和重心晃动；
  - 坐下：脚不动，胯往后往下坐到小腿那么高，中途上身前倾，坐稳后左手搭在腿上。
app 里不做骨骼蒙皮：每一帧的顶点位置和法线都存下来，播放时在相邻两帧之间插值。

输出格式（小端）："TEC3"，顶点数、索引数、待机帧数、走路帧数、坐下帧数（都是 u32），
  身高、走一个循环前进多远、坐下时胯往后挪多远（都是 f32，模型单位），
  每一帧手机的位置（总帧数 × xyz f32），uv（顶点数 × 2 个 f32），索引（u32），
  然后一帧一帧的顶点（位置 3 + 法线 3，f32）：先待机循环，再走路循环（都首尾相接），最后坐下（第 0 帧站着，最后一帧坐稳）。
坐标：x 朝人物的左边，y 朝上，脚底 y = 0，脸朝 +z，站着时胯在 x = z = 0 的正上方。
"""
import math
import struct
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

args = sys.argv[sys.argv.index("--") + 1:]
src, out, out_png = args[:3]
preview = args[3] if len(args) > 3 else None
IDLE, WALK, SIT = 8, 16, 6

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")
meshes = [o for o in bpy.data.objects if o.type == "MESH" and o.find_armature() is not None]
# 模型自带的动作全部摘掉（glb 导入后动作挂在 NLA 轨道上，不摘的话每次刷新场景它都会把手摆的姿势盖掉）。
for o in bpy.data.objects:
    o.animation_data_clear()
scene = bpy.context.scene


def snapped(v):
    i = max(range(3), key=lambda k: abs(v[k]))
    out_v = Vector((0, 0, 0))
    out_v[i] = 1 if v[i] > 0 else -1
    return out_v


def rest_head(name):
    return arm.matrix_world @ arm.data.bones[name].head_local


# 人物的上、左、前各是世界坐标的哪条轴，从骨架自己量，不能假设「z 朝上」：
# 自动绑骨骼的工具会丢掉原模型外面那层旋转，这个 Optimus 出来是「躺着」的（头朝 -Y）。
UP = snapped(rest_head("Head") - rest_head("Hips"))
LEFT = snapped(rest_head("LeftUpperArm") - rest_head("RightUpperArm"))
FWD = LEFT.cross(UP)
W = arm.matrix_world.to_3x3()
WI = arm.matrix_world.inverted().to_3x3()
print("人物朝向：上", tuple(UP), "左", tuple(LEFT), "前", tuple(FWD))


def convert(v):  # 世界坐标 → 人物坐标（x 朝人物的左边，y 朝上，z 朝前）
    return Vector((v.dot(LEFT), v.dot(UP), v.dot(FWD)))


def world(v):  # 人物坐标 → 世界坐标（只转方向）
    return LEFT * v.x + UP * v.y + FWD * v.z


def joint(name):  # 当前姿势下某个关节的位置（人物坐标）
    return convert(arm.matrix_world @ arm.pose.bones[name].head)


def rest(name):
    return convert(rest_head(name))


def rot(axis, degrees):  # 人物坐标里绕 axis 转
    return Quaternion(Vector(axis), math.radians(degrees))


def reset():
    for pb in arm.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion()
        pb.location = Vector()
    bpy.context.view_layer.update()


def set_rot(name, q):
    """让一根骨头的朝向 = 静止朝向再转 q（q 在人物坐标里）。位置跟着父骨头，不动。"""
    pb = arm.pose.bones[name]
    m = Matrix((LEFT, UP, FWD)).transposed()  # 人物坐标 → 世界
    delta = WI @ m @ q.to_matrix() @ m.transposed() @ W
    h = pb.matrix.translation.copy()
    pb.matrix = Matrix.Translation(h) @ (delta @ arm.data.bones[name].matrix_local.to_3x3()).to_4x4()
    bpy.context.view_layer.update()


def point(name, child, direction, twist=0.0):
    """让「这根骨头的根 → 下一个关节」指向 direction（人物坐标），从静止姿势走最短的转法，再绕自身拧 twist 度。"""
    d = Vector(direction).normalized()
    q = (rest(child) - rest(name)).normalized().rotation_difference(d)
    set_rot(name, Quaternion(d, math.radians(twist)) @ q)


def ik(root, target, l1, l2, pole):
    d = target - root
    reach = min(max(d.length, abs(l1 - l2) + 1e-4), (l1 + l2) * 0.999)
    direction = d.normalized()
    x = (l1 * l1 - l2 * l2 + reach * reach) / (2 * reach)
    h = math.sqrt(max(0.0, l1 * l1 - x * x))
    perp = (pole - direction * pole.dot(direction)).normalized()
    return root + direction * x + perp * h, root + direction * reach


def smooth(a, b, x):
    t = min(1.0, max(0.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def keys(points, p):
    """分段平滑插值；points 是 (相位, 值)，按相位排好，首尾相位是 0 和 1。"""
    for (p0, v0), (p1, v1) in zip(points, points[1:]):
        if p <= p1:
            return v0 + (v1 - v0) * smooth(p0, p1, p)
    return points[-1][1]


# ---- 静止姿势里量尺寸 ----
reset()


def sample():
    deps = bpy.context.evaluated_depsgraph_get()
    pos, nrm = [], []
    for obj in meshes:
        ev = obj.evaluated_get(deps)
        mesh = ev.to_mesh()
        mesh.calc_loop_triangles()
        mw = ev.matrix_world
        nm = mw.to_3x3().inverted().transposed()
        for tri in mesh.loop_triangles:
            for loop in tri.loops:
                pos.append(convert(mw @ mesh.vertices[mesh.loops[loop].vertex_index].co))
                nrm.append(convert((nm @ Vector(mesh.corner_normals[loop].vector)).normalized()))
        ev.to_mesh_clear()
    return pos, nrm


rest_pos = sample()[0]
FLOOR = min(p.y for p in rest_pos)
height = max(p.y for p in rest_pos) - FLOOR
SIDES = (("Left", 1.0), ("Right", -1.0))
hip_rest = {s: rest(f"{s}UpperLeg") for s, _ in SIDES}
L1 = sum((rest(f"{s}LowerLeg") - rest(f"{s}UpperLeg")).length for s, _ in SIDES) / 2
L2 = sum((rest(f"{s}Foot") - rest(f"{s}LowerLeg")).length for s, _ in SIDES) / 2
LEG = L1 + L2
ANKLE = sum(rest(f"{s}Foot").y for s, _ in SIDES) / 2 - FLOOR  # 脚平放时脚踝离地多高
BALL = sum(rest(f"{s}Toes").y for s, _ in SIDES) / 2 - FLOOR  # 前脚掌关节离地多高
# 前脚掌 → 脚踝（往后、往上）。
HEEL_Y = ANKLE - BALL
HEEL_Z = sum(rest(f"{s}Foot").z - rest(f"{s}Toes").z for s, _ in SIDES) / 2
HIP_Y = sum(hip_rest[s].y for s, _ in SIDES) / 2
CENTER = Vector((sum(hip_rest[s].x for s, _ in SIDES) / 2, 0, sum(hip_rest[s].z for s, _ in SIDES) / 2))
HALF = abs(hip_rest["Left"].x - hip_rest["Right"].x) / 2 * 0.92  # 两脚分开多宽（比胯略窄）
STRIDE = 1.10 * LEG  # 走一个循环（左右各一步）前进多远
STANCE = 0.62  # 一只脚一个循环里有多大比例踩在地上
MID = 0.29  # 踩地的脚在这个相位正好在胯的正下方
SIT_BACK = 0.95 * L1
print(f"身高 {height:.3f} 腿长 {LEG:.3f}（大腿 {L1:.3f} 小腿 {L2:.3f}）脚踝高 {ANKLE:.3f} 前脚掌高 {BALL:.3f} "
      f"脚长 {HEEL_Z:.3f} 两脚半宽 {HALF:.3f}")


def body(lift, back=0.0, sway=0.0, lean=0.0, yaw=0.0, chest_lean=None, chest_yaw=0.0, head=12.0):
    """摆躯干。lift：胯关节比脚踝高多少；back：胯往后挪多少；lean：前倾角度。"""
    hips = arm.pose.bones["Hips"]
    delta = Vector((sway, FLOOR + ANKLE + lift - HIP_Y, -back))
    m = hips.matrix.copy()
    m.translation = m.translation + WI @ world(delta)
    hips.matrix = m
    bpy.context.view_layer.update()
    set_rot("Hips", rot((0, 1, 0), yaw) @ rot((1, 0, 0), lean))
    chest = rot((0, 1, 0), chest_yaw) @ rot((1, 0, 0), lean if chest_lean is None else chest_lean)
    for name in ("Spine", "Chest", "Neck"):
        set_rot(name, chest)
    set_rot("Head", chest @ rot((1, 0, 0), head))  # 低头看手机
    set_rot("LeftShoulder", chest)
    set_rot("RightShoulder", chest)


def leg(side, sign, ball, pitch, toe_out=7.0):
    """摆一条腿。ball：前脚掌关节的位置（人物坐标，y 是离地高度）；pitch：脚尖朝下多少度（负的是脚尖翘起）。"""
    if pitch >= 0:  # 蹬地：绕前脚掌转，脚跟抬起来
        c, s = math.cos(math.radians(pitch)), math.sin(math.radians(pitch))
        ankle = ball + Vector((0, HEEL_Y * c - HEEL_Z * s, HEEL_Y * s + HEEL_Z * c))
    else:  # 脚跟着地：脚踝留在平放时的高度，脚尖翘起
        ankle = ball + Vector((0, HEEL_Y, HEEL_Z))
    ankle = Vector((CENTER.x + sign * HALF, FLOOR + ankle.y, CENTER.z + ankle.z))
    hip = joint(f"{side}UpperLeg")
    knee, end = ik(hip, ankle, L1, L2, Vector((sign * 0.06, 0.25, 1)))
    point(f"{side}UpperLeg", f"{side}LowerLeg", knee - hip)
    point(f"{side}LowerLeg", f"{side}Foot", end - knee)
    turn = rot((0, 1, 0), sign * toe_out)
    set_rot(f"{side}Foot", turn @ rot((1, 0, 0), pitch))
    set_rot(f"{side}Toes", turn @ rot((1, 0, 0), min(pitch, 0.0)))  # 蹬地时脚趾留在地上


def arms(swing, elbow, chest_lean=0.0, rest_on_thigh=0.0):
    """左臂：垂着，前后摆 swing 度，肘弯 elbow 度；rest_on_thigh → 1 时手搭到大腿上。右臂：举着手机。"""
    a = math.radians(swing + 22 * rest_on_thigh)
    upper = Vector((0.13, -math.cos(a), math.sin(a)))
    b = a + math.radians(elbow + 50 * rest_on_thigh)
    lower = Vector((0.04 - 0.10 * rest_on_thigh, -math.cos(b), math.sin(b)))
    point("LeftUpperArm", "LeftLowerArm", upper)
    point("LeftLowerArm", "LeftHand", lower)
    set_rot("LeftHand", (rest("LeftHand") - rest("LeftLowerArm")).normalized().rotation_difference(lower.normalized()))
    # 右臂：上臂垂着稍微往前，前臂往前平伸、略微上抬、往身体中间收，手心转向上。
    point("RightUpperArm", "RightLowerArm", Vector((-0.10, -0.93, 0.30)))
    fore = Vector((0.20, 0.30, 0.93))
    point("RightLowerArm", "RightHand", fore, twist=35)
    q = (rest("RightHand") - rest("RightLowerArm")).normalized().rotation_difference(fore.normalized())
    set_rot("RightHand", Quaternion(fore.normalized(), math.radians(70)) @ q)


def pose_idle(q):
    """待机。q：循环里的相位 0~1。"""
    reset()
    w = 2 * math.pi * q
    body(lift=LEG * (0.992 + 0.0025 * math.sin(w)), sway=LEG * 0.006 * math.sin(w - 0.9),
         lean=0.6, chest_lean=1.2 + 0.8 * math.sin(w - 0.5), head=13 + 1.0 * math.sin(w - 1.6))
    leg("Left", 1.0, Vector((0, BALL, -HEEL_Z + 0.03 * LEG)), 0.0, toe_out=9)
    leg("Right", -1.0, Vector((0, BALL, -HEEL_Z - 0.01 * LEG)), 0.0, toe_out=9)
    arms(swing=2 + 1.5 * math.sin(w - 1.2), elbow=14)


def foot_path(p):
    """走路时一只脚的前脚掌位置（相对胯的正下方）和脚尖朝下的角度。p = 0 是这只脚落地的时刻。"""
    front = STRIDE * MID - HEEL_Z
    if p < STANCE:
        z, y = front - STRIDE * p, 0.0
    else:
        u = (p - STANCE) / (1 - STANCE)
        z0, m = front - STRIDE * STANCE, -STRIDE * (1 - STANCE)  # 起落两头的速度都接上踩地时的速度
        z = ((2 * u**3 - 3 * u**2 + 1) * z0 + (u**3 - 2 * u**2 + u) * m
             + (-2 * u**3 + 3 * u**2) * front + (u**3 - u**2) * m)
        y = 0.055 * LEG * math.sin(math.pi * u) ** 2
    pitch = keys([(0, -13), (0.07, 0), (0.36, 0), (0.50, 17), (STANCE, 46), (0.72, 30), (0.86, -4),
                  (0.94, -13), (1, -13)], p)
    return Vector((0, BALL + y, z)), pitch


def pose_walk(p):
    """走路。p：循环里的相位 0~1，0 是左脚落地。"""
    reset()
    w = 2 * math.pi * p
    body(lift=LEG * (0.9625 - 0.0225 * math.cos(4 * math.pi * (p - 0.05))),
         sway=LEG * 0.022 * math.sin(2 * math.pi * (p - 0.04)),
         lean=3.5, yaw=-4.0 * math.cos(w), chest_lean=4.5, chest_yaw=1.2 * math.cos(w), head=10)
    for side, sign, phase in (("Left", 1.0, p), ("Right", -1.0, (p + 0.5) % 1.0)):
        ball, pitch = foot_path(phase)
        leg(side, sign, ball, pitch)
    forward = -math.cos(w)  # 左臂在右脚往前时往前摆
    arms(swing=3 + 16 * forward, elbow=16 + 9 * max(0.0, forward))


def pose_sit(a):
    """坐下。a = 0 站着，1 坐稳。脚不动，胯往后往下。"""
    reset()
    e = smooth(0, 1, a)
    lift = LEG * 0.992 + (L2 * 1.0 + LEG * 0.03 - LEG * 0.992) * e
    lean = 0.6 + 26 * math.sin(math.pi * a) ** 1.3 + 3.4 * e
    body(lift=lift, back=SIT_BACK * e, lean=lean * 0.55, chest_lean=lean, head=13 - 4 * math.sin(math.pi * a))
    leg("Left", 1.0, Vector((0, BALL, -HEEL_Z + 0.03 * LEG)), 0.0, toe_out=9)
    leg("Right", -1.0, Vector((0, BALL, -HEEL_Z - 0.01 * LEG)), 0.0, toe_out=9)
    arms(swing=2, elbow=14, rest_on_thigh=e)


# ---- 每个面角的 uv 和原始顶点编号（合并重复顶点用） ----
keys_, uvs = [], []
for obj in meshes:
    mesh = obj.data
    mesh.calc_loop_triangles()
    uv = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    for tri in mesh.loop_triangles:
        for loop in tri.loops:
            t = uv[loop].uv if uv else (0.0, 0.0)
            n = mesh.corner_normals[loop].vector
            keys_.append((obj.name, mesh.loops[loop].vertex_index, round(t[0], 5), round(t[1], 5),
                          round(n.x, 3), round(n.y, 3), round(n.z, 3)))
            uvs.append((t[0], 1 - t[1]))
unique, indices, first = {}, [], []
for i, k in enumerate(keys_):
    j = unique.get(k)
    if j is None:
        j = unique[k] = len(first)
        first.append(i)
    indices.append(j)

# 手机放在右手手心：主要跟着右手（和它下面的手指骨头）动的那些顶点的中心。
hand_bones = set()
stack = [arm.data.bones["RightHand"]]
while stack:
    b = stack.pop()
    hand_bones.add(b.name)
    stack.extend(b.children)
hand_vertices = set()
for obj in meshes:
    groups = {g.index for g in obj.vertex_groups if g.name in hand_bones}
    for v in obj.data.vertices:
        if sum(g.weight for g in v.groups if g.group in groups) > 0.5:
            hand_vertices.add((obj.name, v.index))
hand_corners = [i for i, k in enumerate(keys_) if (k[0], k[1]) in hand_vertices]

plan = ([(pose_idle, k / IDLE) for k in range(IDLE)] + [(pose_walk, k / WALK) for k in range(WALK)]
        + [(pose_sit, k / (SIT - 1)) for k in range(SIT)])
frames, hands, lows = [], [], []
origin = Vector((CENTER.x, FLOOR, CENTER.z))
for fn, phase in plan:
    fn(phase)
    pos, nrm = sample()
    pos = [p - origin for p in pos]
    lows.append(min(p.y for p in pos))
    frames.append((pos, nrm))
    hands.append(sum((pos[i] for i in hand_corners), Vector()) / max(1, len(hand_corners)))
print("每帧最低点离地（应该都接近 0）：", " ".join(f"{v / height * 100:+.1f}%" for v in lows))

with open(out, "wb") as fh:
    n = len(first)
    fh.write(b"TEC3" + struct.pack("<IIIIIfff", n, len(indices), IDLE, WALK, SIT, height, STRIDE, SIT_BACK))
    for h in hands:
        fh.write(struct.pack("<3f", h.x, h.y, h.z))
    for i in first:
        fh.write(struct.pack("<2f", *uvs[i]))
    fh.write(struct.pack(f"<{len(indices)}I", *indices))
    for pos, nrm in frames:
        buf = bytearray()
        for i in first:
            p, q = pos[i], nrm[i]
            buf += struct.pack("<6f", p.x, p.y, p.z, q.x, q.y, q.z)
        fh.write(buf)
print(f"导出 {len(first)} 个顶点 × {len(frames)} 帧（待机 {IDLE} + 走路 {WALK} + 坐下 {SIT}），身高 {height:.2f}，"
      f"一个走路循环前进 {STRIDE:.2f}，坐下往后 {SIT_BACK:.2f}，手机位置（第 0 帧）"
      f"({hands[0].x:.2f}, {hands[0].y:.2f}, {hands[0].z:.2f})")

if preview:
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x, scene.render.resolution_y = 300, 440
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = height * 1.3
    mid = world(Vector((CENTER.x, FLOOR + height * 0.5, CENTER.z - 0.1 * height)))
    shots = [(pose_idle, 0.0)] + [(pose_walk, k / 8) for k in range(8)] + [(pose_sit, 0.5), (pose_sit, 1.0)]
    for i, (fn, phase) in enumerate(shots):
        fn(phase)
        for view, offset in (("side", LEFT * 5 + FWD * 0.6 + UP * 0.2), ("front", FWD * 5 + LEFT * 1.6 + UP * 0.4)):
            loc = mid + offset * height
            zc = (loc - mid).normalized()
            xc = UP.cross(zc).normalized()
            yc = zc.cross(xc)
            cam.matrix_world = Matrix.Translation(loc) @ Matrix((xc, yc, zc)).transposed().to_4x4()
            scene.render.filepath = f"{preview}/anim_{i:02d}_{view}.png"
            bpy.ops.render.render(write_still=True)

image = max(bpy.data.images, key=lambda i: i.size[0] * i.size[1])
image.filepath_raw = out_png
image.file_format = "PNG"
image.save()
