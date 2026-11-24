"""把带开合动画的折叠手机模型逐帧烘焙成 app 用的网格：第 0 帧合上，最后一帧完全展开。
用法: /Applications/Blender.app/Contents/MacOS/Blender -b --python tools/bake_phone.py -- \
        <模型.glb> <输出.bin> <输出调色板.png> [几帧，默认 9] [减面比例，默认 0.3] [预览图目录]

输出格式和 bake_animated.py 的一样（"TEC2"），app 里按真机铰链的角度在相邻两帧之间插值。
材质不带贴图过去：每种材质取一个平均色，排成一条调色板，顶点的 u 指向自己那一格。
两块屏幕例外：保留它们自己的 uv，u 加 2（内屏）或加 4（外屏）做记号，app 里在这两块上显示相机画面。
坐标：原点在展开状态的中心，x 朝右，y 朝上，z 是展开时内屏朝的方向（对着用户），单位按模型原样（app 里再缩放）。
"""
import struct
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
src, out, out_png = args[0], args[1], args[2]
count = int(args[3]) if len(args) > 3 else 9
ratio = float(args[4]) if len(args) > 4 else 0.3
preview = args[5] if len(args) > 5 else None
INNER, OUTER = "Screen_Inner", "Screen_Outer"  # 材质名里含这两个词的是内屏、外屏

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=src)
for o in list(bpy.data.objects):  # 导入器给骨头配的显示用小球，不是模型的一部分
    if o.type == "MESH" and o.name.startswith("Icosphere"):
        bpy.data.objects.remove(o)
meshes = [o for o in bpy.data.objects if o.type == "MESH"]
action = max(bpy.data.actions, key=lambda a: a.frame_range[1] - a.frame_range[0])
f0, f1 = action.frame_range
scene = bpy.context.scene

# 减面：模型有十万个三角形，画在屏幕上只有巴掌大。屏幕那两块不减（要保住 uv）。
for o in meshes:
    names = [m.name for m in o.data.materials if m]
    if len(o.data.vertices) > 200 and not any(INNER in n or OUTER in n for n in names):
        mod = o.modifiers.new("decimate", "DECIMATE")
        mod.ratio = ratio
        while o.modifiers[0] != mod:
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.modifier_move_up(modifier=mod.name)


def sample(frame):
    scene.frame_set(int(round(frame)))
    deps = bpy.context.evaluated_depsgraph_get()
    pos, nrm, uvs, mats = [], [], [], []
    for obj in meshes:
        ev = obj.evaluated_get(deps)
        mesh = ev.to_mesh()
        mesh.calc_loop_triangles()
        world = ev.matrix_world
        nm = world.to_3x3().inverted().transposed()
        uv = mesh.uv_layers.active.data if mesh.uv_layers.active else None
        for tri in mesh.loop_triangles:
            mat = obj.material_slots[tri.material_index].material if obj.material_slots else None
            for loop in tri.loops:
                pos.append(world @ mesh.vertices[mesh.loops[loop].vertex_index].co)
                nrm.append((nm @ Vector(mesh.corner_normals[loop].vector)).normalized())
                uvs.append(tuple(uv[loop].uv) if uv else (0.0, 0.0))
                mats.append(mat.name if mat else "")
        ev.to_mesh_clear()
    return pos, nrm, uvs, mats


def size(pos):
    lo = Vector(tuple(min(p[i] for p in pos) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in pos) for i in range(3)))
    return lo, hi


a, b = sample(f0), sample(f1)
ea, eb = size(a[0]), size(b[0])
vol = lambda e: max(e[1][i] - e[0][i] for i in range(3)) * sorted(e[1][i] - e[0][i] for i in range(3))[1]  # noqa: E731
# 展开的那一头，正面看过去的面积更大。
open_frame, closed_frame = (f1, f0) if vol(eb) > vol(ea) else (f0, f1)
opened = sample(open_frame)
lo, hi = size(opened[0])
center = (lo + hi) / 2
# 展开时内屏朝哪：内屏那些面的法线取平均。
inner_normals = [n for n, m in zip(opened[1], opened[3]) if INNER in m]
front = sum(inner_normals, Vector()).normalized() if inner_normals else Vector((0, -1, 0))
up = Vector((0, 0, 1))
front = (front - up * front.dot(up)).normalized()
right = up.cross(front)
print(f"动作 {action.name}：第 {int(closed_frame)} 帧合上，第 {int(open_frame)} 帧展开；"
      f"展开时大小 {tuple(round(hi[i] - lo[i], 3) for i in range(3))}，内屏朝 {tuple(round(x, 2) for x in front)}")


def convert(p):
    d = p - center
    return Vector((d.dot(right), d.dot(up), d.dot(front)))


def turn(n):
    return Vector((n.dot(right), n.dot(up), n.dot(front)))


# 调色板：每种材质一格。带贴图的取贴图的平均色。
materials = sorted({m for m in opened[3]})
palette = []
for name in materials:
    mat = bpy.data.materials.get(name)
    color = (0.5, 0.5, 0.5)
    bsdf = mat.node_tree.nodes.get("Principled BSDF") if mat and mat.use_nodes else None
    if bsdf:
        inp = bsdf.inputs["Base Color"]
        color = tuple(inp.default_value)[:3]
        if inp.is_linked:
            node = inp.links[0].from_node
            image = getattr(node, "image", None)
            if image and image.size[0] > 0:
                px = image.pixels[:]
                step = max(1, (len(px) // 4) // 4000)
                samples = [px[i * 4:i * 4 + 3] for i in range(0, len(px) // 4, step)]
                color = tuple(sum(s[k] for s in samples) / len(samples) for k in range(3))
    # Blender 里的颜色是线性的，存成图要转成 sRGB。
    palette.append(tuple(c ** (1 / 2.2) for c in color))
index_of = {n: i for i, n in enumerate(materials)}

frames = [sample(closed_frame + (open_frame - closed_frame) * k / (count - 1)) for k in range(count)]
uvs = []
for (u, v), m in zip(opened[2], opened[3]):
    if INNER in m:
        uvs.append((u + 2, 1 - v))
    elif OUTER in m:
        uvs.append((u + 4, 1 - v))
    else:
        uvs.append(((index_of[m] + 0.5) / len(materials), 0.5))
# 位置、法线、uv 都一样的面角并成一个顶点（按展开那一帧判断），文件能小好几倍。
unique, indices, first = {}, [], []
for i, (p, q, t) in enumerate(zip(opened[0], opened[1], uvs)):
    key = (round(p.x, 5), round(p.y, 5), round(p.z, 5), round(q.x, 2), round(q.y, 2), round(q.z, 2),
           round(t[0], 4), round(t[1], 4))
    j = unique.get(key)
    if j is None:
        j = unique[key] = len(first)
        first.append(i)
    indices.append(j)
n = len(first)
height = hi[2] - lo[2]
with open(out, "wb") as fh:
    fh.write(b"TEC2" + struct.pack("<IIIf3f", n, len(indices), count, height, 0.0, 0.0, 0.0))
    for i in first:
        fh.write(struct.pack("<2f", *uvs[i]))
    fh.write(struct.pack(f"<{len(indices)}I", *indices))
    for pos, nrm, _, _ in frames:
        buf = bytearray()
        for i in first:
            c, d = convert(pos[i]), turn(nrm[i])
            buf += struct.pack("<6f", c.x, c.y, c.z, d.x, d.y, d.z)
        fh.write(buf)

image = bpy.data.images.new("palette", len(materials), 1)
flat = []
for c in palette:
    flat += [c[0], c[1], c[2], 1.0]
image.pixels = flat
image.filepath_raw = out_png
image.file_format = "PNG"
image.save()
print(f"导出 {n} 个顶点 × {count} 帧（{len(indices) // 3} 个三角形），{len(materials)} 种材质，高 {height:.3f}")

if preview:
    from mathutils import Matrix
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "TEXTURE"
    scene.render.resolution_x, scene.render.resolution_y = 420, 420
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam"))
    scene.collection.objects.link(cam)
    scene.camera = cam
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = height * 1.6
    for k, fr in enumerate((closed_frame, (closed_frame + open_frame) / 2, open_frame)):
        scene.frame_set(int(round(fr)))
        for view, offset in (("front", front * 5 + right * 1.5 + up * 1.0), ("back", -front * 5 + right * 1.5 + up * 1.0)):
            loc = center + offset * height
            zc = (loc - center).normalized()
            xc = up.cross(zc).normalized()
            yc = zc.cross(xc)
            cam.matrix_world = Matrix.Translation(loc) @ Matrix((xc, yc, zc)).transposed().to_4x4()
            scene.render.filepath = f"{preview}/duo_{k}_{view}.png"
            bpy.ops.render.render(write_still=True)
