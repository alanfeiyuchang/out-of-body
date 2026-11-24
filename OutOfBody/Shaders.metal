#include <metal_stdlib>
using namespace metal;

// ───────────────────────── splat ─────────────────────────
//
// 不排序的 splat 混合，分两遍：
//   1. 只写深度，把每个圆片沿视线往后推一点；
//   2. 深度测试 <=、不写深度，按高斯权重把结果加到三张目标上：
//        目标 0：颜色·w 和 w（splat 自带的颜色，很糊，当兜底）
//        目标 1：到相机的距离·w（除以 w 得到一张平滑的深度）
//        目标 2：取最大值，记下「这个像素上权重最大的 splat 用哪张关键帧」
// resolve 时用平滑深度算出世界坐标，投回那张关键帧取色。
// 这样清晰度由相机图决定，不再由 2 cm 的体素决定。

#define KEYFRAME_SLOTS 60
// 混合遍写进 id 目标的槽位号：1~60 关键帧，61/62 最新一帧，这个值表示「预测出来的」。
#define PREDICTED_SLOT 200u

struct Splat {
    packed_float3 pos;
    uint color;  // r | g<<8 | b<<16 | 可信度<<24
    uint tex;    // 关键帧槽位(6) | 代数<<6 (8) | 大小档<<14 (2) | 最后被看到的帧号<<16
    uint normal; // 表面朝向，每个分量 8 位；0 = 不知道
};

struct Keyframe {
    float4x4 worldToUV;
    float4 info;  // x = 代数, y = 1 可用
};

struct SplatUniforms {
    float4x4 view;
    float4x4 proj;
    float4 phonePos;
    float4 fog;      // rgb + 起始距离
    float4 params;   // 半径, 近裁剪距离, 深度后推倍数, 实时层的像素张角(0 = 用固定半径)
    float4 params2;  // 最低可信度, 雾的过渡距离, 这一批是不是预测出来的, 高过这个高度的不画
    float4 live;     // 实时槽位, 实时帧号, 贴图开关, 实时帧是否有效
    float4 eye;      // 这次渲染的相机位置 + 圆片贴着表面画的开关
};

struct SplatOut {
    float4 position [[position]];
    float2 uv;
    float3 color;
    float dist;
    float code [[flat]];
};

vertex SplatOut splat_vs(uint vid [[vertex_id]],
                         uint iid [[instance_id]],
                         const device Splat *splats [[buffer(0)]],
                         constant SplatUniforms &u [[buffer(1)]],
                         const device Keyframe *kfs [[buffer(2)]]) {
    Splat s = splats[iid];
    float2 q = float2((vid & 1) ? 1.0 : -1.0, (vid & 2) ? 1.0 : -1.0);
    float3 wp = float3(s.pos);
    float4 vp = u.view * float4(wp, 1.0);
    float dist = -vp.z;
    float hits = float(s.color >> 24);

    SplatOut o;
    o.uv = q;
    o.dist = dist;
    o.code = 0.0;
    if (hits < u.params2.x || dist < u.params.y || wp.y > u.params2.w) {
        o.position = float4(0.0, 0.0, -1.0, 1.0);  // 裁掉
        o.color = float3(0.0);
        return o;
    }

    // 采样稀的地方（远处、斜着看的表面）圆片画大一些，不然中间是空的。
    uint cls = (s.tex >> 14) & 3u;
    float grow = cls == 0u ? 1.0 : (cls == 1u ? 1.6 : (cls == 2u ? 2.4 : 3.6));
    float r = u.params.w > 0.0
        ? max(0.004, distance(wp, u.phonePos.xyz) * u.params.w)
        : u.params.x * grow;
    float3 p = vp.xyz;
    uint nb = s.normal & 0xFFFFFFu;
    if (nb != 0u && u.eye.w > 0.5) {
        // 知道表面朝哪：圆片躺在表面上。墙和地面画出来是平的，物体的轮廓也不会鼓出一圈。
        float3 n = normalize((float3(float(nb & 255u), float((nb >> 8) & 255u),
                                     float((nb >> 16) & 255u)) - 128.0) / 127.0);
        float3 v = normalize(u.eye.xyz - wp);
        float c = dot(n, v);
        if (c < 0.0) {
            n = -n;
            c = -c;
        }
        // 几乎侧对着镜头时圆片会缩成一条线，往镜头方向掰回来一点。
        n = normalize(n + v * max(0.0, 0.45 - c) * 1.2);
        float3 t = normalize(cross(n, abs(n.y) < 0.9 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0)));
        float3 b = cross(n, t);
        p = (u.view * float4(wp + (t * q.x + b * q.y) * r, 1.0)).xyz;
    } else {
        p.xy += q * r;
    }
    // 深度按顶点给：斜着的圆片上每一点的深度不一样，resolve 靠它还原世界坐标去取贴图。
    o.dist = -p.z;
    p *= 1.0 + u.params.z * r / max(length(p), 1e-4);
    o.position = u.proj * float4(p, 1.0);

    float3 c = float3(float(s.color & 0xFF), float((s.color >> 8) & 0xFF),
                      float((s.color >> 16) & 0xFF)) / 255.0;
    o.color = c;  // 雾在 resolve 里统一加
    if (u.eye.w > 1.5) {
        // 调试：直接把朝向当颜色画出来，没有朝向的画成黑的。
        o.color = nb != 0u
            ? (float3(float(nb & 255u), float((nb >> 8) & 255u), float((nb >> 16) & 255u)) / 255.0)
            : float3(0.0);
        return o;
    }

    // 用哪张图：这一帧刚被看到的用最新的相机画面，否则用它自己记着的关键帧（如果还在）。
    if (u.params2.z > 0.5) {
        o.code = float(PREDICTED_SLOT);
    } else if (u.live.z > 0.5) {
        uint slot = s.tex & 63u;
        uint gen = (s.tex >> 6) & 255u;
        uint stamp = s.tex >> 16;
        if (u.live.w > 0.5 && stamp == uint(u.live.y)) {
            o.code = 65536.0 + u.live.x + 1.0;
        } else if (slot < KEYFRAME_SLOTS && kfs[slot].info.y > 0.5
                   && uint(kfs[slot].info.x + 0.5) == gen) {
            o.code = float(slot) + 1.0;
        }
    }
    return o;
}

fragment void splat_depth_fs(SplatOut in [[stage_in]]) {
    if (dot(in.uv, in.uv) > 1.0) discard_fragment();
}

struct BlendOut {
    float4 color [[color(0)]];
    float depth [[color(1)]];
    float id [[color(2)]];
};

fragment BlendOut splat_blend_fs(SplatOut in [[stage_in]]) {
    float r2 = dot(in.uv, in.uv);
    if (r2 > 1.0) discard_fragment();
    float w = exp(-3.0 * r2);
    BlendOut o;
    o.color = float4(in.color * w, w);
    o.depth = in.dist * w;
    // 一个 float 里塞三样：最高位是「实时」，中间 8 位是权重，最低 8 位是槽位+1。
    // 混合方式是取最大值，所以留下来的是「实时优先、其次权重最大」的那个 splat 的槽位。
    o.id = in.code + floor(w * 255.0) * 256.0;
    return o;
}

// ───────────────────────── resolve ─────────────────────────

struct ResolveUniforms {
    float4x4 invViewProj;
    float4 cameraPos;
    float4 cameraFwd;
    float4 bg;         // 没数据时的底色
    float4 fog;        // rgb + 起始距离
    float4 params;     // 地面高度, 补洞开关, 雾的过渡距离, 贴图开关
    float4 misc;       // 网格亮度, 背景那个球的半径, 那个球的强度（相机拉远了淡掉）, 高过这个高度的不画
    float4 envCenter;  // 用户所在位置 + 预测强度
    float4 genCenter;  // 生成式预测那张全景图的中心 + 强度
    float4 room;       // 房间轮廓的中心 x、z, 天花板高度, 平均半径（0 = 没有轮廓，用球）
    float4 floorMap;   // 地面颜色图的中心 x、z, 1 / 整张图的边长, 是否可用
    float4 extra;      // x: 调试染色, y: 有没有网络画好的图（服务器上的扩散模型或手机上它的学生）, z: 那幅画立在镜头前多远, w: 那张图还有多可信（镜头转开太多就不用它）
    float4x4 genViewProj;  // 那张图是从哪个机位画的（世界 → 它的裁剪空间）
};

// 从轮廓中心往 q 这个水平方向看，墙有多远。
static float roomRadius(texture2d<float> room, sampler smp, float2 q) {
    return room.sample(smp, float2(atan2(q.y, q.x) / (2.0 * M_PI_F) + 0.5, 0.5), level(0.0)).r;
}

// 视线打在房间轮廓的墙上：返回沿视线的距离，打不到返回 -1。
// 轮廓是「每个方向一个半径」，没法直接解。先当它是个圆柱求交，
// 看交点落在哪个方向、那个方向的半径是多少，换成这个半径再求，几轮就收敛了。
// 取的是远的那个交点：相机在房间外面往里看时，近处那面墙是透明的。
static float roomWall(float3 o, float3 d, texture2d<float> room, sampler smp, float2 c, float r0) {
    float2 oc = o.xz - c;
    float2 dd = d.xz;
    float a = dot(dd, dd);
    if (a < 1e-6) return -1.0;
    float b = dot(oc, dd);
    float R = r0;
    float t = -1.0;
    for (int i = 0; i < 6; i++) {
        float disc = b * b - a * (dot(oc, oc) - R * R);
        if (disc < 0.0) return i == 0 ? -1.0 : t;
        t = (-b + sqrt(disc)) / a;
        if (t <= 0.0) return -1.0;
        R = roomRadius(room, smp, oc + dd * t);
    }
    return t;
}

struct FullscreenOut {
    float4 position [[position]];
    float2 ndc;
};

vertex FullscreenOut fullscreen_vs(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    FullscreenOut o;
    o.position = float4(p * 2.0 - 1.0, 0.0, 1.0);
    o.ndc = p * 2.0 - 1.0;
    return o;
}

// 点状表面补洞。
//
// 只被看到过几次的表面，画出来是一颗一颗隔着缝的圆片，缝里透出后面的墙。
// 往八个方向、三个距离上各看一眼：如果大多数方向上都有一层更靠前的表面，
// 说明这个像素其实是那层表面上的一个洞，用周围的深度和颜色把它补上。
// 要求「大多数方向都有」，所以物体的轮廓不会往外长。
static bool fillHole(texture2d<float> accum, texture2d<float> accumZ, texture2d<float> idTex,
                     int2 px, float ownZ, thread float3 &color, thread float &z,
                     thread uint &slot) {
    const int2 dirs[8] = {int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1),
                          int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1)};
    int2 size = int2(accum.get_width(), accum.get_height());
    int base = max(3, size.y / 280);
    const int rings = 3;
    int radius[3] = {base, base * 5 / 2, base * 5};

    float zs[24];
    float zmin = 1e9;
    for (int i = 0; i < 8; i++) {
        for (int k = 0; k < rings; k++) {
            int2 q = clamp(px + dirs[i] * radius[k], int2(0), size - 1);
            float w = accum.read(uint2(q)).a;
            float zq = w > 0.02 ? accumZ.read(uint2(q)).r / w : 0.0;
            zs[i * rings + k] = zq;
            if (zq > 0.0) zmin = min(zmin, zq);
        }
    }
    if (zmin > 1e8) return false;
    // 自己已经在最靠前的那层上了，没什么好补的。
    if (ownZ > 0.0 && ownZ < zmin * 1.10 + 0.03) return false;

    float limit = zmin * 1.10 + 0.03;
    int hitDirs = 0;
    float wsum = 0.0, zsum = 0.0;
    float3 csum = float3(0.0);
    float best = 0.0;
    int2 bestQ = px;
    for (int i = 0; i < 8; i++) {
        bool hit = false;
        for (int k = 0; k < rings && !hit; k++) {
            float zq = zs[i * rings + k];
            if (zq <= 0.0 || zq > limit) continue;
            hit = true;
            int2 q = clamp(px + dirs[i] * radius[k], int2(0), size - 1);
            float4 a = accum.read(uint2(q));
            float wgt = 1.0 / float(radius[k]);
            wsum += wgt;
            zsum += zq * wgt;
            csum += a.rgb / a.a * wgt;
            if (wgt > best) {
                best = wgt;
                bestQ = q;
            }
        }
        if (hit) hitDirs++;
    }
    if (hitDirs < 6) return false;
    color = csum / wsum;
    z = zsum / wsum;
    slot = uint(idTex.read(uint2(bestQ)).r + 0.5) & 255u;
    return true;
}

// 这个像素上的表面：颜色（投影贴图或者 splat 自己的颜色）和盖住的程度，另外给出平滑后的深度。
// 屏幕画面和全景图的面都用它。
static float4 surfaceAt(uint2 px, float3 d,
                        texture2d<float> accum, texture2d<float> accumZ, texture2d<float> idTex,
                        texture2d_array<float> frames, texture2d_array<float> kfDepth,
                        texture2d<float> gen, sampler smp, sampler envSmp,
                        constant ResolveUniforms &u, const device Keyframe *kfs,
                        thread float &zOut) {
    float4 a = accum.read(px);
    // 圆片边缘的权重只有中心的 5%。只要有东西落在这个像素上就算盖住了，
    // 否则深度参差的地方每个圆片都会透出一圈背景色，像一堆带黑边的泡泡。
    float coverage = saturate(a.w * 40.0);
    float3 c = a.rgb / max(a.w, 1e-4);
    float z = a.w > 0.004 ? accumZ.read(px).r / a.w : 0.0;
    uint slot = uint(idTex.read(px).r + 0.5) & 255u;

    if (u.params.y > 0.0) {
        float3 fc;
        float fz;
        uint fslot;
        if (fillHole(accum, accumZ, idTex, int2(px), z, fc, fz, fslot)) {
            c = fc;
            z = fz;
            slot = fslot;
            coverage = 1.0;
        }
    }
    zOut = z;
    if (coverage <= 0.0) return float4(0.0);

    float3 world = u.cameraPos.xyz + d * (z / max(dot(d, u.cameraFwd.xyz), 1e-3));
    if (slot == PREDICTED_SLOT) {
        // 预测出来的表面：颜色取自那张补全的全景图，从它的中心看过来的方向去查。
        // 压暗一点，让人看得出这是猜的。
        float3 gd = normalize(world - u.genCenter.xyz);
        float2 guv = float2(atan2(gd.z, gd.x) / (2.0 * M_PI_F) + 0.5,
                            acos(clamp(gd.y, -1.0, 1.0)) / M_PI_F);
        // 明确取第 0 层：这里在分支里面，自动选层用的导数不可靠，会取到很糊的那几层。
        c = gen.sample(envSmp, guv, level(0.0)).rgb * 0.86;
        coverage *= u.genCenter.w;
    } else if (u.params.w > 0.5 && slot > 0u) {
        // ---- 投影贴图 ----
        float4 h = kfs[slot - 1u].worldToUV * float4(world, 1.0);
        float2 uv = h.xy / h.w;
        if (h.w > 0.05 && all(uv > float2(0.002)) && all(uv < float2(0.998))) {
            // 遮挡检查：那张图当时在这个方向上看到的深度，和这个点到那台相机的距离对不对得上。
            // 对不上说明这个点当时被别的东西挡着（或者正好在物体边缘），贴上去的会是别人的颜色。
            bool seen = true;
            if (kfDepth.get_width() > 1) {
                float2 dp = uv * float2(kfDepth.get_width(), kfDepth.get_height()) - 0.5;
                int2 p0 = int2(floor(dp));
                int2 dmax = int2(kfDepth.get_width(), kfDepth.get_height()) - 1;
                float gap = 1e9;
                for (int j = 0; j < 4; j++) {
                    int2 q = clamp(p0 + int2(j & 1, j >> 1), int2(0), dmax);
                    float kd = kfDepth.read(uint2(q), slot - 1u).r;
                    if (kd > 0.0) gap = min(gap, abs(kd - h.w));
                }
                seen = gap < 0.05 + 0.07 * h.w;
            }
            if (seen) c = frames.sample(smp, uv, slot - 1u).rgb;
        }
    }
    return float4(c, coverage);
}

fragment float4 resolve_fs(FullscreenOut in [[stage_in]],
                           texture2d<float> accum [[texture(0)]],
                           texture2d<float> accumZ [[texture(1)]],
                           texture2d<float> idTex [[texture(2)]],
                           texture2d_array<float> frames [[texture(3)]],
                           texture2d<float> env [[texture(4)]],
                           texture2d<float> gen [[texture(5)]],
                           texture2d_array<float> kfDepth [[texture(6)]],
                           texture2d<float> room [[texture(7)]],
                           texture2d<float> floorTex [[texture(8)]],
                           texture2d<float> refined [[texture(9)]],
                           sampler smp [[sampler(0)]],
                           sampler envSmp [[sampler(1)]],
                           constant ResolveUniforms &u [[buffer(0)]],
                           const device Keyframe *kfs [[buffer(1)]]) {
    // 调试：点云画面染成洋红（`-rawTint YES`），录屏后数洋红占屏幕几成，就知道生成的图没盖住多少。
    if (u.extra.x > 0.5) return float4(1.0, 0.0, 1.0, 1.0);
    uint2 px = uint2(in.position.xy);

    float4 pn = u.invViewProj * float4(in.ndc, 0.0, 1.0);
    float4 pf = u.invViewProj * float4(in.ndc, 1.0, 1.0);
    float3 o = pn.xyz / pn.w;
    float3 d = normalize(pf.xyz / pf.w - o);

    float tFloor = d.y < -1e-3 ? (u.params.x - o.y) / d.y : -1.0;
    // 地面网格线要用屏幕空间的导数，得在任何提前返回之前算。
    float gridLine = 0.0;
    if (tFloor > 0.0) {
        float2 hit = (o + d * tFloor).xz;
        float2 g = abs(fract(hit - 0.5) - 0.5) / max(fwidth(hit), 1e-5);
        gridLine = (1.0 - min(min(g.x, g.y), 1.0)) * u.misc.x * exp(-tFloor * 0.14);
    }

    // 先看这个像素上有没有扫到的表面。盖满了就直接出结果，后面那一大段「猜背景」不用算：
    // 画面里大部分像素都是这种。
    // 网络画好的那张图（服务器上微调过的扩散模型，或者手机上它的学生）：它来得慢，是之前的机位画的。
    // 不直接铺满屏幕（那样画面一秒只跳几下），而是当成立在人偶那个距离上的一整幅画，
    // 每帧按现在的机位重新看它：镜头转动时完全跟得上，平移时人偶周围是准的，远近差得多的地方
    // 会轻微游动，等下一张图来了就对回去。
    // （试过按每个像素自己的深度去贴：真机上的几何太毛糙，物体边缘会把图撕开。）
    float genWeight = 0.0;
    float3 genColor = float3(0.0);
    if (u.extra.y > 0.5) {
        float t = u.extra.z / max(dot(d, u.cameraFwd.xyz), 1e-3);
        float4 h = u.genViewProj * float4(o + d * t, 1.0);
        float2 guv = float2(h.x / h.w * 0.5 + 0.5, 0.5 - h.y / h.w * 0.5);
        if (h.w > 0.05) {
            // 快出那张图的边了就淡回自己渲染的，不留一道硬边。
            float2 edge = min(guv, 1.0 - guv);
            genWeight = smoothstep(0.0, 0.03, min(edge.x, edge.y)) * u.extra.w;
            genColor = refined.sample(smp, clamp(guv, 0.0, 1.0)).rgb;
        }
        if (genWeight >= 0.999) return float4(genColor, 1.0);
    }

    float z;
    float4 surf = surfaceAt(px, d, accum, accumZ, idTex, frames, kfDepth, gen, smp, envSmp, u, kfs, z);
    float f = saturate((z - u.fog.w) / u.params.z);
    if (surf.a >= 0.999) return float4(mix(mix(surf.rgb, u.fog.rgb, f), genColor, genWeight), 1.0);

    // ---- 没扫到的地方 ----
    float3 bg = mix(u.bg.rgb * 0.7, u.bg.rgb * 1.5, saturate(d.y * 0.5 + 0.5));

    // 预测：把「从用户往各个方向看到过什么颜色」存成一张很小的全景图，没看过的方向用周围的颜色补。
    // 这里假设房间大概是以用户为中心的一个球加一块地面，视线打到哪，就取从用户看那个点的方向的颜色。
    if (u.envCenter.w > 0.0) {
        float3 c = u.envCenter.xyz;
        float3 oc = o - c;
        float b = dot(oc, d);
        float R = u.misc.y;
        float disc = b * b - (dot(oc, oc) - R * R);
        float tSphere = disc > 0.0 ? -b + sqrt(disc) : -1.0;
        // 没扫到的地方，视线打在哪：地面、墙还是天花板。
        float t = -1.0;
        float strength = 0.0;
        if (u.room.w > 0.0) {
            // 有房间的轮廓：地面铺到墙根为止，墙按轮廓竖起来，上面盖天花板。
            float2 rc = u.room.xy;
            if (tFloor > 0.0) {
                float2 q = (o + d * tFloor).xz - rc;
                float hd = length(q) / roomRadius(room, envSmp, q);
                if (hd < 1.0) {
                    t = tFloor;
                    strength = 1.0;
                }
            }
            if (t < 0.0) {
                float tw = roomWall(o, d, room, envSmp, rc, u.room.w);
                // 天花板只在相机在它下面的时候画；相机升到跟它差不多高或者更高时，
                // 画出来就是一块盖子，把屋里挡住。
                float tc = (d.y > 1e-3 && o.y < u.room.z - 0.35) ? (u.room.z - o.y) / d.y : -1.0;
                bool ceilOk = false;
                if (tc > 0.0 && u.room.z < u.misc.w) {
                    float2 q = (o + d * tc).xz - rc;
                    ceilOk = length(q) < roomRadius(room, envSmp, q);
                }
                if (ceilOk) {
                    t = tc;
                    strength = 1.0;
                } else if (tw > 0.0) {
                    float y = o.y + d.y * tw;
                    // 墙从地面画到天花板，再往上淡出一米多（不然画面顶上是一条硬边）；
                    // 俯视掀屋顶的时候，高处也一起掀掉。
                    if (y > u.params.x - 0.05 && y < u.misc.w) {
                        t = tw;
                        strength = 1.0 - smoothstep(u.room.z, u.room.z + 1.6, y);
                    }
                }
            }
        } else {
            // 还没有轮廓：用户周围的地面 + 一个球。地面是真知道在哪的（ARKit 认出来的），
            // 相机拉远了俯视的时候也照样补；那个球是瞎猜的，拉远了会露馅，要淡掉（misc.z）。
            if (tFloor > 0.0) {
                float hd = length((o + d * tFloor).xz - c.xz) / R;
                if (hd < 1.1) {
                    t = tFloor;
                    strength = 1.0 - smoothstep(0.6, 1.1, hd);
                }
            }
            if (t < 0.0 && tSphere > 0.0 && u.misc.z > 0.0) {
                t = tSphere;
                strength = u.misc.z * exp(-max(t - 5.0, 0.0) * 0.10);
            }
        }
        if (t > 0.0) {
            float3 rel = o + d * t - c;
            // 地面上离得远的点，从用户看过去几乎是平视，查到的会是墙的颜色。
            // 把它拉近到两米以内再查，取到的才是地面的颜色。
            if (t == tFloor) rel.xz *= min(1.0, 2.0 / max(length(rel.xz), 1e-3));
            float3 dir = normalize(rel);
            float2 uv = float2(atan2(dir.z, dir.x) / (2.0 * M_PI_F) + 0.5,
                               acos(clamp(dir.y, -1.0, 1.0)) / M_PI_F);
            float3 guess = env.sample(envSmp, uv).rgb;
            // 打在地面上的：有地面颜色图就用它，比按方向猜的准。
            if (t == tFloor && u.floorMap.w > 0.5) {
                float2 fuv = ((o + d * t).xz - u.floorMap.xy) * u.floorMap.z + 0.5;
                if (all(fuv > float2(0.0)) && all(fuv < float2(1.0))) {
                    guess = floorTex.sample(smp, fuv, level(0.0)).rgb * 1.18;
                }
            }
            // 压暗一点：让人一眼看出这是猜的，不是扫到的。
            bg = mix(bg, guess * 0.72, u.envCenter.w * strength);

            // 生成式的预测：同一个打到的点，换成从「那张全景图的中心」看过去的方向去取。
            // alpha 只在原来没扫到的方向上是 1，所以它只补缺的地方。
            if (u.genCenter.w > 0.0) {
                float3 gd = normalize(o + d * t - u.genCenter.xyz);
                float2 guv = float2(atan2(gd.z, gd.x) / (2.0 * M_PI_F) + 0.5,
                                    acos(clamp(gd.y, -1.0, 1.0)) / M_PI_F);
                // alpha 约 0.5 的地方另有一层预测表面在画，这里只铺一层很糊的底色，
                // 免得同一件家具在表面上一份、背景上又一份。
                float ga = gen.sample(envSmp, guv, level(0.0)).a;
                float lod = (1.0 - smoothstep(0.55, 0.95, ga)) * 5.0;
                float3 gc = gen.sample(envSmp, guv, level(lod)).rgb;
                bg = mix(bg, gc * 0.86, saturate(ga * 2.0) * u.genCenter.w * strength);
            }
        }
    }

    bg += gridLine;
    float3 own = surf.a <= 0.0 ? bg : mix(bg, mix(surf.rgb, u.fog.rgb, f), surf.a);
    return float4(mix(own, genColor, genWeight), 1.0);
}

// 全景图的一个面：只要表面本身的颜色，不要背景、雾和预测；另外输出到中心的距离。
struct PanoOut {
    float4 color [[color(0)]];
    float dist [[color(1)]];
};

fragment PanoOut pano_fs(FullscreenOut in [[stage_in]],
                         texture2d<float> accum [[texture(0)]],
                         texture2d<float> accumZ [[texture(1)]],
                         texture2d<float> idTex [[texture(2)]],
                         texture2d_array<float> frames [[texture(3)]],
                         texture2d<float> env [[texture(4)]],
                         texture2d<float> gen [[texture(5)]],
                         texture2d_array<float> kfDepth [[texture(6)]],
                         texture2d<float> floorTex [[texture(8)]],
                         sampler smp [[sampler(0)]],
                         sampler envSmp [[sampler(1)]],
                         constant ResolveUniforms &u [[buffer(0)]],
                         const device Keyframe *kfs [[buffer(1)]]) {
    uint2 px = uint2(in.position.xy);
    float4 pn = u.invViewProj * float4(in.ndc, 0.0, 1.0);
    float4 pf = u.invViewProj * float4(in.ndc, 1.0, 1.0);
    float3 o = pn.xyz / pn.w;
    float3 d = normalize(pf.xyz / pf.w - o);
    float z;
    float4 surf = surfaceAt(px, d, accum, accumZ, idTex, frames, kfDepth, gen, smp, envSmp, u, kfs, z);
    PanoOut out;
    bool known = surf.a > 0.6;
    out.color = known ? float4(surf.rgb, 1.0) : float4(0.5, 0.5, 0.5, 0.0);
    out.dist = known ? z / max(dot(d, u.cameraFwd.xyz), 1e-3) : 0.0;
    // 给生成模型的提示（只有 floorMap 可用时才做，导出训练素材时不做）：没扫到、但视线打在地面上的地方，
    // 按已经见过的地面颜色铺上。人脚下那一圈永远扫不到（镜头是朝前的），不铺的话模型会把它画成
    // 一个台子，人物像站在箱子上。铺的范围要大（人周围 4 米），边界落在墙根或者画面外：
    // 试过只铺 2.5 米的一个圆，模型把那个圆画成了一张圆桌。
    if (!known && u.floorMap.w > 0.5 && d.y < -0.03) {
        float t = (u.params.x - o.y) / d.y;
        float3 hit = o + d * t;
        float2 fuv = (hit.xz - u.floorMap.xy) * u.floorMap.z + 0.5;
        if (t > 0.0 && t < 9.0 && distance(hit.xz, u.envCenter.xz) < 4.0
            && all(fuv > float2(0.0)) && all(fuv < float2(1.0))) {
            out.color = float4(floorTex.sample(smp, fuv, level(0.0)).rgb * 1.18, 1.0);
            out.dist = t;
        }
    }
    return out;
}

// ───────────────────────── 人偶 / 手机 ─────────────────────────

struct MeshVertex {
    float3 position [[attribute(0)]];
    float3 normal [[attribute(1)]];
    float2 uv [[attribute(2)]];
};

struct MeshInstance {
    float4x4 model;
    float4 color;  // rgb + 种类：0 受光, 1 贴屏幕纹理, 2 地面软阴影, 3 人物模型（贴自己的纹理再受光）
};

struct MeshUniforms {
    float4x4 viewProj;
    float4 lightDir;
    float4 cameraPos;
};

struct MeshOut {
    float4 position [[position]];
    float3 normal;
    float3 world;
    float2 uv;
    float4 color;
};

vertex MeshOut mesh_vs(MeshVertex in [[stage_in]],
                       uint iid [[instance_id]],
                       const device MeshInstance *instances [[buffer(1)]],
                       constant MeshUniforms &u [[buffer(2)]]) {
    MeshInstance inst = instances[iid];
    float4 world = inst.model * float4(in.position, 1.0);
    MeshOut o;
    o.position = u.viewProj * world;
    o.world = world.xyz;
    float3x3 m = float3x3(inst.model[0].xyz, inst.model[1].xyz, inst.model[2].xyz);
    o.normal = m * in.normal;
    o.uv = in.uv;
    o.color = inst.color;
    return o;
}

// 人物模型：每一帧的顶点都是烘焙好的。站姿、走路循环里相邻的两帧，三份顶点按权重混在一起。
struct CharacterVertex {
    packed_float3 position;
    packed_float3 normal;
};

struct CharacterUniforms {
    float4x4 model;
    // 三段动作各取相邻两帧插值，再按权重混在一起。
    float4 blend;  // x: 待机两帧之间的位置, y: 走路两帧之间的位置, z: 坐下两帧之间的位置
    float4 weight; // x: 走路的程度（0 站着，1 走路）, y: 坐下的程度（0 用站/走，1 用坐下那段）
};

vertex MeshOut character_vs(uint vid [[vertex_id]],
                            const device CharacterVertex *idleA [[buffer(0)]],
                            const device CharacterVertex *idleB [[buffer(7)]],
                            const device CharacterVertex *walkA [[buffer(3)]],
                            const device CharacterVertex *walkB [[buffer(4)]],
                            const device CharacterVertex *sitA [[buffer(8)]],
                            const device CharacterVertex *sitB [[buffer(9)]],
                            const device packed_float2 *uvs [[buffer(5)]],
                            constant MeshUniforms &u [[buffer(2)]],
                            constant CharacterUniforms &c [[buffer(6)]]) {
    float3 p = mix(float3(idleA[vid].position), float3(idleB[vid].position), c.blend.x);
    float3 n = mix(float3(idleA[vid].normal), float3(idleB[vid].normal), c.blend.x);
    if (c.weight.x > 0.001) {
        p = mix(p, mix(float3(walkA[vid].position), float3(walkB[vid].position), c.blend.y), c.weight.x);
        n = mix(n, mix(float3(walkA[vid].normal), float3(walkB[vid].normal), c.blend.y), c.weight.x);
    }
    if (c.weight.y > 0.001) {
        p = mix(p, mix(float3(sitA[vid].position), float3(sitB[vid].position), c.blend.z), c.weight.y);
        n = mix(n, mix(float3(sitA[vid].normal), float3(sitB[vid].normal), c.blend.z), c.weight.y);
    }
    float4 world = c.model * float4(p, 1.0);
    MeshOut o;
    o.position = u.viewProj * world;
    o.world = world.xyz;
    o.normal = float3x3(c.model[0].xyz, c.model[1].xyz, c.model[2].xyz) * normalize(n);
    o.uv = float2(uvs[vid]);
    o.color = float4(1.0, 1.0, 1.0, 3.0);
    return o;
}

// 假的摄影棚环境，给人物和手机的外壳照出反光：上亮下暗，头顶一块大柔光箱，两侧各一条灯带。
// 表面越粗糙，灯的边缘越糊。
static float3 studioEnv(float3 r, float rough) {
    float soft = mix(0.05, 0.45, rough);
    float3 c = mix(float3(0.07, 0.07, 0.08), float3(0.42, 0.44, 0.47), smoothstep(-0.4, 0.9, r.y));
    c += smoothstep(0.78 - soft, 0.78 + soft, r.y) * 1.5;
    c += smoothstep(0.80 - soft, 0.80 + soft, abs(r.x)) * smoothstep(-0.3, 0.3, r.y) * 0.8;
    return c;
}

// 带贴图的硬壳（机器人、手机）：漫反射 + 一盏主灯的高光 + 环境反光。
// 模型不带可用的粗糙度贴图，按颜色深浅估：黑的部分是亮面，白的外壳是缎面。贴图是 sRGB，先换成线性再算光。
static float3 glossy(float3 tex, float3 n, float3 v, float3 l) {
    float3 albedo = pow(tex, float3(2.2));
    float dark = 1.0 - smoothstep(0.015, 0.10, dot(albedo, float3(0.2126, 0.7152, 0.0722)));
    float rough = mix(0.45, 0.22, dark);
    float f0 = mix(0.04, 0.07, dark);
    float3 h = normalize(l + v);
    float ndl = saturate(dot(n, l));
    float ndv = saturate(dot(n, v));
    float ndh = saturate(dot(n, h));
    float a2 = rough * rough * rough * rough;
    float d = ndh * ndh * (a2 - 1.0) + 1.0;
    float spec = a2 / (3.14159 * d * d) * 0.25 / max(0.25, ndl + ndv);
    float grazing = pow(1.0 - ndv, 5.0);
    float fresnel = f0 + (max(1.0 - rough, f0) - f0) * grazing;
    float3 ambient = mix(float3(0.16, 0.15, 0.14), float3(0.50, 0.51, 0.53), 0.5 + 0.5 * n.y);
    float3 c = albedo * (ambient + ndl * 1.05);
    c += spec * ndl * 1.05 * (f0 + (1.0 - f0) * pow(1.0 - saturate(dot(h, v)), 5.0));
    c += studioEnv(reflect(-v, n), rough) * fresnel;
    return pow(saturate(c), float3(1.0 / 2.2));
}

fragment float4 mesh_fs(MeshOut in [[stage_in]],
                        texture2d<float> screenTex [[texture(0)]],
                        texture2d<float> characterTex [[texture(1)]],
                        sampler smp [[sampler(0)]],
                        constant MeshUniforms &u [[buffer(2)]]) {
    float3 base = in.color.rgb;
    if (in.color.a > 2.5) {
        // 折叠手机的两块屏幕：烘焙时把它们的 u 加了 2（内屏）或 4（外屏）做记号，这里显示相机画面。
        // 相机图是按传感器方向存的，长边顺着铰链，所以转四分之一圈贴上去。
        if (in.uv.x > 1.5) {
            float s = fract(in.uv.x);
            return float4(screenTex.sample(smp, float2(in.uv.y, 1.0 - s)).rgb, 1.0);
        }
        return float4(glossy(characterTex.sample(smp, in.uv).rgb, normalize(in.normal),
                             normalize(u.cameraPos.xyz - in.world), u.lightDir.xyz), 1.0);
    } else if (in.color.a > 1.5) {
        float r = length(in.uv - 0.5) * 2.0;
        return float4(0.0, 0.0, 0.0, (1.0 - smoothstep(0.15, 1.0, r)) * 0.5);
    }
    else if (in.color.a > 0.5) {
        return float4(screenTex.sample(smp, in.uv).rgb, 1.0);
    }
    float3 n = normalize(in.normal);
    float3 v = normalize(u.cameraPos.xyz - in.world);
    float ndl = saturate(dot(n, u.lightDir.xyz));
    float hemi = 0.5 + 0.5 * n.y;
    float rim = pow(1.0 - saturate(dot(n, v)), 3.0);
    float3 c = base * (0.30 + 0.25 * hemi + 0.55 * ndl) + rim * 0.18;
    return float4(c, 1.0);
}

// ───────────────────────── 人偶（距离场）─────────────────────────
//
// 人偶不是网格：身上每一段是一根两头粗细不同的圆棒，所有圆棒在距离场里平滑地并成一个整体，
// 关节处自然地连在一起，没有接缝。屏幕上每个像素沿视线一步步走到表面（球面步进），
// 再用距离场的梯度当法线。环境光取自「未扫区域预测」那张小全景图，
// 所以身上和面罩上映出来的是周围房间的颜色。

#define AVATAR_LIMBS 24

struct AvatarUniforms {
    float4x4 viewProj;
    float4x4 invViewProj;
    float4 cameraPos;
    float4 lightDir;    // xyz + 环境图是否可用
    float4 bounds;      // 包围球：中心 + 半径
    float4 head;        // 头的中心 + 半径
    float4 faceDir;     // 脸朝哪 + 圆棒数量
    float4 bodyColor;
    float4 visorColor;
    float4 a[AVATAR_LIMBS];  // 一头的位置 + 半径
    float4 b[AVATAR_LIMBS];  // 另一头的位置 + 半径
};

static float avatarDistance(float3 p, constant AvatarUniforms &u) {
    int n = int(u.faceDir.w);
    float d = 1e9;
    const float k = 0.032;  // 并在一起时过渡带的宽度
    for (int i = 0; i < n; i++) {
        float3 a = u.a[i].xyz;
        float3 ba = u.b[i].xyz - a;
        float3 pa = p - a;
        float h = saturate(dot(pa, ba) / max(dot(ba, ba), 1e-8));
        float di = length(pa - ba * h) - mix(u.a[i].w, u.b[i].w, h);
        float m = max(k - abs(d - di), 0.0) / k;
        d = min(d, di) - m * m * k * 0.25;
    }
    return d;
}

struct AvatarOut {
    float4 color [[color(0)]];
    float depth [[depth(any)]];
};

static float3 envLookup(texture2d<float> env, sampler smp, float3 dir) {
    float2 uv = float2(atan2(dir.z, dir.x) / (2.0 * M_PI_F) + 0.5,
                       acos(clamp(dir.y, -1.0, 1.0)) / M_PI_F);
    return env.sample(smp, uv).rgb;
}

fragment AvatarOut avatar_fs(FullscreenOut in [[stage_in]],
                             texture2d<float> env [[texture(0)]],
                             sampler envSmp [[sampler(0)]],
                             constant AvatarUniforms &u [[buffer(0)]]) {
    float4 pn = u.invViewProj * float4(in.ndc, 0.0, 1.0);
    float4 pf = u.invViewProj * float4(in.ndc, 1.0, 1.0);
    float3 o = pn.xyz / pn.w;
    float3 d = normalize(pf.xyz / pf.w - o);

    // 先跟包围球求交，球外的像素直接不要。
    float3 oc = o - u.bounds.xyz;
    float bq = dot(oc, d);
    float disc = bq * bq - (dot(oc, oc) - u.bounds.w * u.bounds.w);
    if (disc < 0.0) discard_fragment();
    float sq = sqrt(disc);
    float t = max(-bq - sq, 0.0);
    float tEnd = -bq + sq;

    bool hit = false;
    for (int i = 0; i < 64; i++) {
        float dist = avatarDistance(o + d * t, u);
        if (dist < 0.0012 * max(t, 1.0)) {
            hit = true;
            break;
        }
        t += dist * 0.92;
        if (t > tEnd) break;
    }
    if (!hit) discard_fragment();

    float3 p = o + d * t;
    const float2 e = float2(0.002, -0.002);
    float3 n = normalize(e.xyy * avatarDistance(p + e.xyy, u) + e.yyx * avatarDistance(p + e.yyx, u)
                         + e.yxy * avatarDistance(p + e.yxy, u) + e.xxx * avatarDistance(p + e.xxx, u));
    float3 v = -d;

    // 缝里的暗：沿法线往外走几步，周围有别的表面挡着的地方（腋下、两腿之间）会暗下来。
    float ao = 1.0;
    for (int i = 1; i <= 3; i++) {
        float h = 0.025 * float(i);
        ao -= max(h - avatarDistance(p + n * h, u), 0.0) * (1.6 / float(i));
    }
    ao = saturate(ao);

    float3 l = u.lightDir.xyz;
    float ndl = saturate(dot(n, l));
    float hemi = 0.5 + 0.5 * n.y;
    float fres = pow(1.0 - saturate(dot(n, v)), 3.0);
    float3 ambient = mix(float3(0.34, 0.36, 0.42), float3(0.62, 0.64, 0.68), hemi);
    float3 refl = reflect(d, n);
    float3 sky = mix(float3(0.10, 0.11, 0.14), float3(0.55, 0.60, 0.70), saturate(refl.y * 0.5 + 0.5));
    if (u.lightDir.w > 0.5) {
        // 环境图是按方向存的平均颜色，本来就很糊，直接当漫反射的环境光用。
        // 只借亮度和一点点色调：整块颜色映上去像蹭脏了。
        float3 e0 = envLookup(env, envSmp, n);
        e0 = mix(float3(dot(e0, float3(0.3, 0.6, 0.1))), e0, 0.3);
        ambient = mix(ambient, e0 * 0.8 + 0.16, 0.5);
        sky = mix(sky, envLookup(env, envSmp, refl) * 1.25, 0.8);
    }

    // 面罩：头上朝着脸的那一块。
    float3 hd = p - u.head.xyz;
    float onHead = step(length(hd), u.head.w * 1.12);
    float facing = dot(normalize(hd), u.faceDir.xyz);
    float visor = onHead * smoothstep(0.42, 0.50, facing) * smoothstep(-0.62, -0.50, -abs(normalize(hd).y - 0.08));

    float3 body = u.bodyColor.rgb * (ambient * 0.78 + ndl * 0.50) * (0.45 + 0.55 * ao);
    float3 hv = normalize(l + v);
    // 身上只要一圈柔和的轮廓光；房间的倒影留给面罩，映在身上会变成一道道色条。
    body += pow(saturate(dot(n, hv)), 40.0) * 0.08 + fres * (ambient * 0.35 + 0.06);

    float3 glass = u.visorColor.rgb * (0.25 + 0.3 * ndl) + sky * (0.22 + 0.65 * fres)
        + pow(saturate(dot(n, hv)), 180.0) * 0.9;
    float3 c = mix(body, glass, visor);

    AvatarOut out;
    out.color = float4(c, 1.0);
    float4 clip = u.viewProj * float4(p, 1.0);
    out.depth = clip.z / clip.w;
    return out;
}

// ───────────────────────── 整屏贴一张图 ─────────────────────────

// 「生成式画面」开着时用：把重画好的那张图（或者还没等到时的原始渲染）铺满屏幕。
fragment float4 blit_fs(FullscreenOut in [[stage_in]],
                        texture2d<float> tex [[texture(0)]],
                        sampler smp [[sampler(0)]]) {
    float2 uv = float2(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    return float4(tex.sample(smp, uv).rgb, 1.0);
}

// ---- 网络画好的图，贴在它自己的形状上（见 GenKeyframe.swift） ----
struct GenMeshUniforms {
    float4x4 viewProj;
    float4 params;  // x: 这张图现在用几成, y: 网格横向几格, z: 纵向几格, w: 锐化多少
    float4 screen;  // xy: 屏幕宽高（像素）, z: 图的边缘多宽的一圈用来淡出（占图宽的比例）
};

struct GenMeshOut {
    float4 position [[position]];
    float2 uv;
};

vertex GenMeshOut genmesh_vs(uint vid [[vertex_id]],
                             const device float4 *pos [[buffer(0)]],
                             constant GenMeshUniforms &u [[buffer(1)]]) {
    uint stride = uint(u.params.y) + 1;
    GenMeshOut o;
    o.position = u.viewProj * float4(pos[vid].xyz, 1.0);
    o.uv = float2(float(vid % stride) / u.params.y, float(vid / stride) / u.params.z);
    return o;
}

fragment float4 genmesh_fs(GenMeshOut in [[stage_in]],
                           texture2d<float> tex [[texture(0)]],
                           sampler smp [[sampler(0)]],
                           constant GenMeshUniforms &u [[buffer(1)]]) {
    // 图的边缘淡出去，和旁边别的图（或者实时画面）接上。
    // 淡出的一圈给得宽：几张不同角度画的图拼在一起时，亮度和位置都对不太齐，
    // 边界窄了就是一道道直线把画面割开，宽了是一片渐变。两个方向相乘，角上是圆的。
    float2 e = min(in.uv, 1.0 - in.uv);
    float edge = smoothstep(0.0, u.screen.z, e.x) * smoothstep(0.0, u.screen.z, e.y);
    // 从侧面看，前后景交界的地方这张皮会被拉得很长（一小条图撑一大片屏幕）：拉得太长的地方不用它。
    float dens = min(length(dfdx(in.uv)) * u.screen.x, length(dfdy(in.uv)) * u.screen.y);
    float stretch = smoothstep(0.12, 0.3, dens);
    // 网络画的图只有 576×768，铺到屏幕上放大了两倍多，发软。做一点锐化（原图减去周围四个点的平均，再加回去）。
    float3 c = tex.sample(smp, in.uv).rgb;
    float sharpen = u.params.w;
    if (sharpen > 0.0) {
        float2 px = 1.25 / float2(tex.get_width(), tex.get_height());
        float3 around = (tex.sample(smp, in.uv + float2(px.x, 0)).rgb + tex.sample(smp, in.uv - float2(px.x, 0)).rgb
                         + tex.sample(smp, in.uv + float2(0, px.y)).rgb + tex.sample(smp, in.uv - float2(0, px.y)).rgb) * 0.25;
        c = clamp(c + (c - around) * sharpen, 0.0, 1.0);
    }
    return float4(c, u.params.x * edge * stretch);
}

// ---- 以人为中心的全景壳（见 GenShell.swift） ----
struct ShellUniforms {
    float4x4 viewProj;
    float4 center;     // xyz 壳的中心
    float4 oldCenter;  // 换中心时：xyz 旧中心；画进贴图时：w = 1 表示旧贴图是空的
    float4 eye;        // xyz 镜头位置
    float4 clip;       // 显示时：xyz 镜头朝向, w 沿朝向至少这么远才画
    float4 params;     // x 横向几格, y 纵向几格, z 高过这个高度不画, w 稳定后新图盖几成
    float4 extra;      // x 地面高度, y 面和视线夹角的余弦小于这个就算「不存在的面」
};

static float3 shellDir(float2 uv) {
    float phi = uv.x * 6.2831853, th = uv.y * 3.14159265;
    float s = sin(th);
    return float3(s * cos(phi), cos(th), s * sin(phi));
}

static float2 shellUV(float3 d) {
    float phi = atan2(d.z, d.x);
    if (phi < 0.0) phi += 6.2831853;
    return float2(phi / 6.2831853, acos(clamp(d.y, -1.0, 1.0)) / 3.14159265);
}

// 某个方向上壳有多远：在格点之间线性插值（和画出来的那张网一致）。
static float shellRadius(texture2d<float> rad, float2 uv, float2 cells) {
    float2 g = clamp(float2(uv.x * cells.x, uv.y * cells.y), float2(0.0), cells);
    float2 i0 = min(floor(g), cells - 1.0);
    float2 f = g - i0;
    uint2 a = uint2(i0);
    float r00 = rad.read(a).r, r10 = rad.read(a + uint2(1, 0)).r;
    float r01 = rad.read(a + uint2(0, 1)).r, r11 = rad.read(a + uint2(1, 1)).r;
    return mix(mix(r00, r10, f.x), mix(r01, r11, f.x), f.y);
}

// 把一张生成图画进全景贴图：贴图上每个点对应壳上一个位置，看它在那张图里落在哪。
fragment float4 shell_integrate_fs(FullscreenOut in [[stage_in]],
                                   texture2d<float> oldAtlas [[texture(0)]],
                                   texture2d<float> rad [[texture(1)]],
                                   texture2d<float> gen [[texture(2)]],
                                   texture2d<float> genDist [[texture(3)]],
                                   sampler smp [[sampler(0)]],
                                   constant ShellUniforms &u [[buffer(0)]]) {
    float2 uv = float2(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    float4 old = u.oldCenter.w > 0.5 ? float4(0.0) : oldAtlas.read(uint2(in.position.xy));
    float3 p = u.center.xyz + shellDir(uv) * shellRadius(rad, uv, u.params.xy);
    float4 c = u.viewProj * float4(p, 1.0);
    if (c.w < 0.05) return old;
    float2 n = c.xy / c.w;
    if (abs(n.x) >= 1.0 || abs(n.y) >= 1.0) return old;
    float2 guv = float2(n.x * 0.5 + 0.5, 0.5 - n.y * 0.5);
    // 这个点从那个机位看不看得见：和粗渲染在这个像素上量到的距离对一下。
    // 粗渲染那里是洞（0）的话，网络是凭空补的，壳上这里也是补出来的形状，照用。
    float2 size = float2(genDist.get_width(), genDist.get_height());
    float gd = genDist.read(uint2(min(guv * size, size - 1.0))).r;
    float pd = distance(p, u.eye.xyz);
    if (gd > 0.05 && abs(pd - gd) > 0.12 * pd + 0.15) return old;
    float2 e = min(guv, 1.0 - guv);
    float edge = smoothstep(0.0, 0.06, min(e.x, e.y));
    float3 g = gen.sample(smp, guv).rgb;
    if (old.a < 0.01) {
        // 第一次画到这里：整个盖上（太靠图边的先不画，等更正的机位）。
        if (edge < 0.5) return old;
        return float4(g * 0.5, 0.5);
    }
    // 之后和原来的混：可信度每次加 0.25，新图的分量从一半降到 params.w。
    float3 was = old.rgb / old.a;
    float w = max(u.params.w, 1.0 - old.a) * edge;
    float a = min(1.0, old.a + 0.25 * edge);
    return float4(mix(was, g, w) * a, a);
}

// 换中心：新壳上每个点，回旧壳上找同一个位置的颜色。位置对不上（那里以前被别的东西挡着）就留空。
fragment float4 shell_reproject_fs(FullscreenOut in [[stage_in]],
                                   texture2d<float> oldAtlas [[texture(0)]],
                                   texture2d<float> oldRad [[texture(1)]],
                                   texture2d<float> newRad [[texture(2)]],
                                   sampler smp [[sampler(0)]],
                                   constant ShellUniforms &u [[buffer(0)]]) {
    float2 uv = float2(in.ndc.x * 0.5 + 0.5, 0.5 - in.ndc.y * 0.5);
    float3 p = u.center.xyz + shellDir(uv) * shellRadius(newRad, uv, u.params.xy);
    float3 v = p - u.oldCenter.xyz;
    float len = max(length(v), 1e-4);
    float2 ouv = shellUV(v / len);
    float ro = shellRadius(oldRad, ouv, u.params.xy);
    if (abs(len - ro) > 0.12 * ro + 0.12) return float4(0.0);
    return oldAtlas.sample(smp, ouv, level(0.0));
}

// 把一张生成图的内容换成全景贴图里累积的那份。逐像素：这个像素看到的那个点（按粗渲染量到的距离）
// 在全景贴图里是哪个位置，就取那里的颜色。图还是这张图、这个视角，只是每个地方的颜色是所有图累积出来的，
// 所以前后两张图在同一个地方颜色一样，换图时不会跳。贴图里没有的地方（粗渲染是洞、或者从中心看不见）用原图。
fragment float4 shell_lookup_fs(FullscreenOut in [[stage_in]],
                                texture2d<float> atlas [[texture(0)]],
                                texture2d<float> rad [[texture(1)]],
                                texture2d<float> gen [[texture(2)]],
                                texture2d<float> genDist [[texture(3)]],
                                sampler smp [[sampler(0)]],
                                constant ShellUniforms &u [[buffer(0)]]) {
    uint2 px = uint2(in.position.xy);
    float3 g = gen.read(px).rgb;
    float gd = genDist.read(px).r;
    // 这里 viewProj 放的是那张图机位的逆矩阵。
    float4 a = u.viewProj * float4(in.ndc, 0.0, 1.0);
    float4 b = u.viewProj * float4(in.ndc, 1.0, 1.0);
    float3 dir = normalize(b.xyz / b.w - a.xyz / a.w);
    // 沿这个像素的视线找它和壳的交点（从粗渲染量到的距离出发，牛顿法迭代几步）。
    // 不能直接拿粗渲染的距离去算位置：那个距离逐像素有噪声，算出来的位置是乱的，取回来的颜色满是毛刺。
    // 画进贴图时用的是壳的形状，取的时候也必须用同一份形状，一来一回才对得上。
    float3 o = u.eye.xyz - u.center.xyz;
    float tt = gd > 0.05 ? gd : 3.0;
    float2 auv = float2(0.0);
    bool hit = false;
    for (int i = 0; i < 7; i++) {
        float3 v = o + dir * tt;
        float len = max(length(v), 1e-4);
        auv = shellUV(v / len);
        float f = len - shellRadius(rad, auv, u.params.xy);
        if (abs(f) < 0.003 * len + 0.003) { hit = true; break; }
        float slope = dot(v / len, dir);
        tt -= f / (abs(slope) > 0.3 ? slope : (slope < 0.0 ? -0.3 : 0.3));
        if (tt < 0.05 || tt > 40.0) break;
    }
    if (!hit) return float4(g, 1.0);
    // 交点得是这个像素真看到的那个面（和粗渲染量到的距离对得上），不然是被挡住的另一层。
    if (gd > 0.05 && abs(tt - gd) > 0.12 * tt + 0.15) return float4(g, 1.0);
    float4 t = atlas.sample(smp, auv, level(0.35));
    if (t.a < 0.05) return float4(g, 1.0);
    // 贴图里的颜色只在「和这张图自己周围差不多」的范围里用：夹到这个像素 3×3 邻域的最小最大值之间（留一点余量）。
    // 物体边缘上按距离找位置难免找偏（前景找成了背景），不夹的话边缘上全是别处串过来的脏点（第一版就是）；
    // 夹了之后，贴图只负责压掉每张图之间细微的抖动，压不掉的（真变了、或者找偏了）就以这张图为准。
    int2 ip = int2(px);
    int2 top = int2(int(gen.get_width()) - 1, int(gen.get_height()) - 1);
    float3 lo = g, hi = g;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            float3 s = gen.read(uint2(clamp(ip + int2(dx, dy), int2(0), top))).rgb;
            lo = min(lo, s);
            hi = max(hi, s);
        }
    }
    float3 held = clamp(t.rgb / t.a, lo - 0.02, hi + 0.02);
    return float4(mix(g, held, u.params.w), 1.0);
}

struct ShellOut {
    float4 position [[position]];
    float2 uv;
    float3 world;
};

vertex ShellOut shell_vs(uint vid [[vertex_id]],
                         const device float4 *pos [[buffer(0)]],
                         constant ShellUniforms &u [[buffer(1)]]) {
    uint stride = uint(u.params.x) + 1;
    ShellOut o;
    o.world = pos[vid].xyz;
    o.position = u.viewProj * float4(o.world, 1.0);
    // 深度按「往后推半米」的位置写：壳的形状是抹平过的，贴着人的地方（脚下的地面、身边的椅子）
    // 不能因为差几厘米就把人挡住；离人半米以上、确实在前面的东西才挡。
    float4 pushed = u.viewProj * float4(o.world + normalize(o.world - u.eye.xyz) * 0.5, 1.0);
    o.position.z = pushed.z / pushed.w * o.position.w;
    o.uv = float2(float(vid % stride) / u.params.x, float(vid / stride) / u.params.y);
    return o;
}

fragment float4 shell_fs(ShellOut in [[stage_in]],
                         texture2d<float> atlas [[texture(0)]],
                         sampler smp [[sampler(0)]],
                         constant ShellUniforms &u [[buffer(1)]]) {
    // 镜头和人之间的、人背后的东西不画；镜头升到头顶俯视时高处的东西不画。
    // （地面不裁：镜头和人之间的那片地面要留着。）
    if (dot(in.world - u.eye.xyz, u.clip.xyz) < u.clip.w && in.world.y > u.extra.x + 0.1) discard_fragment();
    if (in.world.y > u.params.z) discard_fragment();
    float4 t = atlas.sample(smp, in.uv);
    if (t.a < 0.03) discard_fragment();
    // 壳是从人的位置量的，只有一层。从人那里看，桌沿后面直接就是远处的墙；这张网在桌沿和墙之间连着一片
    // 「不存在的面」（顺着从人出发的视线方向拉出去的）。跟拍镜头在人后面两米多，会从侧面看到这些面，
    // 像一根根刺。这种面不画（露出下面垫底的那层）：面的朝向和从中心出发的视线几乎平行的就是。地面除外。
    float3 nrm = normalize(cross(dfdx(in.world), dfdy(in.world)));
    float facing = abs(dot(nrm, normalize(in.world - u.center.xyz)));
    float sheet = in.world.y < u.extra.x + 0.1 ? 1.0 : smoothstep(u.extra.y, u.extra.y * 1.8, facing);
    if (sheet < 0.02) discard_fragment();
    return float4(t.rgb / t.a, smoothstep(0.03, 0.3, t.a) * sheet);
}
