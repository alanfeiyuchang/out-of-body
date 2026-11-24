import Foundation
import simd

/// 合成房间本身：墙、地板、几个盒子，和一条光线打上去是什么颜色。
///
/// `-roomSeed 0`（默认）是一直在用的那间房，评分基准靠它，不能动。
/// 其他种子随机换房间大小、家具摆放和配色，给「粗渲染 → 干净画面」的网络攒训练素材用：
/// 合成房间的好处是任何机位都能直接算出标准答案，包括第三人称机位。
struct SyntheticScene {
    struct Box {
        var lo: Vec3
        var hi: Vec3
        var color: Vec3
    }

    static let shared = SyntheticScene(seed: UserDefaults.standard.integer(forKey: "roomSeed"))
    /// `-noMover YES`：不要那个来回滑的箱子（攒训练素材时场景要静止）。
    static let moverDisabled = UserDefaults.standard.bool(forKey: "noMover")

    var roomLo = Vec3(-3.5, 0, -3)
    var roomHi = Vec3(3.5, 2.8, 3)
    var staticBoxes: [Box] = [
        Box(lo: Vec3(-1.9, 0, -2.7), hi: Vec3(-0.6, 0.75, -1.9), color: Vec3(0.55, 0.38, 0.22)),  // 桌子
        Box(lo: Vec3(2.2, 0, -1.1), hi: Vec3(3.3, 0.8, 1.1), color: Vec3(0.22, 0.36, 0.62)),  // 沙发
        Box(lo: Vec3(-3.45, 0, 0.9), hi: Vec3(-2.95, 1.9, 2.3), color: Vec3(0.30, 0.52, 0.36)),  // 柜子
        Box(lo: Vec3(0.7, 0, 2.1), hi: Vec3(1.3, 0.6, 2.7), color: Vec3(0.70, 0.22, 0.20)),  // 箱子
        Box(lo: Vec3(-0.18, 0, -0.18), hi: Vec3(0.18, 2.8, 0.18), color: Vec3(0.78, 0.76, 0.70)),  // 柱子
    ]
    var palette: [Vec3] = [
        Vec3(0.85, 0.35, 0.25), Vec3(0.25, 0.55, 0.80), Vec3(0.90, 0.72, 0.25),
        Vec3(0.35, 0.68, 0.45), Vec3(0.62, 0.40, 0.75), Vec3(0.92, 0.52, 0.62),
    ]
    var floorA = Vec3(0.62, 0.50, 0.38)
    var floorB = Vec3(0.50, 0.40, 0.29)
    /// 一米里有几块地砖。
    var tiles: Float = 2
    var wallColor = Vec3(0.86, 0.84, 0.80)
    var ceilingColor = Vec3(0.90, 0.90, 0.92)
    var skirting = Vec3(0.30, 0.28, 0.26)
    /// 墙上色块「画」的间距（米）。
    var pictureSpacing: Float = 1.7
    var light = simd_normalize(Vec3(0.4, 1.0, 0.3))

    init(seed: Int) {
        guard seed != 0 else { return }
        var state = UInt32(truncatingIfNeeded: seed) &* 2_654_435_761 &+ 12345
        func rand() -> Float {
            state = state &* 1_664_525 &+ 1_013_904_223
            return Float(state >> 8) / Float(1 << 24)
        }
        func range(_ a: Float, _ b: Float) -> Float { a + (b - a) * rand() }
        func hue(_ h: Float, _ s: Float, _ v: Float) -> Vec3 {
            let k = Vec3(5, 3, 1).map { n -> Float in
                let t = (n + h * 6).truncatingRemainder(dividingBy: 6)
                return v - v * s * max(0, min(t, min(4 - t, 1)))
            }
            return k
        }

        // 手机沿半径 1.5 m 的椭圆走，房间要把这条路包住。
        let hx = range(2.9, 4.2), hz = range(2.8, 3.6)
        roomLo = Vec3(-hx, 0, -hz)
        roomHi = Vec3(hx, range(2.5, 3.1), hz)

        // 家具靠墙摆：沿四面墙随机放 4~7 个盒子，都离房间中心 2 m 以外，不挡路。
        staticBoxes = []
        let count = 4 + Int(rand() * 4)
        for _ in 0..<count {
            let wall = Int(rand() * 4)
            let along = range(0.5, 1.8), depth = range(0.4, 0.9), tall = range(0.4, 2.0)
            let color = hue(rand(), range(0.25, 0.7), range(0.35, 0.8))
            let alongX = wall < 2  // 贴着 z 方向的两面墙，沿 x 摆
            let span = (alongX ? hx : hz) - along / 2 - 0.1
            let c = range(-span, span)
            var lo: Vec3, hi: Vec3
            if alongX {
                let z0 = wall == 0 ? -hz + 0.05 : hz - 0.05 - depth
                lo = Vec3(c - along / 2, 0, z0)
                hi = Vec3(c + along / 2, tall, z0 + depth)
            } else {
                let x0 = wall == 2 ? -hx + 0.05 : hx - 0.05 - depth
                lo = Vec3(x0, 0, c - along / 2)
                hi = Vec3(x0 + depth, tall, c + along / 2)
            }
            // 盒子离中心最近的那个角要在 2 m 以外。
            let nx = max(lo.x, min(0, hi.x)), nz = max(lo.z, min(0, hi.z))
            if (nx * nx + nz * nz).squareRoot() < 2.0 { continue }
            staticBoxes.append(Box(lo: lo, hi: hi, color: color))
        }
        // 一半的房间中间有根柱子。
        if rand() < 0.5 {
            let r = range(0.12, 0.25)
            staticBoxes.append(Box(
                lo: Vec3(-r, 0, -r), hi: Vec3(r, roomHi.y, r),
                color: hue(rand(), 0.1, range(0.6, 0.85))))
        }

        palette = (0..<6).map { _ in hue(rand(), range(0.4, 0.75), range(0.6, 0.95)) }
        let fh = range(0.04, 0.14), fs = range(0.15, 0.5), fv = range(0.4, 0.75)
        floorA = hue(fh, fs, fv)
        floorB = hue(fh, fs, fv * range(0.7, 0.92))
        tiles = [1, 2, 2, 3, 4][Int(rand() * 5)]
        wallColor = hue(rand(), range(0.0, 0.18), range(0.72, 0.92))
        ceilingColor = Vec3(repeating: range(0.85, 0.95))
        skirting = hue(rand(), 0.1, range(0.2, 0.5))
        pictureSpacing = range(1.3, 2.4)
        light = simd_normalize(Vec3(range(-0.6, 0.6), 1.0, range(-0.6, 0.6)))
    }

    /// 某一刻场景里所有的盒子（静止的 + 来回滑的那个橙色箱子）。
    func boxes(at time: Float) -> [Box] {
        if SyntheticScene.moverDisabled { return staticBoxes }
        let mz = -1.0 + 1.5 * sin(0.5 * time)
        let x0 = roomLo.x + 0.95
        return staticBoxes + [
            Box(
                lo: Vec3(x0, 0, mz - 0.25), hi: Vec3(x0 + 0.5, 0.5, mz + 0.25),
                color: Vec3(0.95, 0.55, 0.12))
        ]
    }

    /// 点在不在某个盒子里面（留 margin 的余量），或者出了房间。给虚拟机位避让用。
    func blocked(_ p: Vec3, boxes: [Box], margin: Float) -> Bool {
        let m = Vec3(repeating: margin)
        if simd_reduce_min(p - roomLo - m) < 0 || simd_reduce_min(roomHi - m - p) < 0 { return true }
        for b in boxes where simd_reduce_min(p - b.lo + m) > 0 && simd_reduce_min(b.hi + m - p) > 0 {
            return true
        }
        return false
    }

    /// 从 cam 这个位姿拍一张图（RGBA）。cam 的约定是「看 -z，+x 对应图像右边，+y 对应图像上边」。
    func render(cam: Mat4, boxes: [Box], width hw: Int, height hh: Int, focal hf: Float) -> [UInt8] {
        let o = cam.position
        let ax = cam.xAxis, ay = cam.yAxis, az = cam.zAxis
        var hi = [UInt8](repeating: 255, count: hw * hh * 4)
        hi.withUnsafeMutableBufferPointer { buf in
            let out = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: hh) { v in
                for u in 0..<hw {
                    let dx = (Float(u) + 0.5 - Float(hw) / 2) / hf
                    let dy = -(Float(v) + 0.5 - Float(hh) / 2) / hf
                    let (_, c) = self.trace(o, ax * dx + ay * dy - az, boxes)
                    let i = (v * hw + u) * 4
                    out[i] = UInt8(max(0, min(255, c.x * 255)))
                    out[i + 1] = UInt8(max(0, min(255, c.y * 255)))
                    out[i + 2] = UInt8(max(0, min(255, c.z * 255)))
                }
            }
        }
        return hi
    }

    func trace(_ o: Vec3, _ d: Vec3, _ boxes: [Box]) -> (Float, Vec3) {
        let inv = Vec3(1 / d.x, 1 / d.y, 1 / d.z)

        // 房间：人在盒子里面，找射线离开盒子的那个面。
        let t1 = (roomLo - o) * inv
        let t2 = (roomHi - o) * inv
        let tFar = simd_max(t1, t2)
        var best = min(tFar.x, min(tFar.y, tFar.z))
        var normal: Vec3
        if best == tFar.x {
            normal = Vec3(d.x > 0 ? -1 : 1, 0, 0)
        } else if best == tFar.y {
            normal = Vec3(0, d.y > 0 ? -1 : 1, 0)
        } else {
            normal = Vec3(0, 0, d.z > 0 ? -1 : 1)
        }
        var hitBox = -1

        for (i, b) in boxes.enumerated() {
            let a1 = (b.lo - o) * inv
            let a2 = (b.hi - o) * inv
            let lo = simd_min(a1, a2)
            let hi = simd_max(a1, a2)
            let tn = max(lo.x, max(lo.y, lo.z))
            let tf = min(hi.x, min(hi.y, hi.z))
            if tn > 0.05 && tn < tf && tn < best {
                best = tn
                hitBox = i
                if tn == lo.x {
                    normal = Vec3(d.x > 0 ? -1 : 1, 0, 0)
                } else if tn == lo.y {
                    normal = Vec3(0, d.y > 0 ? -1 : 1, 0)
                } else {
                    normal = Vec3(0, 0, d.z > 0 ? -1 : 1)
                }
            }
        }

        let p = o + d * best
        var base: Vec3
        if hitBox >= 0 {
            base = boxes[hitBox].color
            // 细线纹理，方便看 splat 的清晰度。
            let g = p * 4
            let fr = g - g.rounded(.down)
            let lines =
                (abs(normal.x) < 0.5 && fr.x < 0.07) || (abs(normal.y) < 0.5 && fr.y < 0.07)
                || (abs(normal.z) < 0.5 && fr.z < 0.07)
            if lines { base *= 0.72 }
        } else if normal.y > 0.5 {
            // 地板：棋盘格木色。
            let k = Int(floor(p.x * tiles)) + Int(floor(p.z * tiles))
            base = (k & 1) == 0 ? floorA : floorB
            // 6 毫米宽的砖缝：比体素细得多，只有贴图能留住。
            let fxz = Vec3(p.x * tiles, 0, p.z * tiles)
            let seam = fxz - fxz.rounded(.down)
            if seam.x < 0.006 * tiles || seam.z < 0.006 * tiles { base *= 0.45 }
        } else if normal.y < -0.5 {
            base = ceilingColor
        } else {
            // 墙：浅色底，挂几幅色块「画」，再加踢脚线。
            let sCoord = abs(normal.x) > 0.5 ? p.z : p.x
            let wall = abs(normal.x) > 0.5 ? (normal.x > 0 ? 0 : 1) : (normal.z > 0 ? 2 : 3)
            base = wallColor
            let cell = (sCoord + 10) / pictureSpacing
            let f = cell - floor(cell)
            if p.y > 1.05 && p.y < 1.95 && f > 0.2 && f < 0.72 {
                base = palette[(Int(floor(cell)) + wall * 2) % palette.count]
                let inner = (f > 0.3 && f < 0.62 && p.y > 1.25 && p.y < 1.75)
                if inner {
                    base = base * 0.6 + Vec3(0.35, 0.35, 0.35)
                    // 画心里 3 厘米的小棋盘格。
                    if (Int(floor(sCoord / 0.03)) + Int(floor(p.y / 0.03))) & 1 == 0 { base *= 0.78 }
                }
            } else {
                // 墙纸上 4 厘米一道的细条纹。
                let stripe = sCoord * 25
                if stripe - floor(stripe) < 0.2 { base *= 0.94 }
            }
            if p.y < 0.12 { base = skirting }
        }
        let shade = 0.55 + 0.45 * max(0, simd_dot(normal, light))
        return (best, base * shade)
    }
}

extension SIMD3 where Scalar == Float {
    fileprivate func map(_ f: (Float) -> Float) -> Vec3 { Vec3(f(x), f(y), f(z)) }
}
