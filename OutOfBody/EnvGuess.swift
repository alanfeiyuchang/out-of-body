import Foundation
import simd

/// 对「还没扫到的地方」的猜测。
///
/// 手机上跑不动生成式补全，所以这里做的是最便宜的一种预测：
/// 记下从用户往每个方向看到过的平均颜色（一张 96×48 的全景小图），
/// 没看过的方向用周围看过的颜色一层层往里补，最后整体糊一下。
/// 效果是一团模糊但颜色合理的房间底色，真实的 splat 一到就把它盖住。
final class EnvGuess {
    static let width = 96
    static let height = 48

    private var color = [SIMD3<Float>](repeating: .zero, count: width * height)
    private var known = [Bool](repeating: false, count: width * height)
    private(set) var hasData = false

    func reset() {
        color = [SIMD3<Float>](repeating: .zero, count: EnvGuess.width * EnvGuess.height)
        known = [Bool](repeating: false, count: EnvGuess.width * EnvGuess.height)
        hasData = false
    }

    /// direction 不用归一化：从用户指向那个点。
    func add(direction d: Vec3, r: UInt32, g: UInt32, b: UInt32) {
        let len = simd_length(d)
        if len < 1e-4 { return }
        let u = atan2(d.z, d.x) / (2 * .pi) + 0.5
        let v = acos(max(-1, min(1, d.y / len))) / .pi
        let x = min(EnvGuess.width - 1, Int(u * Float(EnvGuess.width)))
        let y = min(EnvGuess.height - 1, Int(v * Float(EnvGuess.height)))
        let i = y * EnvGuess.width + x
        let c = SIMD3<Float>(Float(r), Float(g), Float(b))
        if known[i] {
            color[i] += (c - color[i]) * 0.15
        } else {
            color[i] = c
            known[i] = true
            hasData = true
        }
    }

    /// 补全之后的全景图（RGBA8）。
    func filled() -> [UInt8] {
        var w = EnvGuess.width
        var h = EnvGuess.height

        // 一层层缩小：每一格是它下面四格里「已知」的那些的平均。
        var levels: [(w: Int, h: Int, c: [SIMD3<Float>], k: [Bool])] = [(w, h, color, known)]
        while w > 6 {
            let src = levels[levels.count - 1]
            let nw = w / 2
            let nh = h / 2
            var c = [SIMD3<Float>](repeating: .zero, count: nw * nh)
            var k = [Bool](repeating: false, count: nw * nh)
            for y in 0..<nh {
                for x in 0..<nw {
                    var sum = SIMD3<Float>.zero
                    var n: Float = 0
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let i = (2 * y + dy) * w + 2 * x + dx
                        if src.k[i] {
                            sum += src.c[i]
                            n += 1
                        }
                    }
                    if n > 0 {
                        c[y * nw + x] = sum / n
                        k[y * nw + x] = true
                    }
                }
            }
            levels.append((nw, nh, c, k))
            w = nw
            h = nh
        }

        // 最粗的一层还空着的格子用全局平均。
        var top = levels[levels.count - 1]
        var sum = SIMD3<Float>.zero
        var n: Float = 0
        for i in 0..<top.c.count where top.k[i] {
            sum += top.c[i]
            n += 1
        }
        let average = n > 0 ? sum / n : SIMD3<Float>(repeating: 20)
        for i in 0..<top.c.count where !top.k[i] { top.c[i] = average }
        levels[levels.count - 1] = top

        // 再一层层放大回去：空着的格子从上一层拿。
        for l in stride(from: levels.count - 2, through: 0, by: -1) {
            let coarse = levels[l + 1]
            var fine = levels[l]
            for y in 0..<fine.h {
                for x in 0..<fine.w where !fine.k[y * fine.w + x] {
                    fine.c[y * fine.w + x] =
                        coarse.c[min(coarse.h - 1, y / 2) * coarse.w + min(coarse.w - 1, x / 2)]
                }
            }
            levels[l] = fine
        }

        // 糊三遍，把补出来的方块边和已知区域的硬边都抹掉。水平方向首尾相接。
        var img = levels[0].c
        let fw = EnvGuess.width
        let fh = EnvGuess.height
        for _ in 0..<3 {
            var out = img
            for y in 0..<fh {
                for x in 0..<fw {
                    var s = SIMD3<Float>.zero
                    for dy in -1...1 {
                        let yy = min(fh - 1, max(0, y + dy))
                        for dx in -1...1 {
                            s += img[yy * fw + (x + dx + fw) % fw]
                        }
                    }
                    out[y * fw + x] = s / 9
                }
            }
            img = out
        }

        var bytes = [UInt8](repeating: 255, count: fw * fh * 4)
        for i in 0..<fw * fh {
            bytes[i * 4] = UInt8(max(0, min(255, img[i].x)))
            bytes[i * 4 + 1] = UInt8(max(0, min(255, img[i].y)))
            bytes[i * 4 + 2] = UInt8(max(0, min(255, img[i].z)))
        }
        return bytes
    }
}

/// 地面上没扫到的地方该是什么颜色。
///
/// 人拿着手机往前看，脚边那一圈地面总是扫不到，跟拍视角里正好是最显眼的一块。
/// 按方向猜颜色（EnvGuess）在这里不好使：那个方向上要是摆着一只红箱子，地面也跟着被猜成红的。
/// 地面是一块平面，直接在平面上记：扫到的地面按 25 cm 一格存平均颜色，
/// 没扫到的格子用周围扫到的格子一层层补过来，相当于在地面上做了一次很糊的补洞。
final class FloorGuess {
    static let size = 64
    static let cell: Float = 0.25

    private var color = [SIMD3<Float>](repeating: .zero, count: size * size)
    private var known = [Bool](repeating: false, count: size * size)
    /// 这张图的中心在世界里的位置（x、z）。第一次有数据时定下来；人走出去太远就挪过去重来。
    private(set) var origin: SIMD2<Float>?
    private(set) var dirty = false

    func reset() {
        color = [SIMD3<Float>](repeating: .zero, count: FloorGuess.size * FloorGuess.size)
        known = [Bool](repeating: false, count: FloorGuess.size * FloorGuess.size)
        origin = nil
        dirty = false
    }

    /// 相机现在在哪。离图的中心超过 5 米就以这里为中心重来（图一共 16 米见方）。
    func recenter(_ camera: Vec3) {
        let c = SIMD2(camera.x, camera.z)
        if let o = origin, simd_distance(o, c) < 5 { return }
        reset()
        origin = c
    }

    func add(x: Float, z: Float, r: UInt32, g: UInt32, b: UInt32) {
        guard let o = origin else { return }
        let n = FloorGuess.size
        let gx = Int(((x - o.x) / FloorGuess.cell + Float(n) / 2).rounded(.down))
        let gz = Int(((z - o.y) / FloorGuess.cell + Float(n) / 2).rounded(.down))
        if gx < 0 || gz < 0 || gx >= n || gz >= n { return }
        let i = gz * n + gx
        let c = SIMD3<Float>(Float(r), Float(g), Float(b))
        if known[i] {
            color[i] += (c - color[i]) * 0.05
        } else {
            color[i] = c
            known[i] = true
        }
        dirty = true
    }

    /// 补全之后的图（RGBA8）。做法跟 EnvGuess.filled 一样：逐层缩小求平均，再逐层放大填空，最后糊两遍。
    func filled() -> [UInt8] {
        dirty = false
        let n = FloorGuess.size
        var levels: [(n: Int, c: [SIMD3<Float>], k: [Bool])] = [(n, color, known)]
        var w = n
        while w > 2 {
            let src = levels[levels.count - 1]
            let nw = w / 2
            var c = [SIMD3<Float>](repeating: .zero, count: nw * nw)
            var k = [Bool](repeating: false, count: nw * nw)
            for y in 0..<nw {
                for x in 0..<nw {
                    var sum = SIMD3<Float>.zero
                    var cnt: Float = 0
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let i = (2 * y + dy) * w + 2 * x + dx
                        if src.k[i] {
                            sum += src.c[i]
                            cnt += 1
                        }
                    }
                    if cnt > 0 {
                        c[y * nw + x] = sum / cnt
                        k[y * nw + x] = true
                    }
                }
            }
            levels.append((nw, c, k))
            w = nw
        }
        var top = levels[levels.count - 1]
        var sum = SIMD3<Float>.zero
        var cnt: Float = 0
        for i in 0..<top.c.count where top.k[i] {
            sum += top.c[i]
            cnt += 1
        }
        let average = cnt > 0 ? sum / cnt : SIMD3<Float>(repeating: 60)
        for i in 0..<top.c.count where !top.k[i] { top.c[i] = average }
        levels[levels.count - 1] = top
        for l in stride(from: levels.count - 2, through: 0, by: -1) {
            let coarse = levels[l + 1]
            var fine = levels[l]
            for y in 0..<fine.n {
                for x in 0..<fine.n where !fine.k[y * fine.n + x] {
                    fine.c[y * fine.n + x] =
                        coarse.c[min(coarse.n - 1, y / 2) * coarse.n + min(coarse.n - 1, x / 2)]
                }
            }
            levels[l] = fine
        }
        var img = levels[0].c
        for _ in 0..<2 {
            var out = img
            for y in 0..<n {
                for x in 0..<n {
                    var a = SIMD3<Float>.zero
                    for dy in -1...1 {
                        let yy = min(n - 1, max(0, y + dy))
                        for dx in -1...1 { a += img[yy * n + min(n - 1, max(0, x + dx))] }
                    }
                    out[y * n + x] = a / 9
                }
            }
            img = out
        }
        var bytes = [UInt8](repeating: 255, count: n * n * 4)
        for i in 0..<n * n {
            bytes[i * 4] = UInt8(max(0, min(255, img[i].x)))
            bytes[i * 4 + 1] = UInt8(max(0, min(255, img[i].y)))
            bytes[i * 4 + 2] = UInt8(max(0, min(255, img[i].z)))
        }
        return bytes
    }
}
