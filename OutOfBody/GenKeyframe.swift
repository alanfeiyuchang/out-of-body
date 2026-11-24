import Foundation
import Metal
import QuartzCore
import simd

/// 网络画好的一张图，连同它的形状，立在世界里。
///
/// 以前的做法：只留最新的一张，当成立在人偶距离上的一块平板重新投影。镜头一动就对不上——
/// 人一转身，画面就在「生成的」和「实时的」之间来回切（2026-11-23 回放录屏里每秒跳三次），还拉伸。
/// 现在：交给网络的那张粗渲染图本来就带着每个像素离镜头多远。把这个距离降成一张 48×64 格的网
/// （没扫到的格子用旁边的补上，再抹平），每个格点算出它在世界里的位置，图就成了一张贴在房间形状上的皮。
/// 从别的机位看它，远近关系是对的，不拉伸；留好几张（不同方向各一张），转回看过的方向时马上有图可用。
/// 同一个机位来了新图，新的在旧的上面淡入：两张都贴在同一个形状上，过渡看不出跳。
struct GenKeyframe {
    let texture: MTLTexture
    /// 每个格点在世界里的位置（SIMD4，w 不用），一行 cols+1 个，共 rows+1 行。
    let vertices: MTLBuffer
    /// 画这张图时镜头在哪、朝哪（相机到世界）。
    let camera: simd_float4x4
    let born: CFTimeInterval
    /// 同一个机位来了更新的图：这张再留一小会儿垫在下面，等新的淡入完就删。
    var replacedAt: CFTimeInterval?
}

enum GenMesh {
    static let cols = 48
    static let rows = 64
    static let smoothPasses: Int = {
        let v = UserDefaults.standard.integer(forKey: "genSmooth")
        return v > 0 ? v : 60
    }()
    /// 远近差多少算「不是同一个面」，按距离的比例（`-genEdge x`）。
    /// 走动那段回放（单目深度噪声大）：5% 和 12% 时噪声被当成台阶留下来，窗框碎成一片片锯齿；25% 才干净。
    /// 所以只有很大的落差（两米外差半米以上）才当台阶留着。
    static let edge: Float = {
        let v = UserDefaults.standard.float(forKey: "genEdge")
        return v > 0 ? v : 0.25
    }()
    static var vertexCount: Int { (cols + 1) * (rows + 1) }
    static var indexCount: Int { cols * rows * 6 }

    static func indexBuffer(_ device: MTLDevice) -> MTLBuffer? {
        var idx: [UInt16] = []
        idx.reserveCapacity(indexCount)
        let stride = cols + 1
        for j in 0..<rows {
            for i in 0..<cols {
                let a = UInt16(j * stride + i), b = a + 1
                let c = UInt16((j + 1) * stride + i), d = c + 1
                idx += [a, c, b, b, c, d]
            }
        }
        return device.makeBuffer(bytes: idx, length: idx.count * 2)
    }

    /// dist：粗渲染图每个像素沿视线到镜头的距离，0 = 那里什么都没扫到。floorY：地面有多高（知道的话）。
    static func build(
        dist: [Float], width w: Int, height h: Int, viewProj: simd_float4x4, camera: simd_float4x4,
        floorY: Float?, device: MTLDevice
    ) -> MTLBuffer? {
        guard dist.count >= w * h else { return nil }
        let nx = cols + 1, ny = rows + 1
        var depth = [Float](repeating: 0, count: nx * ny)
        var known = [Bool](repeating: false, count: nx * ny)
        // 每个格点的视线方向，和这条视线打到地面要走多远（打不到地面是 0）。
        let inv = viewProj.inverse
        let eye = camera.columns.3.xyz
        var dirs = [Vec3](repeating: Vec3(0, 0, -1), count: nx * ny)
        var toFloor = [Float](repeating: 0, count: nx * ny)
        for j in 0..<ny {
            let ndcY = 1 - 2 * Float(j) / Float(rows)
            for i in 0..<nx {
                let ndcX = 2 * Float(i) / Float(cols) - 1
                let a = inv * SIMD4<Float>(ndcX, ndcY, 0, 1)
                let b = inv * SIMD4<Float>(ndcX, ndcY, 1, 1)
                let d = simd_normalize(b.xyz / b.w - a.xyz / a.w)
                dirs[j * nx + i] = d
                if let floorY, d.y < -0.05 {
                    let t = (floorY - eye.y) / d.y
                    if t > 0.3, t < 12 { toFloor[j * nx + i] = t }
                }
            }
        }
        // 每个格点取它周围一格大小范围内距离的中位数（平均数会被零星的噪点带偏）。有效像素不到三成就当没扫到。
        let rx = max(1, w / cols / 2), ry = max(1, h / rows / 2)
        var samples: [Float] = []
        for j in 0..<ny {
            let cy = j * (h - 1) / rows
            for i in 0..<nx {
                let cx = i * (w - 1) / cols
                samples.removeAll(keepingCapacity: true)
                var total = 0
                var y = max(0, cy - ry)
                while y <= min(h - 1, cy + ry) {
                    var x = max(0, cx - rx)
                    while x <= min(w - 1, cx + rx) {
                        let d = dist[y * w + x]
                        total += 1
                        if d > 0.05, d < 30 { samples.append(d) }
                        x += 2
                    }
                    y += 2
                }
                if samples.count * 10 >= total * 3 {
                    samples.sort()
                    depth[j * nx + i] = samples[samples.count / 2]
                    known[j * nx + i] = true
                }
            }
        }
        // 地面不靠量出来的深度：知道地面多高，视线打到地面上的点直接算。量出来离地面不到 12 厘米的格子算地面；
        // 没扫到、而视线 9 米内能打到地面的格子也当地面（和粗渲染里的地面提示一个道理）。
        var isFloor = [Bool](repeating: false, count: nx * ny)
        for k in 0..<nx * ny where toFloor[k] > 0 {
            if known[k] {
                if eye.y + dirs[k].y * depth[k] < (floorY ?? 0) + 0.12 {
                    depth[k] = toFloor[k]
                    isFloor[k] = true
                }
            } else if toFloor[k] < 9 {
                depth[k] = toFloor[k]
                known[k] = true
                isFloor[k] = true
            }
        }
        // 没扫到的格子：一圈一圈从旁边扫到的往里补。整张都没有就立在 3 米外。
        if !known.contains(true) {
            depth = [Float](repeating: 3, count: nx * ny)
        } else {
            var left = known.filter { !$0 }.count
            var guardCount = 0
            while left > 0, guardCount < nx + ny {
                guardCount += 1
                var next = depth
                var nextKnown = known
                for j in 0..<ny {
                    for i in 0..<nx where !known[j * nx + i] {
                        var sum: Float = 0
                        var n: Float = 0
                        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                            let x = i + dx, y = j + dy
                            if x >= 0, x < nx, y >= 0, y < ny, known[y * nx + x] {
                                sum += depth[y * nx + x]
                                n += 1
                            }
                        }
                        if n > 0 {
                            next[j * nx + i] = sum / n
                            nextKnown[j * nx + i] = true
                            left -= 1
                        }
                    }
                }
                depth = next
                known = nextKnown
            }
        }
        // 抹平。单目深度有噪声，不抹的话网上的坑和包会让贴在上面的图跟着扭（窗框这种直线成了波浪形）。
        // 但抹什么、怎么抹有讲究（都是真机上拖着绕着看时暴露出来的）：
        //   - 直接抹「沿视线的距离」不行：地面近、墙远，抹完墙根被地面拉近，成了一个弧面，绕着看竖线全是歪的。
        //     所以地面不抹（上面已经按地面高度算死了），其余的抹「水平方向离镜头多远」：
        //     一面竖直的墙从上到下这个量是同一个数，怎么抹都还是竖直的。
        //   - 不跨过大的落差（差得越多权重越小）：桌上的显示器和后面的墙不能抹成一个斜坡。
        //     门槛是距离的 25%：再小的话噪声会被当成台阶留下，窗框碎成锯齿（走动那段回放试过 5%、12%）。
        // `-genSmooth n` 改遍数，`-genEdge x` 改门槛。
        var flat = [Float](repeating: 1, count: nx * ny)   // 视线方向的水平分量有多长
        var range = [Float](repeating: 0, count: nx * ny)  // 水平方向离镜头多远
        for k in 0..<nx * ny {
            flat[k] = max(0.08, (dirs[k].x * dirs[k].x + dirs[k].z * dirs[k].z).squareRoot())
            range[k] = depth[k] * flat[k]
        }
        for _ in 0..<GenMesh.smoothPasses {
            var next = range
            for j in 0..<ny {
                for i in 0..<nx where !isFloor[j * nx + i] {
                    let r0 = range[j * nx + i]
                    let tol = GenMesh.edge * (r0 + 0.6)
                    var sum: Float = 0
                    var wsum: Float = 0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let x = min(nx - 1, max(0, i + dx)), y = min(ny - 1, max(0, j + dy))
                            if isFloor[y * nx + x] { continue }
                            let r = (range[y * nx + x] - r0) / tol
                            let k: Float = (dx == 0 ? 2 : 1) * (dy == 0 ? 2 : 1) / (1 + r * r * r * r)
                            sum += range[y * nx + x] * k
                            wsum += k
                        }
                    }
                    if wsum > 0 { next[j * nx + i] = sum / wsum }
                }
            }
            range = next
        }
        var out = [SIMD4<Float>](repeating: .zero, count: nx * ny)
        for k in 0..<nx * ny {
            var t = isFloor[k] ? depth[k] : range[k] / flat[k]
            // 不是地面的格子也不能钻到地面下面去。
            if !isFloor[k], toFloor[k] > 0 { t = min(t, toFloor[k]) }
            let p = eye + dirs[k] * min(20, max(0.3, t))
            out[k] = SIMD4(p.x, p.y, p.z, 1)
        }
        return device.makeBuffer(bytes: out, length: out.count * MemoryLayout<SIMD4<Float>>.stride)
    }
}
