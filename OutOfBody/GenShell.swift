import Foundation
import Metal
import QuartzCore
import simd

/// 以人为中心的「全景壳」：网络画的所有图都画进同一张全景贴图里，屏幕上显示的永远是这一个壳。
///
/// 之前的做法（GenKeyframe）是每张生成图各自立成一张皮，留十张按机位轮换着用。再怎么调，换图那一下
/// 两张图的内容和形状总是不完全一致，画面就是不连贯；拖着视角绕着看时尤其明显。
/// 现在只有一个东西：
///   - 壳的形状：从人所在的位置往四面八方看，每个方向上最近的表面有多远（从地图里渲六个方向量出来，
///     和 GenMesh 一样处理：地面按地面高度算、其余抹平「水平距离」）。经度 192 格 × 纬度 96 格的一张网。
///   - 壳的贴图：一张 4096×2048 的全景图。每来一张生成图，就把它按当时的机位「投影」到壳上，
///     和原来的颜色混在一起（第一次整张盖上，之后越来越慢，稳定后每张只换 15%）。
///   - 用法：每张生成图到了，先画进贴图，再把这张图的内容逐像素换成贴图里累积的那份（stabilized），
///     然后照旧立成一层贴合当时视角的皮去显示（GenKeyframe）。这样同一个地方不管哪张图、从哪看，颜色都是
///     同一份累积结果，换图时内容不跳；网络每张图编得不太一样的细节在贴图里被平均掉了。
///   - 直接显示这个壳（`-shellDraw YES`）试过，不好：它是从人的位置量的单层形状，跟拍镜头在两米外从侧面看，
///     物体边缘全是毛刺。所以壳只当「记性」用，不拿来画。
/// 手机看到的东西本来就都是从人的位置看出去的，所以「从人的位置看得见的那层表面」正好装得下地图里的全部内容。
/// 人走开 0.6 米以上就换一个中心：按新位置重新量形状，把旧贴图搬到新的壳上。
///
/// 贴图里存的是「颜色 × 可信度」和「可信度」（0 = 这里还没画过），取样后相除，画过和没画过的交界不会发黑。
final class GenShell {
    static let cols = 192
    static let rows = 96
    static let atlasWidth = 4096
    static let atlasHeight = 2048
    static let faceSize = 160
    /// 往贴图里画的时候用的那份形状更细（经度 512 格 × 纬度 256 格），而且只轻轻抹一下：见 build 里的说明。
    static let fineCols = 512
    static let fineRows = 256
    static var vertexCount: Int { (cols + 1) * (rows + 1) }
    static var indexCount: Int { cols * rows * 6 }

    struct FaceJob {
        let index: Int
        let generation: Int
        let view: simd_float4x4
        let proj: simd_float4x4
    }

    struct Uniforms {
        var viewProj: simd_float4x4 = matrix_identity_float4x4
        var center: SIMD4<Float> = .zero
        var oldCenter: SIMD4<Float> = .zero
        var eye: SIMD4<Float> = .zero
        /// 显示时：xyz 镜头朝向，w 沿朝向至少这么远才画。
        var clip: SIMD4<Float> = .zero
        /// x 横向几格, y 纵向几格（这次用的那份形状的）, z 高过这个高度不画, w 稳定后新图盖几成。
        var params: SIMD4<Float> = .zero
        /// x 地面高度（显示时：地面不裁）。
        var extra: SIMD4<Float> = .zero
    }

    private struct Built {
        let center: Vec3
        let radii: [Float]
        let vertices: MTLBuffer
        /// 细的那份形状：每个方向上真实的表面有多远。
        let radiusTex: MTLTexture
        let floorY: Float
    }

    private let device: MTLDevice
    private let integratePipeline: MTLRenderPipelineState
    private let reprojectPipeline: MTLRenderPipelineState
    private let drawPipeline: MTLRenderPipelineState
    private let lookupPipeline: MTLRenderPipelineState
    private let linear: MTLSamplerState
    private let trilinear: MTLSamplerState
    private let trilinearRepeat: MTLSamplerState
    /// 换内容时贴图里的颜色占几成（`-shellTrust x`，默认全用）。
    private let trust: Float = UserDefaults.standard.object(forKey: "shellTrust") == nil
        ? 1 : UserDefaults.standard.float(forKey: "shellTrust")
    private let indices: MTLBuffer
    private let queue = DispatchQueue(label: "outofbody.shell", qos: .userInitiated)
    private let lock = NSLock()

    // 现在用着的壳。
    private var center = Vec3(0, 0, 0)
    private var radii: [Float] = []
    private var vertices: MTLBuffer?
    private var radiusTex: MTLTexture?
    private var floorY: Float = -1e6
    private var atlas: [MTLTexture] = []
    private var front = 0
    private var painted = false

    // 正在量的新壳。
    private var capture: (center: Vec3, floorY: Float?, next: Int, faces: [[Float]?], generation: Int)?
    private var built: Built?
    private var building = false
    private var generation = 0
    private var lastBuild: CFTimeInterval = 0

    /// 壳已经有形状、贴图里也画过东西了：可以拿来显示。
    var ready: Bool { vertices != nil && painted }

    /// 稳定后每张新图盖几成（`-shellBlend x`）。
    private let blend: Float = {
        let v = UserDefaults.standard.float(forKey: "shellBlend")
        return v > 0 ? v : 0.15
    }()

    /// 镜头在动时每张新图盖几成（`-shellMoving x`）。
    private let movingBlend: Float = {
        let v = UserDefaults.standard.float(forKey: "shellMoving")
        return v > 0 ? v : 0.4
    }()
    /// 面和「从中心出发的视线」夹角的余弦小于这个，算桌沿和墙之间那种不存在的面，不画（`-shellSheet x`）。
    static let sheetCos: Float = {
        let v = UserDefaults.standard.float(forKey: "shellSheet")
        return v > 0 ? v : 0.14
    }()
    /// 显示用的那张网抹平几遍（`-shellSmooth n`）。
    static let smoothPasses: Int = {
        let v = UserDefaults.standard.integer(forKey: "shellSmooth")
        return v > 0 ? v : 12
    }()

    init?(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        func pipeline(_ vs: String, _ fs: String, _ format: MTLPixelFormat, depth: Bool, blend: Bool)
            -> MTLRenderPipelineState?
        {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = library.makeFunction(name: vs)
            d.fragmentFunction = library.makeFunction(name: fs)
            d.colorAttachments[0].pixelFormat = format
            if depth { d.depthAttachmentPixelFormat = .depth32Float }
            if blend {
                let a = d.colorAttachments[0]!
                a.isBlendingEnabled = true
                a.sourceRGBBlendFactor = .sourceAlpha
                a.destinationRGBBlendFactor = .oneMinusSourceAlpha
                a.sourceAlphaBlendFactor = .one
                a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            return try? device.makeRenderPipelineState(descriptor: d)
        }
        let ls = MTLSamplerDescriptor()
        ls.minFilter = .linear
        ls.magFilter = .linear
        ls.sAddressMode = .repeat
        let ts = MTLSamplerDescriptor()
        ts.minFilter = .linear
        ts.magFilter = .linear
        ts.mipFilter = .linear
        ts.maxAnisotropy = 4
        let tr = MTLSamplerDescriptor()
        tr.minFilter = .linear
        tr.magFilter = .linear
        tr.mipFilter = .linear
        tr.sAddressMode = .repeat
        guard let s3 = device.makeSamplerState(descriptor: tr) else { return nil }
        trilinearRepeat = s3
        guard
            let p1 = pipeline("fullscreen_vs", "shell_integrate_fs", .rgba8Unorm, depth: false, blend: false),
            let p2 = pipeline("fullscreen_vs", "shell_reproject_fs", .rgba8Unorm, depth: false, blend: false),
            let p3 = pipeline("shell_vs", "shell_fs", .bgra8Unorm, depth: true, blend: true),
            let p4 = pipeline("fullscreen_vs", "shell_lookup_fs", .rgba8Unorm, depth: false, blend: false),
            let s1 = device.makeSamplerState(descriptor: ls), let s2 = device.makeSamplerState(descriptor: ts)
        else { return nil }
        integratePipeline = p1
        reprojectPipeline = p2
        drawPipeline = p3
        lookupPipeline = p4
        linear = s1
        trilinear = s2

        var idx: [UInt32] = []
        idx.reserveCapacity(GenShell.indexCount)
        let stride = GenShell.cols + 1
        for j in 0..<GenShell.rows {
            for i in 0..<GenShell.cols {
                let a = UInt32(j * stride + i), b = a + 1
                let c = UInt32((j + 1) * stride + i), d = c + 1
                idx += [a, c, b, b, c, d]
            }
        }
        guard let ib = device.makeBuffer(bytes: idx, length: idx.count * 4) else { return nil }
        indices = ib

        let ad = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: GenShell.atlasWidth, height: GenShell.atlasHeight,
            mipmapped: true)
        ad.usage = [.renderTarget, .shaderRead]
        ad.storageMode = .private
        guard let a0 = device.makeTexture(descriptor: ad), let a1 = device.makeTexture(descriptor: ad)
        else { return nil }
        atlas = [a0, a1]
    }

    func reset() {
        lock.lock()
        vertices = nil
        radiusTex = nil
        radii = []
        painted = false
        capture = nil
        built = nil
        generation += 1
        lock.unlock()
    }

    // MARK: 量形状

    /// 每帧问一次：要不要渲一个方向。要的话返回机位，渲完把距离图交给 faceDone。
    /// user：人现在在哪（地面上的点）。第一次、人离开中心 0.6 米以上、或者离上次量过了两秒，就重新量一遍
    /// （地图一直在长，形状要跟着更新）。六个方向一帧渲一个。
    func nextFace(user: Vec3, floorY: Float?, now: CFTimeInterval) -> FaceJob? {
        lock.lock()
        defer { lock.unlock() }
        if capture == nil, built == nil, !building {
            let moved = simd_length(Vec3(user.x - center.x, 0, user.z - center.z))
            let first = vertices == nil
            guard first || moved > 0.6 || now - lastBuild > 2 else { return nil }
            var c = center
            if first || moved > 0.6 { c = Vec3(user.x, (floorY ?? user.y) + 1.25, user.z) }
            capture = (c, floorY, 0, Array(repeating: nil, count: 6), generation)
        }
        guard var cap = capture, cap.next < 6 else { return nil }
        let i = cap.next
        cap.next += 1
        capture = cap
        let (f, up) = GenShell.faces[i]
        return FaceJob(
            index: i, generation: cap.generation,
            view: Mat4.lookAt(eye: cap.center, target: cap.center + f, up: up), proj: GenShell.faceProj)
    }

    private static let faces: [(Vec3, Vec3)] = [
        (Vec3(1, 0, 0), Vec3(0, 1, 0)), (Vec3(-1, 0, 0), Vec3(0, 1, 0)),
        (Vec3(0, 0, 1), Vec3(0, 1, 0)), (Vec3(0, 0, -1), Vec3(0, 1, 0)),
        (Vec3(0, 1, 0), Vec3(0, 0, -1)), (Vec3(0, -1, 0), Vec3(0, 0, 1)),
    ]
    /// 比 90° 宽一点，相邻两个方向之间有重叠，接缝处不会漏。
    private static let faceProj = Mat4.perspective(fovY: 2 * atan(1.04), aspect: 1, near: 0.05, far: 80)

    func faceDone(_ job: FaceJob, dist: [Float]) {
        lock.lock()
        guard var cap = capture, cap.generation == job.generation else {
            lock.unlock()
            return
        }
        cap.faces[job.index] = dist
        capture = cap
        let complete = cap.faces.allSatisfy { $0 != nil }
        if complete {
            capture = nil
            building = true
        }
        let previous = simd_distance(cap.center, center) < 1e-4 ? radii : []
        lock.unlock()
        guard complete else { return }
        let faces = cap.faces.compactMap { $0 }
        queue.async {
            let result = self.build(center: cap.center, floorY: cap.floorY, faces: faces, previous: previous)
            self.lock.lock()
            if cap.generation == self.generation { self.built = result }
            self.building = false
            self.lock.unlock()
        }
    }

    static func direction(u: Float, v: Float) -> Vec3 {
        let phi = u * 2 * .pi, theta = v * .pi
        let s = sin(theta)
        return Vec3(s * cos(phi), cos(theta), s * sin(phi))
    }

    private func build(center: Vec3, floorY: Float?, faces: [[Float]], previous: [Float]) -> Built? {
        let cols = GenShell.cols, rows = GenShell.rows, n = GenShell.faceSize
        let nx = cols, ny = rows + 1  // 经度方向首尾相接，只算 cols 列，最后把第 0 列抄到末尾
        var depth = [Float](repeating: 0, count: nx * ny)
        var known = [Bool](repeating: false, count: nx * ny)
        var dirs = [Vec3](repeating: Vec3(0, 1, 0), count: nx * ny)
        var toFloor = [Float](repeating: 0, count: nx * ny)
        let vps = GenShell.faces.map {
            GenShell.faceProj * Mat4.lookAt(eye: center, target: center + $0.0, up: $0.1)
        }
        var samples: [Float] = []
        for j in 0..<ny {
            for i in 0..<nx {
                let d = GenShell.direction(u: Float(i) / Float(cols), v: Float(j) / Float(rows))
                let k = j * nx + i
                dirs[k] = d
                if let floorY, d.y < -0.05 {
                    let t = (floorY - center.y) / d.y
                    if t > 0.3, t < 12 { toFloor[k] = t }
                }
                // 这个方向落在六个面里的哪一个：分量最大的那个轴。
                let ax = abs(d.x), ay = abs(d.y), az = abs(d.z)
                let face = ay >= ax && ay >= az ? (d.y > 0 ? 4 : 5)
                    : (ax >= az ? (d.x > 0 ? 0 : 1) : (d.z > 0 ? 2 : 3))
                let c = vps[face] * SIMD4<Float>(center.x + d.x, center.y + d.y, center.z + d.z, 1)
                guard c.w > 1e-4 else { continue }
                let px = Int(((c.x / c.w) * 0.5 + 0.5) * Float(n)), py = Int((0.5 - (c.y / c.w) * 0.5) * Float(n))
                samples.removeAll(keepingCapacity: true)
                var total = 0
                for y in (py - 1)...(py + 1) {
                    for x in (px - 1)...(px + 1) where x >= 0 && x < n && y >= 0 && y < n {
                        total += 1
                        let v = faces[face][y * n + x]
                        if v > 0.05, v < 30 { samples.append(v) }
                    }
                }
                if total > 0, samples.count * 10 >= total * 3 {
                    samples.sort()
                    depth[k] = samples[samples.count / 2]
                    known[k] = true
                }
            }
        }
        // 地面：知道地面多高，打到地面的方向直接算（同 GenMesh）。
        var isFloor = [Bool](repeating: false, count: nx * ny)
        for k in 0..<nx * ny where toFloor[k] > 0 {
            if known[k] {
                if center.y + dirs[k].y * depth[k] < (floorY ?? 0) + 0.12 {
                    depth[k] = toFloor[k]
                    isFloor[k] = true
                }
            } else if toFloor[k] < 9 {
                depth[k] = toFloor[k]
                known[k] = true
                isFloor[k] = true
            }
        }
        // 没扫到的方向：一圈一圈从旁边扫到的往里补。什么都没有就当 3 米外有东西。
        if !known.contains(true) {
            depth = [Float](repeating: 3, count: nx * ny)
        } else {
            var left = known.filter { !$0 }.count
            var rounds = 0
            while left > 0, rounds < nx {
                rounds += 1
                var next = depth
                var nextKnown = known
                for j in 0..<ny {
                    for i in 0..<nx where !known[j * nx + i] {
                        var sum: Float = 0
                        var cnt: Float = 0
                        for (dx, dy) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                            let x = (i + dx + nx) % nx, y = j + dy
                            if y >= 0, y < ny, known[y * nx + x] {
                                sum += depth[y * nx + x]
                                cnt += 1
                            }
                        }
                        if cnt > 0 {
                            next[j * nx + i] = sum / cnt
                            nextKnown[j * nx + i] = true
                            left -= 1
                        }
                    }
                }
                depth = next
                known = nextKnown
            }
        }
        let filled = depth  // 补过洞、还没抹平的
        // 抹平「水平方向离中心多远」，不跨过大的落差，不和地面混（道理见 GenMesh.build）。
        var flat = [Float](repeating: 1, count: nx * ny)
        var range = [Float](repeating: 0, count: nx * ny)
        for k in 0..<nx * ny {
            flat[k] = max(0.08, (dirs[k].x * dirs[k].x + dirs[k].z * dirs[k].z).squareRoot())
            range[k] = depth[k] * flat[k]
        }
        for _ in 0..<GenShell.smoothPasses {
            var next = range
            for j in 0..<ny {
                for i in 0..<nx where !isFloor[j * nx + i] {
                    let r0 = range[j * nx + i]
                    let tol = GenMesh.edge * (r0 + 0.6)
                    var sum: Float = 0
                    var wsum: Float = 0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let x = (i + dx + nx) % nx, y = min(ny - 1, max(0, j + dy))
                            if isFloor[y * nx + x] { continue }
                            let r = (range[y * nx + x] - r0) / tol
                            let w: Float = (dx == 0 ? 2 : 1) * (dy == 0 ? 2 : 1) / (1 + r * r * r * r)
                            sum += range[y * nx + x] * w
                            wsum += w
                        }
                    }
                    if wsum > 0 { next[j * nx + i] = sum / wsum }
                }
            }
            range = next
        }
        // 出结果：每个格点离中心多远。和上一次量的差得不多就各取一半，形状不会一下一下地跳。
        let stride = cols + 1
        var out = [Float](repeating: 3, count: stride * ny)
        for j in 0..<ny {
            for i in 0...cols {
                let k = j * nx + i % nx
                var t = isFloor[k] ? depth[k] : range[k] / flat[k]
                if !isFloor[k], toFloor[k] > 0 { t = min(t, toFloor[k]) }
                t = min(20, max(0.25, t))
                let o = j * stride + i
                if previous.count == out.count, abs(previous[o] - t) < 0.3 * t { t = (t + previous[o]) * 0.5 }
                out[o] = t
            }
        }
        var positions = [SIMD4<Float>](repeating: .zero, count: stride * ny)
        for j in 0..<ny {
            for i in 0...cols {
                let d = GenShell.direction(u: Float(i) / Float(cols), v: Float(j) / Float(rows))
                let p = center + d * out[j * stride + i]
                positions[j * stride + i] = SIMD4(p.x, p.y, p.z, 1)
            }
        }
        // 第二份形状，往贴图里画的时候用：每个方向上「真实的」表面有多远，更细，只轻轻抹一下。
        // 为什么要两份：贴图上的一个点代表「从中心往这个方向看到的那个东西」。要是按抹平过的壳去各张生成图里找它，
        // 不同机位的图会把同一个点对到不同的东西上（抹平过的壳和真实形状不一样，差多少看从哪看），
        // 混在一起就是一片糊（第一版就是这样）。按真实的远近去找，同一个东西不管从哪张图里看，都落在贴图的同一个点上。
        // 显示用的那张网还是抹平的（不抹平的话网上全是坑，图跟着扭），代价是东西的位置会偏一点，但偏得稳定、不闪。
        let fc = GenShell.fineCols, fr = GenShell.fineRows
        let fstride = fc + 1, fny = fr + 1
        var fine = [Float](repeating: 3, count: fstride * fny)
        var fineFloor = [Bool](repeating: false, count: fstride * fny)
        for j in 0..<fny {
            let v = Float(j) / Float(fr)
            for i in 0..<fc {
                let uu = Float(i) / Float(fc)
                let d = GenShell.direction(u: uu, v: v)
                let ax = abs(d.x), ay = abs(d.y), az = abs(d.z)
                let face = ay >= ax && ay >= az ? (d.y > 0 ? 4 : 5)
                    : (ax >= az ? (d.x > 0 ? 0 : 1) : (d.z > 0 ? 2 : 3))
                let c = vps[face] * SIMD4<Float>(center.x + d.x, center.y + d.y, center.z + d.z, 1)
                var t: Float = 0
                if c.w > 1e-4 {
                    let px = Int(((c.x / c.w) * 0.5 + 0.5) * Float(n)), py = Int((0.5 - (c.y / c.w) * 0.5) * Float(n))
                    if px >= 0, px < n, py >= 0, py < n {
                        let raw = faces[face][py * n + px]
                        if raw > 0.05, raw < 30 { t = raw }
                    }
                }
                var tf: Float = 0
                if let floorY, d.y < -0.05 {
                    let x = (floorY - center.y) / d.y
                    if x > 0.3, x < 12 { tf = x }
                }
                var floorHere = false
                if t > 0 {
                    if tf > 0, center.y + d.y * t < (floorY ?? 0) + 0.12 {
                        t = tf
                        floorHere = true
                    }
                } else {
                    // 没量到：用粗的那份补过洞的结果（在格点之间插值）。
                    let gx = uu * Float(cols), gy = v * Float(rows)
                    let x0 = min(Int(gx), cols - 1), y0 = min(Int(gy), rows - 1)
                    let fx = gx - Float(x0), fy = gy - Float(y0)
                    func at(_ x: Int, _ y: Int) -> Float { filled[min(y, ny - 1) * nx + x % nx] }
                    t = (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy)
                        + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy
                    if tf > 0, tf < 9, t >= tf * 0.9 {
                        t = tf
                        floorHere = true
                    }
                }
                if !floorHere, tf > 0 { t = min(t, tf) }
                fine[j * fstride + i] = min(20, max(0.25, t))
                fineFloor[j * fstride + i] = floorHere
            }
            fine[j * fstride + fc] = fine[j * fstride]
            fineFloor[j * fstride + fc] = fineFloor[j * fstride]
        }
        for _ in 0..<2 {
            var next = fine
            for j in 0..<fny {
                for i in 0..<fc where !fineFloor[j * fstride + i] {
                    let r0 = fine[j * fstride + i]
                    let tol = 0.06 * (r0 + 0.5)
                    var sum: Float = 0
                    var wsum: Float = 0
                    for dy in -1...1 {
                        for dx in -1...1 {
                            let x = (i + dx + fc) % fc, y = min(fny - 1, max(0, j + dy))
                            if fineFloor[y * fstride + x] { continue }
                            let r = (fine[y * fstride + x] - r0) / tol
                            let w: Float = (dx == 0 ? 2 : 1) * (dy == 0 ? 2 : 1) / (1 + r * r * r * r)
                            sum += fine[y * fstride + x] * w
                            wsum += w
                        }
                    }
                    if wsum > 0 { next[j * fstride + i] = sum / wsum }
                }
                next[j * fstride + fc] = next[j * fstride]
            }
            fine = next
        }
        let td = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .r32Float, width: fstride, height: fny, mipmapped: false)
        td.usage = [.shaderRead]
        guard
            let vb = device.makeBuffer(
                bytes: positions, length: positions.count * MemoryLayout<SIMD4<Float>>.stride),
            let tex = device.makeTexture(descriptor: td)
        else { return nil }
        tex.replace(
            region: MTLRegionMake2D(0, 0, fstride, fny), mipmapLevel: 0, withBytes: fine,
            bytesPerRow: fstride * 4)
        return Built(center: center, radii: out, vertices: vb, radiusTex: tex, floorY: floorY ?? -1e6)
    }

    // MARK: 贴图

    private func atlasPass(_ cmd: MTLCommandBuffer, _ body: (MTLRenderCommandEncoder) -> Void) {
        let back = 1 - front
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = atlas[back]
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return }
        body(enc)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        if let blit = cmd.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: atlas[back])
            blit.endEncoding()
        }
        front = back
    }

    /// 新量好的壳到了就换上。换了中心的话，先把旧贴图搬到新壳上。每帧调一次。
    func commit(_ cmd: MTLCommandBuffer, now: CFTimeInterval) {
        lock.lock()
        let next = built
        built = nil
        lock.unlock()
        guard let next else { return }
        if painted, let oldRadius = radiusTex, simd_distance(next.center, center) > 1e-4 {
            var u = Uniforms()
            u.center = SIMD4(next.center.x, next.center.y, next.center.z, 0)
            u.oldCenter = SIMD4(center.x, center.y, center.z, 0)
            u.params = SIMD4(Float(GenShell.fineCols), Float(GenShell.fineRows), 0, 0)
            let source = atlas[front]
            atlasPass(cmd) { enc in
                enc.setRenderPipelineState(reprojectPipeline)
                enc.setFragmentTexture(source, index: 0)
                enc.setFragmentTexture(oldRadius, index: 1)
                enc.setFragmentTexture(next.radiusTex, index: 2)
                enc.setFragmentSamplerState(linear, index: 0)
                enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
            }
        }
        lock.lock()
        center = next.center
        radii = next.radii
        vertices = next.vertices
        radiusTex = next.radiusTex
        floorY = next.floorY
        lastBuild = now
        lock.unlock()
    }

    /// 把一张生成图画进贴图。gen：图；dist：交给网络的那张粗渲染每个像素离镜头多远（判断壳上的点从那个机位看不看得见）。
    /// motion：镜头现在动得多快（0~1）。镜头不动时每张新图只换一点（越来越稳）；镜头在动时换得快，
    /// 贴图跟得上现在这个角度看到的样子。
    func integrate(
        _ cmd: MTLCommandBuffer, gen: MTLTexture, dist: MTLTexture, viewProj: simd_float4x4,
        camera: simd_float4x4, motion: Float
    ) {
        guard let radiusTex else { return }
        var u = Uniforms()
        u.viewProj = viewProj
        u.center = SIMD4(center.x, center.y, center.z, 0)
        u.eye = camera.columns.3
        u.params = SIMD4(
            Float(GenShell.fineCols), Float(GenShell.fineRows), 0, blend + (movingBlend - blend) * motion)
        let source = atlas[front]
        let first = !painted
        atlasPass(cmd) { enc in
            enc.setRenderPipelineState(integratePipeline)
            enc.setFragmentTexture(source, index: 0)
            enc.setFragmentTexture(radiusTex, index: 1)
            enc.setFragmentTexture(gen, index: 2)
            enc.setFragmentTexture(dist, index: 3)
            enc.setFragmentSamplerState(linear, index: 0)
            // 第一次：旧贴图里是没初始化的内容，告诉着色器当它是空的。
            u.oldCenter.w = first ? 1 : 0
            enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        }
        painted = true
    }

    /// 把一张生成图的内容换成贴图里累积的那份（见 shell_lookup_fs），返回换好的图。先 integrate 再调它。
    func stabilized(
        _ cmd: MTLCommandBuffer, gen: MTLTexture, dist: MTLTexture, viewProj: simd_float4x4,
        camera: simd_float4x4
    ) -> MTLTexture? {
        guard painted, let radiusTex else { return nil }
        let d = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: gen.width, height: gen.height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        guard let out = device.makeTexture(descriptor: d) else { return nil }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = out
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard let enc = cmd.makeRenderCommandEncoder(descriptor: pass) else { return nil }
        var u = Uniforms()
        u.viewProj = viewProj.inverse
        u.center = SIMD4(center.x, center.y, center.z, 0)
        u.eye = camera.columns.3
        u.params = SIMD4(Float(GenShell.fineCols), Float(GenShell.fineRows), 0, trust)
        enc.setRenderPipelineState(lookupPipeline)
        enc.setFragmentTexture(atlas[front], index: 0)
        enc.setFragmentTexture(radiusTex, index: 1)
        enc.setFragmentTexture(gen, index: 2)
        enc.setFragmentTexture(dist, index: 3)
        enc.setFragmentSamplerState(trilinearRepeat, index: 0)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        return out
    }

    // MARK: 显示

    /// fwd / minDepth：沿镜头朝向不到 minDepth 的东西不画（镜头和人之间的、人背后的）。
    func draw(
        _ enc: MTLRenderCommandEncoder, viewProj: simd_float4x4, eye: Vec3, fwd: Vec3, minDepth: Float,
        roofCut: Float, depth: MTLDepthStencilState
    ) {
        guard painted, let vertices else { return }
        var u = Uniforms()
        u.viewProj = viewProj
        u.center = SIMD4(center.x, center.y, center.z, 0)
        u.eye = SIMD4(eye.x, eye.y, eye.z, 1)
        u.clip = SIMD4(fwd.x, fwd.y, fwd.z, minDepth)
        u.params = SIMD4(Float(GenShell.cols), Float(GenShell.rows), min(roofCut, 1e6), 0)
        u.extra = SIMD4(floorY, GenShell.sheetCos, 0, 0)
        enc.setRenderPipelineState(drawPipeline)
        // 要测深度：从镜头的位置看，壳的近处（桌子）会挡住远处（墙）。
        enc.setDepthStencilState(depth)
        enc.setCullMode(.none)
        enc.setVertexBuffer(vertices, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentTexture(atlas[front], index: 0)
        enc.setFragmentSamplerState(trilinear, index: 0)
        enc.drawIndexedPrimitives(
            type: .triangle, indexCount: GenShell.indexCount, indexType: .uint32, indexBuffer: indices,
            indexBufferOffset: 0)
    }
}
