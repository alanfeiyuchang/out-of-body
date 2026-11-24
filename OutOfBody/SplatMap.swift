import Foundation
import Metal
import simd

/// 一张关键帧在 GPU 上的描述：世界坐标 → 这张图里的 uv（齐次，除以 w 得到 0~1）。
/// 布局要跟 Shaders.metal 里的 Keyframe 一致。
struct KeyframeGPU {
    var worldToUV: Mat4 = matrix_identity_float4x4
    /// x = 代数（槽位被复用一次加一），y = 1 表示可用。
    var info = SIMD4<Float>(0, 0, 0, 0)
}

/// 持久的环境：一个体素一个 splat。
///
/// 每来一张深度图做两件事：
/// 1. 雕刻：已有的 splat 如果落在这张深度图「看穿了」的地方，说明那里现在是空的，扣可信度。
/// 2. 累积：把每个像素反投影到世界，落进体素里，新建或者更新。
///
/// 另外保留一份「实时层」：最近一帧的原始点，不进体素，直接画，移动的东西靠它立刻出现。
///
/// 清晰度靠贴图：splat 自己只有一个颜色（2 cm 一格，很糊），所以另外存一组高清关键帧。
/// 每个 splat 记着「哪张关键帧把我看得最清楚」，渲染时逐像素把世界坐标投回那张图取色。
/// 这一帧刚被看到的 splat 直接用最新的相机画面，所以镜头对着的地方是实时的。
final class SplatMap {
    /// GPU 直接读这个布局（24 字节）。
    /// color = r | g<<8 | b<<16 | 可信度<<24。
    /// tex = 关键帧槽位(6 位) | 槽位代数<<6 (8 位) | 大小档<<14 (2 位) | 最后一次被看到的帧号<<16。
    /// 大小档：这个 splat 被看到时相邻采样点隔多远。远处和斜着看的表面采样稀，圆片要画大才连得上。
    struct Splat {
        var x: Float
        var y: Float
        var z: Float
        var color: UInt32
        var tex: UInt32 = SplatMap.noTex
        /// 表面朝向（世界坐标的法线，每个分量 8 位），0 = 不知道。
        /// 知道朝向的圆片贴着表面画，墙和地面是平的；不知道的朝着镜头画。
        var normal: UInt32 = 0
    }

    @inline(__always)
    static func packNormal(_ n: Vec3) -> UInt32 {
        let x = UInt32(max(1, min(255, n.x * 127 + 128.5)))
        let y = UInt32(max(1, min(255, n.y * 127 + 128.5)))
        let z = UInt32(max(1, min(255, n.z * 127 + 128.5)))
        return x | (y << 8) | (z << 16)
    }

    @inline(__always)
    static func unpackNormal(_ v: UInt32) -> Vec3 {
        (Vec3(Float(v & 255), Float((v >> 8) & 255), Float((v >> 16) & 255)) - 128) / 127
    }

    /// 两个朝向按比例混一下。
    @inline(__always)
    static func blendNormal(_ old: UInt32, _ new: UInt32, _ k: Float) -> UInt32 {
        if new == 0 { return old }
        if old == 0 { return new }
        let a = unpackNormal(old)
        var b = unpackNormal(new)
        if simd_dot(a, b) < 0 { b = -b }
        let m = a + (b - a) * k
        let len = simd_length(m)
        return len > 1e-4 ? packNormal(m / len) : new
    }

    static let voxel: Float = 0.02
    static let maxHits: UInt32 = 12
    /// 被至少这么多帧看到过才画，路过的人不会一帧就烙进地图。
    static let minHitsToDraw: Float = {
        let v = UserDefaults.standard.float(forKey: "minHits")
        return v > 0 ? v : 3
    }()
    /// `-noFuse YES`：关掉沿视线融合，用来做前后对比。
    static let fusionDisabled = UserDefaults.standard.bool(forKey: "noFuse")

    /// 沿视线融合时，splat 每次朝新测量挪的最小比例（0 = 不挪）。`-moveGain 0` 可关掉做对比。
    static let moveGain: Float = {
        let d = UserDefaults.standard
        return d.object(forKey: "moveGain") != nil ? d.float(forKey: "moveGain") : 0.1
    }()

    static let noTex: UInt32 = 63
    /// 槽位号只有 6 位，63 留给「没有」，所以关键帧 + 两个实时槽位最多 62 个。
    /// 60 张 960×720 的图大约占 165 MB。
    static let keyframeSlots = 60
    /// 关键帧之后还有两个槽位轮流放「最新一帧」。
    static let totalSlots = keyframeSlots + 2

    let capacity: Int
    let buffer: MTLBuffer
    let liveCapacity = 320 * 240
    private let device: MTLDevice
    private let liveBuffers: [MTLBuffer]
    private let predictedBuffers: [MTLBuffer]
    private var predictedIndex = 0
    private var predictedCount = 0
    private var liveIndex = 0
    private var liveCount = 0
    private var count = 0
    private var table: [UInt64: Int32] = [:]
    /// 每个 splat 上次被哪一帧碰过，保证一帧只涨一次可信度。只在 CPU 上。
    private let stamps: UnsafeMutablePointer<UInt16>
    /// 每个 splat 当前用的那张关键帧是在多远处拍的；更近的关键帧来了才换。
    private let kfDist: UnsafeMutablePointer<Float>
    private var frameStamp: UInt16 = 0

    // 关键帧。除了 published* 和 texture 之外都只在 `queue` 上读写。
    private var texture: MTLTexture?
    /// 每张关键帧当时的深度图（米）。渲染时用它判断「这张图当时真的看到了这个点吗」。
    private var depthTexture: MTLTexture?
    /// 每张关键帧留一份小图（跟深度图一样大）和它的投影矩阵，只在 CPU 上：
    /// 用来判断「这个 splat 记着的那张关键帧，画的还是它现在的样子吗」。
    private var kfColor: UnsafeMutablePointer<UInt8>?
    private var kfColorW = 0
    private var kfColorH = 0
    private var kfUV = [Mat4](repeating: matrix_identity_float4x4, count: keyframeSlots)
    /// 两个颜色是不是「明显不是同一样东西」。
    /// 主要看色调（各通道占总亮度的比例）：同一块表面换个角度、相机自动曝光变了，亮度会差不少，
    /// 色调基本不变；橙色的箱子和浅棕的地板亮度差不多，色调差得远。亮度差到 1.8 倍以上也算。
    @inline(__always)
    static func colorChanged(
        _ r1: Int, _ g1: Int, _ b1: Int, _ r2: Int, _ g2: Int, _ b2: Int
    ) -> Bool {
        let s1 = Float(r1 + g1 + b1 + 30), s2 = Float(r2 + g2 + b2 + 30)
        let chroma = abs(Float(r1) / s1 - Float(r2) / s2) + abs(Float(g1) / s1 - Float(g2) / s2)
            + abs(Float(b1) / s1 - Float(b2) / s2)
        return chroma > chromaThreshold || max(s1, s2) > 1.8 * min(s1, s2)
    }

    static let chromaThreshold: Float = {
        let d = UserDefaults.standard
        return d.object(forKey: "chromaGap") != nil ? d.float(forKey: "chromaGap") : 0.14
    }()
    private var kfGen = [UInt32](repeating: 0, count: keyframeSlots)
    private var kfRef = [Int](repeating: 0, count: keyframeSlots)
    private var kfSeq = [Int](repeating: 0, count: keyframeSlots)
    private var promoteSeq = 0
    private var lastKeyPose: Mat4?
    private var published = [KeyframeGPU](repeating: KeyframeGPU(), count: totalSlots)
    private var liveSlot = keyframeSlots
    private var liveStamp: UInt32 = 0
    private var liveValid = false
    private var _keyframesUsed = 0

    /// 对没扫到的方向的颜色猜测。envTexture 很小，每来一张深度图整张重写一次。
    private let env = EnvGuess()
    let envTexture: MTLTexture
    private var envValid = false
    /// 地面上没扫到的地方的颜色（见 FloorGuess）。
    private let floorGuess = FloorGuess()
    let floorTexture: MTLTexture
    /// 那张图的中心 x、z，一格对应多少（1 / 整张图的边长），是否可用。
    private var floorInfo = SIMD4<Float>(0, 0, 0, 0)

    private let lock = NSLock()
    private let queue = DispatchQueue(label: "outofbody.splatmap", qos: .userInitiated)
    private var busy = false
    private var resetRequested = false
    private var _lastIntegrateMs = 0.0
    private var _alive = 0

    init?(device: MTLDevice, capacity: Int = 2_000_000) {
        self.device = device
        self.capacity = capacity
        let stride = MemoryLayout<Splat>.stride
        guard let b = device.makeBuffer(length: capacity * stride, options: .storageModeShared),
            let l0 = device.makeBuffer(length: liveCapacity * stride, options: .storageModeShared),
            let l1 = device.makeBuffer(length: liveCapacity * stride, options: .storageModeShared),
            let q0 = device.makeBuffer(
                length: SplatMap.predictedCapacity * stride, options: .storageModeShared),
            let q1 = device.makeBuffer(
                length: SplatMap.predictedCapacity * stride, options: .storageModeShared)
        else { return nil }
        predictedBuffers = [q0, q1]
        let ed = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: EnvGuess.width, height: EnvGuess.height,
            mipmapped: false)
        ed.usage = [.shaderRead]
        ed.storageMode = .shared
        let fd = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: FloorGuess.size, height: FloorGuess.size,
            mipmapped: false)
        fd.usage = [.shaderRead]
        fd.storageMode = .shared
        guard let et = device.makeTexture(descriptor: ed), let ft = device.makeTexture(descriptor: fd)
        else { return nil }
        envTexture = et
        floorTexture = ft
        buffer = b
        liveBuffers = [l0, l1]
        stamps = .allocate(capacity: capacity)
        stamps.initialize(repeating: 0, count: capacity)
        kfDist = .allocate(capacity: capacity)
        kfDist.initialize(repeating: 0, count: capacity)
        table.reserveCapacity(400_000)
    }

    deinit {
        stamps.deallocate()
        kfDist.deallocate()
        kfColor?.deallocate()
    }

    struct Snapshot {
        var count: Int
        var live: MTLBuffer
        var liveCount: Int
        var predicted: MTLBuffer
        var predictedCount: Int
        var alive: Int
        var integrateMs: Double
        var texture: MTLTexture?
        var depthTexture: MTLTexture?
        var keyframes: [KeyframeGPU]
        var liveSlot: Int
        var liveStamp: UInt32
        var liveValid: Bool
        var keyframesUsed: Int
        var envValid: Bool
        var floorInfo: SIMD4<Float>
    }

    var snapshot: Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return Snapshot(
            count: count, live: liveBuffers[liveIndex], liveCount: liveCount,
            predicted: predictedBuffers[predictedIndex], predictedCount: predictedCount,
            alive: _alive, integrateMs: _lastIntegrateMs, texture: texture,
            depthTexture: depthTexture, keyframes: published, liveSlot: liveSlot, liveStamp: liveStamp,
            liveValid: liveValid, keyframesUsed: _keyframesUsed, envValid: envValid,
            floorInfo: floorInfo)
    }

    func reset() {
        lock.lock()
        resetRequested = true
        lock.unlock()
    }

    /// 快速回放时设成 true：submit 等这一帧算完再返回，一帧都不丢。
    var blocking = false

    struct Quality {
        var drawn = 0
        /// 到真实表面距离的中位数、90 分位（米），以及超过 8 cm 的比例。
        var median: Float = 0
        var p90: Float = 0
        var far: Float = 0
        /// 真实表面上有多大比例在 6 cm 之内有 splat。
        var coverage: Float = 0
        /// 调试：按高度分三段（地面附近 / 中间 / 天花板附近）各自的数量和中位误差。
        var bands = ""
    }

    /// 合成场景里知道真实表面在哪，拿来给地图打分。truth 返回 nil 的点不算。
    func evaluate(truth: (Vec3) -> Float?, samples: [Vec3]) -> Quality {
        queue.sync {
            let splats = buffer.contents().bindMemory(to: Splat.self, capacity: capacity)
            var errors: [Float] = []
            errors.reserveCapacity(count)
            var band: [[Float]] = [[], [], []]
            // 调试：误差超过 8 cm 的点都在哪——地板下面、墙外面、天花板上面，还是悬在屋子中间。
            var where_ = [Int](repeating: 0, count: 4)
            var walls = [Int](repeating: 0, count: 4)  // -x（有平面）、+x、-z（有平面）、+z
            for i in 0..<count where Float(splats[i].color >> 24) >= SplatMap.minHitsToDraw {
                let p = Vec3(splats[i].x, splats[i].y, splats[i].z)
                if let e = truth(p) {
                    errors.append(e)
                    band[p.y < 0.25 ? 0 : (p.y > 2.4 ? 2 : 1)].append(e)
                    if e > 0.08 {
                        where_[p.y < 0 ? 0 : (p.y > 2.8 ? 2 : (abs(p.x) > 3.5 || abs(p.z) > 3 ? 1 : 3))] += 1
                        if p.y >= 0, p.y <= 2.8 {
                            if p.x < -3.5 { walls[0] += 1 } else if p.x > 3.5 { walls[1] += 1 }
                            else if p.z < -3 { walls[2] += 1 } else if p.z > 3 { walls[3] += 1 }
                        }
                    }
                }
            }
            let bands: String = band.map { b -> String in
                let sorted = b.sorted()
                return sorted.isEmpty
                    ? "0" : String(format: "%dk/%.1f/%.1f", sorted.count / 1000,
                                   sorted[sorted.count / 2] * 100, sorted[sorted.count * 9 / 10] * 100)
            }.joined(separator: " ")
                + "｜离谱的点 地下/墙外/顶上/悬空：" + where_.map { "\($0 / 1000)k" }.joined(separator: " ")
                + "｜墙外 -x/+x/-z/+z：" + walls.map { "\($0 / 1000)k" }.joined(separator: " ")
            guard !errors.isEmpty else { return Quality() }
            errors.sort()
            let n = errors.count

            // 覆盖率：把画出来的 splat 按 6 cm 的格子登记，真实表面上每个采样点看自己和周围的格子里有没有。
            var cells = Set<UInt64>()
            let invCell: Float = 1 / 0.06
            for i in 0..<count where Float(splats[i].color >> 24) >= SplatMap.minHitsToDraw {
                cells.insert(SplatMap.voxelKey(Vec3(splats[i].x, splats[i].y, splats[i].z), invCell))
            }
            var hit = 0
            for p in samples {
                var found = false
                search: for dx in -1...1 {
                    for dy in -1...1 {
                        for dz in -1...1 {
                            let q = p + Vec3(Float(dx), Float(dy), Float(dz)) * 0.06
                            if cells.contains(SplatMap.voxelKey(q, invCell)) {
                                found = true
                                break search
                            }
                        }
                    }
                }
                if found { hit += 1 }
            }
            return Quality(
                drawn: n, median: errors[n / 2], p90: errors[n * 9 / 10],
                far: Float(errors.reversed().prefix { $0 > 0.08 }.count) / Float(n),
                coverage: samples.isEmpty ? 0 : Float(hit) / Float(samples.count), bands: bands)
        }
    }

    /// 上一帧还没算完就丢掉这一帧，不排队。
    func submit(_ frame: DepthFrame) {
        if blocking {
            queue.sync {
                let t0 = DispatchTime.now().uptimeNanoseconds
                self.integrate(frame)
                let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
                self.lock.lock()
                self._lastIntegrateMs = ms
                self.lock.unlock()
            }
            return
        }
        lock.lock()
        if busy {
            lock.unlock()
            return
        }
        busy = true
        lock.unlock()
        queue.async {
            let t0 = DispatchTime.now().uptimeNanoseconds
            self.integrate(frame)
            let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
            self.lock.lock()
            self._lastIntegrateMs = ms
            self.busy = false
            self.lock.unlock()
        }
    }

    @inline(__always)
    private static func voxelKey(_ p: Vec3, _ invVoxel: Float) -> UInt64 {
        let ix = Int32((p.x * invVoxel).rounded(.down))
        let iy = Int32((p.y * invVoxel).rounded(.down))
        let iz = Int32((p.z * invVoxel).rounded(.down))
        return UInt64(UInt32(bitPattern: ix) & 0x1F_FFFF)
            | (UInt64(UInt32(bitPattern: iy) & 0x1F_FFFF) << 21)
            | (UInt64(UInt32(bitPattern: iz) & 0x1F_FFFF) << 42)
    }

    /// 稀疏点直接进地图：没有深度图可以对照，所以不雕刻，进来就算可信。
    func submit(points: [Splat]) {
        queue.async {
            let splats = self.buffer.contents().bindMemory(to: Splat.self, capacity: self.capacity)
            self.lock.lock()
            var n = self.count
            let wantsReset = self.resetRequested
            self.resetRequested = false
            self.lock.unlock()
            if wantsReset {
                self.table.removeAll(keepingCapacity: true)
                self.clearKeyframes()
                n = 0
            }
            let before = n
            let invVoxel = 1 / SplatMap.voxel
            for p in points where n < self.capacity {
                let key = SplatMap.voxelKey(Vec3(p.x, p.y, p.z), invVoxel)
                if self.table[key] != nil { continue }
                self.table[key] = Int32(n)
                splats[n] = Splat(
                    x: p.x, y: p.y, z: p.z,
                    color: (p.color & 0x00FF_FFFF) | (SplatMap.maxHits << 24))
                n += 1
            }
            self.lock.lock()
            self.count = n
            self._alive = wantsReset ? n : self._alive + (n - before)
            self.lock.unlock()
        }
    }

    // MARK: 全景图和预测

    /// 从某个位置往四面八方看到的地图，一张等距柱状全景图。给生成式补全当输入。
    /// 由 Renderer 用跟屏幕画面同一套管线渲出来（见 Renderer.requestPanorama）。
    struct Panorama {
        var width: Int
        var height: Int
        /// RGBA。没扫到的像素是中灰，alpha = 0；扫到的 alpha = 255。
        var rgba: [UInt8]
        /// 每个像素到中心的距离（米），没扫到的是 0。
        var distance: [Float]
        var center: Vec3
        /// 扫到的像素占比。
        var coverage: Float
    }

    /// 等距柱状图的像素 ↔ 方向。全工程只有这一处定义，着色器里的写法要跟它一致。
    static func panoDirection(x: Float, y: Float, width: Int, height: Int) -> Vec3 {
        let theta = (x / Float(width) - 0.5) * 2 * .pi
        let phi = y / Float(height) * .pi
        return Vec3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta))
    }

    static func panoPixel(_ d: Vec3, width: Int, height: Int) -> (x: Float, y: Float) {
        let len = max(simd_length(d), 1e-6)
        return (
            (atan2(d.z, d.x) / (2 * .pi) + 0.5) * Float(width),
            acos(max(-1, min(1, d.y / len))) / .pi * Float(height)
        )
    }

    /// 预测出来的表面（生成式补全的全景图 + 单目深度，见 ScenePredictor）。
    /// 跟真实的 splat 分开放：真实数据一到，挡在它前面或者跟它重合的预测点就被删掉。
    static let predictedCapacity = 262_144

    func setPredicted(_ points: [Splat]) {
        queue.async {
            let target = 1 - self.predictedIndex
            let n = min(points.count, SplatMap.predictedCapacity)
            points.withUnsafeBytes { raw in
                self.predictedBuffers[target].contents().copyMemory(
                    from: raw.baseAddress!, byteCount: n * MemoryLayout<Splat>.stride)
            }
            self.lock.lock()
            self.predictedIndex = target
            self.predictedCount = n
            self.lock.unlock()
        }
    }

    // MARK: 关键帧

    private func clearKeyframes() {
        kfRef = [Int](repeating: 0, count: SplatMap.keyframeSlots)
        kfSeq = [Int](repeating: 0, count: SplatMap.keyframeSlots)
        lastKeyPose = nil
        env.reset()
        floorGuess.reset()
        lock.lock()
        for i in 0..<SplatMap.totalSlots { published[i].info.y = 0 }
        predictedCount = 0
        liveValid = false
        envValid = false
        floorInfo = .zero
        _keyframesUsed = 0
        lock.unlock()
    }

    private func ensureTexture(_ w: Int, _ h: Int) -> MTLTexture? {
        if let t = texture { return t.width == w && t.height == h ? t : nil }
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .rgba8Unorm
        d.width = w
        d.height = h
        d.arrayLength = SplatMap.totalSlots
        d.usage = [.shaderRead]
        d.storageMode = .shared
        let t = device.makeTexture(descriptor: d)
        lock.lock()
        texture = t
        lock.unlock()
        return t
    }

    private func ensureDepthTexture(_ w: Int, _ h: Int) -> MTLTexture? {
        if let t = depthTexture { return t.width == w && t.height == h ? t : nil }
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.pixelFormat = .r32Float
        d.width = w
        d.height = h
        d.arrayLength = SplatMap.totalSlots
        d.usage = [.shaderRead]
        d.storageMode = .shared
        let t = device.makeTexture(descriptor: d)
        lock.lock()
        depthTexture = t
        lock.unlock()
        return t
    }

    /// 相机挪了 35 cm 或者转了 18° 以上，就值得留一张新的关键帧。
    /// 门槛太低的话原地转一圈就把槽位用光了，被挤掉的区域会退回成纯色圆片。
    private func shouldPromote(_ m: Mat4) -> Bool {
        guard let last = lastKeyPose else { return true }
        return simd_distance(m.position, last.position) > 0.35
            || simd_dot(m.zAxis, last.zAxis) < 0.951
    }

    /// 先用空槽位；满了就换掉「还在用它的 splat 最少」的那张，最近的 4 张不动。
    private func pickSlot() -> Int {
        if let free = (0..<SplatMap.keyframeSlots).first(where: { kfSeq[$0] == 0 }) { return free }
        var best = 0
        var bestRef = Int.max
        for s in 0..<SplatMap.keyframeSlots where promoteSeq - kfSeq[s] >= 4 && kfRef[s] < bestRef {
            best = s
            bestRef = kfRef[s]
        }
        return best
    }

    private static func worldToUV(_ f: DepthFrame) -> Mat4 {
        let w = Float(f.width), h = Float(f.height)
        // 相机坐标 (x, y, z) 看 -z：u·d = fx/w·x − cx/w·z，v·d = −fy/h·y − cy/h·z，d = −z。
        let p = Mat4(rows: [
            SIMD4(f.fx / w, 0, -f.cx / w, 0),
            SIMD4(0, -f.fy / h, -f.cy / h, 0),
            SIMD4(0, 0, 0, 0),
            SIMD4(0, 0, -1, 0),
        ])
        return p * f.cameraToWorld.inverse
    }

    /// 地图里被看过很多次的 splat 投到一帧上，返回（像素, 深度），给单目深度的对齐当对照点。
    /// 只挑看得够多的、被前面的东西挡住的不要，最多几百个。
    func alignmentSamples(for c: MonoCapture) -> [DepthAlign.Sample] {
        queue.sync {
            let splats = buffer.contents().bindMemory(to: Splat.self, capacity: capacity)
            let n = count
            guard n > 2000 else { return [] }
            let w = c.width, h = c.height
            let inv = c.cameraToWorld.inverse
            let r0 = Vec3(inv.columns.0.x, inv.columns.1.x, inv.columns.2.x)
            let r1 = Vec3(inv.columns.0.y, inv.columns.1.y, inv.columns.2.y)
            let r2 = Vec3(inv.columns.0.z, inv.columns.1.z, inv.columns.2.z)
            let it = inv.columns.3
            // 先在一张粗网格（8 像素一格）上求每格最近的深度，后面用它剔掉被挡住的点。
            let cw = (w + 7) / 8, ch = (h + 7) / 8
            var near = [Float](repeating: .infinity, count: cw * ch)
            var picked: [(u: Int, v: Int, z: Float)] = []
            let stride = max(1, n / 6000)
            var i = 0
            while i < n {
                defer { i += stride }
                let s = splats[i]
                let hits = s.color >> 24
                if hits < 3 { continue }
                let p = Vec3(s.x, s.y, s.z)
                let z = -(simd_dot(r2, p) + it.z)
                if z < 0.3 || z > 6 { continue }
                let u = Int(c.fx * (simd_dot(r0, p) + it.x) / z + c.cx)
                let v = Int(c.cy - c.fy * (simd_dot(r1, p) + it.y) / z)
                if u < 1 || v < 1 || u >= w - 1 || v >= h - 1 { continue }
                let g = (v / 8) * cw + u / 8
                if z < near[g] { near[g] = z }
                if hits >= SplatMap.maxHits { picked.append((u, v, z)) }
            }
            var out: [DepthAlign.Sample] = []
            let keepEvery = max(1, picked.count / 300)
            for (k, s) in picked.enumerated() where k % keepEvery == 0 {
                if s.z > near[(s.v / 8) * cw + s.u / 8] * 1.08 { continue }
                out.append(DepthAlign.Sample(u: s.u, v: s.v, z: s.z, kind: .map))
            }
            return out
        }
    }

    /// 快满了：把死掉的 splat 清出去，位置腾给新的。清完还是很满，就把离相机最远的那些丢掉。
    /// 走得越远地图越大，但容量是固定的；丢远处的是最不心疼的，走回去会重新扫出来。
    /// 返回清理之后的数量。
    private func compact(
        splats: UnsafeMutablePointer<Splat>, count n: Int, camera: Vec3
    ) -> Int {
        let slots = SplatMap.keyframeSlots
        var alive = 0
        for i in 0..<n where splats[i].color >> 24 != 0 { alive += 1 }

        // 活着的超过容量的 75% 才需要丢远处的：按距离做直方图，找出「留下 60% 容量」对应的半径。
        var cutoff2 = Float.infinity
        if alive > capacity * 3 / 4 {
            let binSize: Float = 0.1
            var bins = [Int](repeating: 0, count: 400)
            for i in 0..<n where splats[i].color >> 24 != 0 {
                let d = simd_distance(Vec3(splats[i].x, splats[i].y, splats[i].z), camera)
                bins[min(399, Int(d / binSize))] += 1
            }
            let keep = capacity * 3 / 5
            var acc = 0
            for (b, c) in bins.enumerated() {
                acc += c
                if acc > keep {
                    let r = Float(b) * binSize
                    cutoff2 = r * r
                    break
                }
            }
        }

        table.removeAll(keepingCapacity: true)
        var j = 0
        for i in 0..<n {
            let s = splats[i]
            if s.color >> 24 == 0 { continue }
            let p = Vec3(s.x, s.y, s.z)
            if simd_distance_squared(p, camera) > cutoff2 {
                let slot = Int(s.tex & 63)
                if slot < slots, kfSeq[slot] != 0, kfGen[slot] == (s.tex >> 6) & 0xFF {
                    kfRef[slot] -= 1
                }
                continue
            }
            if j != i {
                splats[j] = s
                stamps[j] = stamps[i]
                kfDist[j] = kfDist[i]
            }
            table[SplatMap.voxelKey(p, 1 / SplatMap.voxel)] = Int32(j)
            j += 1
        }
        DebugLog.write("MAP 清理：\(n) → \(j)" + (cutoff2.isFinite
            ? String(format: "（丢掉 %.1f 米以外的）", cutoff2.squareRoot()) : ""))
        return j
    }

    /// 深度落在某块已知平面附近的像素，直接换成视线和平面的交点。
    /// 单目深度里的墙和地面是鼓的、弯的，而 ARKit 认出来的平面是准的。
    /// 代价是贴着平面、比误差还矮的东西（地上一根线）会被压进平面里，只剩贴图。
    /// 返回每个像素被压到的那块平面的法线（打包过的），没被压的是 0。
    private func snapToPlanes(_ f: inout DepthFrame) -> [UInt32] {
        let w = f.width, h = f.height
        var snapped = [UInt32](repeating: 0, count: w * h)
        let m = f.cameraToWorld
        let ax = m.xAxis, ay = m.yAxis, az = m.zAxis, o = m.position
        struct Prepared {
            var n: Vec3
            var c: Vec3
            var r2: Float
            var k: Float  // (c - o)·n
        }
        let planes = f.planes.map {
            Prepared(n: $0.normal, c: $0.center, r2: $0.radius * $0.radius,
                     k: simd_dot($0.center - o, $0.normal))
        }
        let packed = f.planes.map { SplatMap.packNormal(simd_normalize($0.normal)) }
        for v in 0..<h {
            let dy = -(Float(v) + 0.5 - f.cy) / f.fy
            for u in 0..<w {
                let z = f.depth[v * w + u]
                if z <= 0 { continue }
                let dx = (Float(u) + 0.5 - f.cx) / f.fx
                // 方向的 -z 分量恒为 1，所以沿它走的参数就是深度。
                let d = ax * dx + ay * dy - az
                var best = z
                var bestGap = 0.04 + 0.06 * z
                var bestPlane = -1
                for (pi, p) in planes.enumerated() {
                    let dn = simd_dot(d, p.n)
                    if abs(dn) < 1e-3 { continue }
                    let t = p.k / dn
                    let gap = abs(t - z)
                    // 沿视线差得不多，而且离平面的垂直距离也不超过 10 cm。
                    if t > 0.15, gap < bestGap, gap * abs(dn) < 0.10 * simd_length(d),
                        simd_length_squared(o + d * t - p.c) < p.r2
                    {
                        bestGap = gap
                        best = t
                        bestPlane = pi
                    }
                }
                f.depth[v * w + u] = best
                if bestPlane >= 0 { snapped[v * w + u] = packed[bestPlane] }
            }
        }
        return snapped
    }

    /// 每个像素的表面朝向：被压到平面上的直接用平面的法线，其余的从深度图的坡度算。
    /// 取左右、上下各隔两个像素的点连成两条边，叉乘就是法线。
    private func surfaceNormals(_ f: DepthFrame, snapped: [UInt32]) -> [UInt32] {
        let w = f.width, h = f.height
        var out = snapped.isEmpty ? [UInt32](repeating: 0, count: w * h) : snapped
        let m = f.cameraToWorld
        let c0 = m.columns.0.xyz, c1 = m.columns.1.xyz, c2 = m.columns.2.xyz
        let fx = f.fx, fy = f.fy, cx = f.cx, cy = f.cy
        f.depth.withUnsafeBufferPointer { depth in
            for v in 0..<h {
                for u in 0..<w {
                    let i = v * w + u
                    if out[i] != 0 { continue }
                    let z = depth[i]
                    if z <= 0 { continue }
                    // 左右、上下各找一个有效的邻居：先试隔两个像素的，不行就隔一个，再不行就用自己
                    // （图像边上和物体边缘只有一侧有邻居）。
                    var ul = u, ur = u, vt = v, vb = v
                    if u >= 2, depth[i - 2] > 0 { ul = u - 2 } else if u >= 1, depth[i - 1] > 0 { ul = u - 1 }
                    if u + 2 < w, depth[i + 2] > 0 { ur = u + 2 } else if u + 1 < w, depth[i + 1] > 0 { ur = u + 1 }
                    if v >= 2, depth[i - 2 * w] > 0 { vt = v - 2 } else if v >= 1, depth[i - w] > 0 { vt = v - 1 }
                    if v + 2 < h, depth[i + 2 * w] > 0 { vb = v + 2 } else if v + 1 < h, depth[i + w] > 0 { vb = v + 1 }
                    if ul == ur || vt == vb { continue }
                    let zl = depth[v * w + ul], zr = depth[v * w + ur]
                    let zt = depth[vt * w + u], zb = depth[vb * w + u]
                    let uc = Float(u) + 0.5 - cx, vc = Float(v) + 0.5 - cy
                    let ucl = Float(ul) + 0.5 - cx, ucr = Float(ur) + 0.5 - cx
                    let vct = Float(vt) + 0.5 - cy, vcb = Float(vb) + 0.5 - cy
                    // 相机坐标里的点：x = (u-cx)/fx·z，y = -(v-cy)/fy·z，z分量 = -z。
                    let a = Vec3((ucr * zr - ucl * zl) / fx, -vc * (zr - zl) / fy, -(zr - zl))
                    let b = Vec3(uc * (zb - zt) / fx, -(vcb * zb - vct * zt) / fy, -(zb - zt))
                    var n = simd_cross(a, b)
                    let len = simd_length(n)
                    if len < 1e-9 { continue }
                    n /= len
                    // 朝向相机那一侧。
                    if simd_dot(n, Vec3(uc / fx * z, -vc / fy * z, -z)) > 0 { n = -n }
                    out[i] = SplatMap.packNormal(c0 * n.x + c1 * n.y + c2 * n.z)
                }
            }
        }
        return out
    }

    private func integrate(_ frame: DepthFrame) {
        var f = frame
        let splats = buffer.contents().bindMemory(to: Splat.self, capacity: capacity)
        let slots = SplatMap.keyframeSlots

        lock.lock()
        var n = count
        let wantsReset = resetRequested
        resetRequested = false
        let liveTarget = 1 - liveIndex
        let liveSlotNow = liveSlot
        lock.unlock()

        if wantsReset {
            table.removeAll(keepingCapacity: true)
            clearKeyframes()
            n = 0
        }
        frameStamp &+= 1
        if frameStamp == 0 { frameStamp = 1 }
        let stampBits = UInt32(frameStamp) << 16

        let snappedNormals = f.planes.isEmpty ? [] : snapToPlanes(&f)
        let normals = surfaceNormals(f, snapped: snappedNormals)
        if n > capacity * 9 / 10 {
            n = compact(splats: splats, count: n, camera: f.cameraToWorld.position)
            // 渲染那边读到的数量也得马上跟着变，不然尾巴上是搬走之后留下的旧数据。
            lock.lock()
            count = n
            lock.unlock()
        }

        // ---- 0. 贴图：最新一帧总是上传；相机挪得够远时再留一份当关键帧 ----
        var keySlot = -1
        var keyCode = SplatMap.noTex
        var liveSlotTarget = liveSlotNow
        var hasImage = false
        if !f.image.isEmpty, let tex = ensureTexture(f.imageWidth, f.imageHeight) {
            hasImage = true
            let region = MTLRegionMake2D(0, 0, f.imageWidth, f.imageHeight)
            let depthTex = ensureDepthTexture(f.width, f.height)
            func upload(_ slot: Int) {
                f.image.withUnsafeBytes { raw in
                    tex.replace(
                        region: region, mipmapLevel: 0, slice: slot, withBytes: raw.baseAddress!,
                        bytesPerRow: f.imageWidth * 4, bytesPerImage: 0)
                }
                f.depth.withUnsafeBytes { raw in
                    depthTex?.replace(
                        region: MTLRegionMake2D(0, 0, f.width, f.height), mipmapLevel: 0,
                        slice: slot, withBytes: raw.baseAddress!, bytesPerRow: f.width * 4,
                        bytesPerImage: 0)
                }
            }
            // 渲染器这时候读的是另一个实时槽位，写这个不会撕裂。
            liveSlotTarget = liveSlotNow == slots ? slots + 1 : slots
            upload(liveSlotTarget)
            if shouldPromote(f.cameraToWorld) {
                keySlot = pickSlot()
                promoteSeq += 1
                kfSeq[keySlot] = promoteSeq
                kfGen[keySlot] = kfGen[keySlot] % 255 + 1
                kfRef[keySlot] = 0
                // 先把这个槽位标成不可用再覆盖，还指着旧图的 splat 会退回自己的颜色。
                lock.lock()
                published[keySlot].info = SIMD4(Float(kfGen[keySlot]), 0, 0, 0)
                lock.unlock()
                upload(keySlot)
                lastKeyPose = f.cameraToWorld
                keyCode = UInt32(keySlot) | (kfGen[keySlot] << 6)
                // 小图留一份在 CPU 上。
                let bytes = f.width * f.height * 4
                if kfColor == nil || kfColorW != f.width || kfColorH != f.height {
                    kfColor?.deallocate()
                    kfColor = .allocate(capacity: bytes * slots)
                    kfColor?.initialize(repeating: 0, count: bytes * slots)
                    kfColorW = f.width
                    kfColorH = f.height
                }
                f.color.withUnsafeBufferPointer { src in
                    (kfColor! + keySlot * bytes).update(from: src.baseAddress!, count: bytes)
                }
                kfUV[keySlot] = SplatMap.worldToUV(f)
            }
        }

        /// 相邻采样点隔 spacing 米时用哪一档大小（对应 Shaders.metal 里的 1 / 1.6 / 2.4 / 3.6 倍）。
        @inline(__always)
        func sizeClass(_ spacing: Float) -> UInt32 {
            spacing < 0.025 ? 0 : (spacing < 0.04 ? 1 : (spacing < 0.06 ? 2 : 3))
        }

        /// 这个 splat 记着的那张关键帧里，它所在的位置是什么颜色；跟现在看到的差得多不多。
        /// 差得多说明那里变了（东西挪走了、灯开了），那张图不能再用。
        let kfBytes = kfColorW * kfColorH * 4
        @inline(__always)
        func textureStale(_ slot: Int, _ i: Int, _ r: UInt32, _ g: UInt32, _ b: UInt32) -> Bool {
            guard let kc = kfColor else { return false }
            let hp = kfUV[slot] * SIMD4(splats[i].x, splats[i].y, splats[i].z, 1)
            if hp.w < 0.05 { return false }
            let ux = hp.x / hp.w, uy = hp.y / hp.w
            if ux <= 0 || uy <= 0 || ux >= 1 || uy >= 1 { return false }
            let j = slot * kfBytes
                + (min(kfColorH - 1, Int(uy * Float(kfColorH))) * kfColorW
                    + min(kfColorW - 1, Int(ux * Float(kfColorW)))) * 4
            return SplatMap.colorChanged(
                Int(kc[j]), Int(kc[j + 1]), Int(kc[j + 2]), Int(r), Int(g), Int(b))
        }

        /// 这个 splat 这一帧被看到了（r、g、b 是这一帧在它那个像素上的颜色）：
        /// 记上帧号；如果这一帧是关键帧，看要不要把它的贴图换成这一张。
        @inline(__always)
        func touch(_ i: Int, _ dist: Float, _ cls: UInt32, _ r: UInt32, _ g: UInt32, _ b: UInt32) {
            var t = (splats[i].tex & 0xFFFF) | stampBits
            // 离得更近、看得更密了，就换成小一档的圆片。
            if cls < (t >> 14) & 3 { t = (t & ~UInt32(0xC000)) | (cls << 14) }
            if keySlot >= 0 {
                let oldSlot = Int(t & 63)
                let oldValid =
                    oldSlot < slots && kfSeq[oldSlot] != 0 && kfGen[oldSlot] == (t >> 6) & 0xFF
                // 换成这张新关键帧的三种情况：原来那张没了；这张拍得明显更近；
                // 原来那张里它的颜色跟现在对不上了。
                if !oldValid || dist < kfDist[i] * 0.8
                    || (oldSlot != keySlot && textureStale(oldSlot, i, r, g, b))
                {
                    if oldValid { kfRef[oldSlot] -= 1 }
                    kfRef[keySlot] += 1
                    kfDist[i] = dist
                    t = stampBits | keyCode | (t & 0xC000)
                }
            }
            splats[i].tex = t
        }

        let w = f.width
        let h = f.height
        let fx = f.fx, fy = f.fy, cx = f.cx, cy = f.cy

        let fuse = (f.fuseBase > 0 || f.fusePerMeter > 0) && !SplatMap.fusionDisabled
        /// 这一帧里哪些像素已经被现有的 splat「解释」了。
        var explained = [UInt8](repeating: 0, count: fuse ? w * h : 0)

        f.depth.withUnsafeBufferPointer { depth in
            f.color.withUnsafeBufferPointer { color in
                // ---- 1. 雕刻 ----
                let inv = f.cameraToWorld.inverse
                let r0 = Vec3(inv.columns.0.x, inv.columns.1.x, inv.columns.2.x)
                let r1 = Vec3(inv.columns.0.y, inv.columns.1.y, inv.columns.2.y)
                let r2 = Vec3(inv.columns.0.z, inv.columns.1.z, inv.columns.2.z)
                let it = inv.columns.3
                let camPos = f.cameraToWorld.position
                let invVoxelCarve = 1 / SplatMap.voxel
                let moveGain = SplatMap.moveGain
                var alive = 0
                for i in 0..<n {
                    let s = splats[i]
                    let hits = s.color >> 24
                    if hits == 0 { continue }
                    alive += 1
                    let p = Vec3(s.x, s.y, s.z)
                    let z = -(simd_dot(r2, p) + it.z)
                    if z < 0.15 || z > f.maxRange { continue }
                    // 取投影点周围 2x2 个像素里最近的那个深度来比。
                    // 斜着看地面时相邻像素的深度差好几厘米，只比一个像素就得留很大的余量，
                    // 结果贴着地面的残影永远雕不掉；跟四个里最近的比，余量可以收得很小。
                    let uf = fx * (simd_dot(r0, p) + it.x) / z + cx - 0.5
                    let vf = cy - fy * (simd_dot(r1, p) + it.y) / z - 0.5
                    if uf < 0 || vf < 0 || uf >= Float(w - 1) || vf >= Float(h - 1) { continue }
                    let u = Int(uf)
                    let v = Int(vf)
                    let i00 = v * w + u
                    let d0 = depth[i00], d1 = depth[i00 + 1]
                    let d2 = depth[i00 + w], d3 = depth[i00 + w + 1]
                    if d0 <= 0 || d1 <= 0 || d2 <= 0 || d3 <= 0 { continue }
                    let measured = min(min(d0, d1), min(d2, d3))
                    // 这个 splat 比实测表面明显更靠近相机 = 相机看穿了它。
                    if z < measured - (f.carveBase + f.carvePerMeter * z) {
                        let left = hits > 2 ? hits - 2 : 0
                        splats[i].color = (s.color & 0x00FF_FFFF) | (left << 24)
                        if left == 0 {
                            alive -= 1
                            let slot = Int(s.tex & 63)
                            if slot < slots, kfSeq[slot] != 0, kfGen[slot] == (s.tex >> 6) & 0xFF {
                                kfRef[slot] -= 1
                            }
                            splats[i].tex = SplatMap.noTex
                        }
                    } else if fuse {
                        // 沿视线融合：这个 splat 和它投影处的实测深度差不多远，
                        // 那这一帧在这里看到的就是它。刷新它，并把它盖住的像素标成「已解释」。
                        let un = Int(uf + 0.5)
                        let vn = Int(vf + 0.5)
                        let tol = f.fuseBase + f.fusePerMeter * z
                        let dn = depth[vn * w + un]
                        let ci = (vn * w + un) * 4
                        let changed = SplatMap.colorChanged(
                            Int(color[ci]), Int(color[ci + 1]), Int(color[ci + 2]),
                            Int(s.color & 0xFF), Int((s.color >> 8) & 0xFF),
                            Int((s.color >> 16) & 0xFF))
                        // 量到的表面在它前面一截、颜色又完全不一样：那是挡在它前面的另一样东西
                        //（比如箱子挡住了后面的地面），不是它自己。不认，否则地面会被染上箱子的颜色。
                        // 单目深度的容差很宽（两三米外有二十多厘米），光看距离分不出这两种情况。
                        let occluded = changed && dn < z - (0.02 + 0.02 * z)
                        if dn > 0, abs(dn - z) <= tol, !occluded {
                            // 一个体素在画面上占多少像素，就标多大一块。
                            let rp = min(3, Int(0.5 * SplatMap.voxel * fx / z))
                            let u0 = max(0, un - rp), u1 = min(w - 1, un + rp)
                            let v0 = max(0, vn - rp), v1 = min(h - 1, vn + rp)
                            for vv in v0...v1 {
                                for uu in u0...u1 where abs(depth[vv * w + uu] - z) <= tol {
                                    explained[vv * w + uu] = 1
                                }
                            }
                            var newHits = hits
                            let firstThisFrame = stamps[i] != frameStamp
                            if firstThisFrame {
                                stamps[i] = frameStamp
                                newHits = min(hits + 1, SplatMap.maxHits)
                            }
                            // 颜色慢慢跟；差得很多说明这里的样子变了，跟快一点。
                            let keep: UInt32 = changed ? 1 : 7
                            let add: UInt32 = 8 - keep
                            let nr = ((s.color & 0xFF) * keep + UInt32(color[ci]) * add + 4) / 8
                            let ng = (((s.color >> 8) & 0xFF) * keep + UInt32(color[ci + 1]) * add + 4) / 8
                            let nb = (((s.color >> 16) & 0xFF) * keep + UInt32(color[ci + 2]) * add + 4) / 8
                            splats[i].color = nr | (ng << 8) | (nb << 16) | (newHits << 24)
                            touch(
                                i, z, sizeClass(z / fx), UInt32(color[ci]), UInt32(color[ci + 1]),
                                UInt32(color[ci + 2]))
                            if firstThisFrame {
                                splats[i].normal = SplatMap.blendNormal(
                                    s.normal, normals[vn * w + un], max(1 / Float(hits + 1), 0.15))
                            }

                            // 往这次量到的深度挪一步：看到的次数越多挪得越少，
                            // 于是每个 splat 落在历次测量的平均值上，单帧的误差被平均掉。
                            if moveGain > 0, firstThisFrame {
                                let step = max(1 / Float(hits + 1), moveGain)
                                let k = 1 + (dn - z) * step / z
                                let np = camPos + (p - camPos) * k
                                splats[i].x = np.x
                                splats[i].y = np.y
                                splats[i].z = np.z
                                let oldKey = SplatMap.voxelKey(p, invVoxelCarve)
                                let newKey = SplatMap.voxelKey(np, invVoxelCarve)
                                if newKey != oldKey {
                                    // 挪出了原来的格子：登记表跟着改。新格子里已经有别的 splat，就并给它。
                                    if table[oldKey] == Int32(i) { table[oldKey] = nil }
                                    if let j = table[newKey], Int(j) != i, splats[Int(j)].color >> 24 != 0 {
                                        splats[i].color &= 0x00FF_FFFF
                                        alive -= 1
                                        let slot = Int(splats[i].tex & 63)
                                        if slot < slots, kfSeq[slot] != 0,
                                            kfGen[slot] == (splats[i].tex >> 6) & 0xFF
                                        {
                                            kfRef[slot] -= 1
                                        }
                                        splats[i].tex = SplatMap.noTex
                                    } else {
                                        table[newKey] = Int32(i)
                                    }
                                }
                            }
                        }
                    }
                }

                // ---- 1.5 预测出来的表面：这一帧在那个方向上有了真实数据，预测就让位 ----
                lock.lock()
                let predicted = predictedBuffers[predictedIndex].contents()
                    .bindMemory(to: Splat.self, capacity: SplatMap.predictedCapacity)
                let predictedN = predictedCount
                lock.unlock()
                for i in 0..<predictedN {
                    let s = predicted[i]
                    if s.color >> 24 == 0 { continue }
                    let p = Vec3(s.x, s.y, s.z)
                    let z = -(simd_dot(r2, p) + it.z)
                    if z < 0.15 { continue }
                    let uf = fx * (simd_dot(r0, p) + it.x) / z + cx
                    let vf = cy - fy * (simd_dot(r1, p) + it.y) / z
                    if uf < 0 || vf < 0 || uf >= Float(w) || vf >= Float(h) { continue }
                    let dm = depth[Int(vf) * w + Int(uf)]
                    // 真实表面明显在它前面：它被挡着，这一帧说明不了什么，留着。
                    // 否则要么被看穿了，要么真实表面就在这附近，都该删。
                    if dm > 0, z < dm + 0.06 + 0.06 * z { predicted[i].color &= 0x00FF_FFFF }
                }

                // ---- 2. 累积 + 实时层 ----
                let live = liveBuffers[liveTarget].contents()
                    .bindMemory(to: Splat.self, capacity: liveCapacity)
                var ln = 0
                let step = max(1, Int((Float(w * h) / Float(liveCapacity)).squareRoot().rounded(.up)))
                let m = f.cameraToWorld
                let c0 = m.columns.0.xyz, c1 = m.columns.1.xyz, c2 = m.columns.2.xyz
                let t = m.columns.3.xyz
                let invVoxel = 1 / SplatMap.voxel
                let newTex = stampBits | keyCode

                // 地面：朝上的平面里最低的那块。
                // 最低的那块水平面当地面，但要比镜头低 0.95 米以上：坐在桌前时最低的平面是桌面，
                // 把它当地面的话，「地面颜色」记下来的就是桌面的颜色。
                var floorY = Float.infinity
                let cameraY = f.cameraToWorld.position.y
                for pl in f.planes where pl.normal.y > 0.9 && pl.center.y < cameraY - 0.95 {
                    floorY = min(floorY, pl.center.y)
                }
                let hasFloor = floorY.isFinite
                if hasFloor { floorGuess.recenter(camPos) }

                var v = 0
                while v < h {
                    var u = 0
                    while u < w {
                        defer { u += step }
                        let z = depth[v * w + u]
                        if z < 0.15 || z > f.maxRange { continue }
                        let xc = (Float(u) + 0.5 - cx) / fx * z
                        let yc = -(Float(v) + 0.5 - cy) / fy * z
                        let p = c0 * xc + c1 * yc - c2 * z + t
                        let ci = (v * w + u) * 4
                        let r = UInt32(color[ci])
                        let g = UInt32(color[ci + 1])
                        let b = UInt32(color[ci + 2])
                        // 每隔几个像素往「方向 → 颜色」的全景小图里记一笔，给没扫到的地方做预测用。
                        if (u & 3) == 0, (v & 3) == 0 { env.add(direction: p - t, r: r, g: g, b: b) }
                        // 被压到地面上的像素（法线朝上、高度就是地面）顺手记进地面颜色图。
                        if (u & 1) == 0, (v & 1) == 0, hasFloor, abs(p.y - floorY) < 0.04,
                            !snappedNormals.isEmpty, (snappedNormals[v * w + u] >> 8) & 255 > 240
                        {
                            floorGuess.add(x: p.x, z: p.z, r: r, g: g, b: b)
                        }

                        // 已经有 splat 对上了的像素，不新建，也不进实时层：
                        // 那里由地图里位置稳定的 splat 来画，贴的是这一帧的画面。
                        if fuse, explained[v * w + u] != 0 { continue }
                        let rgb = r | (g << 8) | (b << 16)

                        // 相邻采样点在空间里隔多远：横向是「一个像素张开多宽」，
                        // 斜着看的表面还要算上相邻像素的深度差。
                        var dz: Float = 0
                        if u + step < w {
                            let zr = depth[v * w + u + step]
                            if zr > 0 { dz = abs(zr - z) }
                        }
                        if v + step < h {
                            let zd = depth[(v + step) * w + u]
                            if zd > 0 { dz = max(dz, abs(zd - z)) }
                        }
                        let lateral = Float(step) * z / fx
                        let cls = sizeClass((lateral * lateral + min(dz, lateral * 4) * min(dz, lateral * 4)).squareRoot())
                        let clsBits = cls << 14

                        if ln < liveCapacity {
                            live[ln] = Splat(
                                x: p.x, y: p.y, z: p.z, color: rgb | 0xFF00_0000,
                                tex: stampBits | SplatMap.noTex, normal: normals[v * w + u])
                            ln += 1
                        }

                        let key = SplatMap.voxelKey(p, invVoxel)

                        if let existing = table[key] {
                            let i = Int(existing)
                            let s = splats[i]
                            var hits = s.color >> 24
                            if hits == 0 {
                                // 之前被雕掉了，现在又有东西：当新的。
                                splats[i] = Splat(
                                    x: p.x, y: p.y, z: p.z, color: rgb | (1 << 24),
                                    tex: newTex | clsBits, normal: normals[v * w + u])
                                stamps[i] = frameStamp
                                kfDist[i] = z
                                if keySlot >= 0 { kfRef[keySlot] += 1 }
                                alive += 1
                                continue
                            }
                            if stamps[i] != frameStamp {
                                stamps[i] = frameStamp
                                hits = min(hits + 1, SplatMap.maxHits)
                            }
                            let or = s.color & 0xFF, og = (s.color >> 8) & 0xFF
                            let ob = (s.color >> 16) & 0xFF
                            let nr = (or * 3 + r + 2) / 4
                            let ng = (og * 3 + g + 2) / 4
                            let nb = (ob * 3 + b + 2) / 4
                            splats[i].x = s.x + (p.x - s.x) * 0.2
                            splats[i].y = s.y + (p.y - s.y) * 0.2
                            splats[i].z = s.z + (p.z - s.z) * 0.2
                            splats[i].color = nr | (ng << 8) | (nb << 16) | (hits << 24)
                            splats[i].normal = SplatMap.blendNormal(s.normal, normals[v * w + u], 0.2)
                            touch(i, z, cls, r, g, b)
                        } else if n < capacity {
                            table[key] = Int32(n)
                            splats[n] = Splat(
                                x: p.x, y: p.y, z: p.z, color: rgb | (1 << 24),
                                tex: newTex | clsBits, normal: normals[v * w + u])
                            stamps[n] = frameStamp
                            kfDist[n] = z
                            if keySlot >= 0 { kfRef[keySlot] += 1 }
                            n += 1
                            alive += 1
                        }
                    }
                    v += step
                }

                if env.hasData {
                    env.filled().withUnsafeBytes { raw in
                        envTexture.replace(
                            region: MTLRegionMake2D(0, 0, EnvGuess.width, EnvGuess.height),
                            mipmapLevel: 0, withBytes: raw.baseAddress!,
                            bytesPerRow: EnvGuess.width * 4)
                    }
                }
                var newFloorInfo: SIMD4<Float>?
                if floorGuess.dirty, let fo = floorGuess.origin {
                    floorGuess.filled().withUnsafeBytes { raw in
                        floorTexture.replace(
                            region: MTLRegionMake2D(0, 0, FloorGuess.size, FloorGuess.size),
                            mipmapLevel: 0, withBytes: raw.baseAddress!,
                            bytesPerRow: FloorGuess.size * 4)
                    }
                    newFloorInfo = SIMD4(fo.x, fo.y, 1 / (Float(FloorGuess.size) * FloorGuess.cell), 1)
                }
                let uv = hasImage ? SplatMap.worldToUV(f) : matrix_identity_float4x4
                lock.lock()
                if let newFloorInfo { floorInfo = newFloorInfo }
                envValid = env.hasData
                count = n
                liveIndex = liveTarget
                liveCount = ln
                _alive = alive
                if hasImage {
                    published[liveSlotTarget] = KeyframeGPU(worldToUV: uv, info: SIMD4(0, 1, 0, 0))
                    if keySlot >= 0 {
                        published[keySlot] = KeyframeGPU(
                            worldToUV: uv, info: SIMD4(Float(kfGen[keySlot]), 1, 0, 0))
                    }
                    liveSlot = liveSlotTarget
                    liveStamp = UInt32(frameStamp)
                    liveValid = true
                    _keyframesUsed = kfSeq.reduce(0) { $0 + ($1 != 0 ? 1 : 0) }
                } else {
                    liveValid = false
                }
                lock.unlock()
            }
        }
    }
}
