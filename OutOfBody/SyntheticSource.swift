import Foundation
import simd

/// 模拟器里没有 ARKit，用一间程序生成的房间代替：
/// 一台「手机」沿椭圆路线走，每帧光线求交出深度图和颜色，喂给跟真机完全一样的管线。
/// 房间里有一个来回滑动的箱子，用来验证「东西挪走后旧的 splat 会被雕掉」。
final class SyntheticSource: FrameSource {
    var onPose: ((PoseSample) -> Void)?
    var onDepth: ((DepthFrame) -> Void)?
    var onSparsePoints: (([SplatMap.Splat]) -> Void)?
    var onThumbnail: ((Int, Int, [UInt8]) -> Void)?
    var onCapture: ((MonoCapture) -> Void)?
    private(set) var statusText = "合成场景"

    private let queue = DispatchQueue(label: "outofbody.synthetic", qos: .userInitiated)
    private var timer: DispatchSourceTimer?
    private var tick = 0
    private let width = 256
    private let height = 192
    private let focal: Float = 212
    /// `-monoTest YES`：假装只有单目深度。把真实深度换成差一个缩放和偏移的逆深度，
    /// 再撒一些带错的稀疏对照点，让 DepthAlign 把它还原回来，用来验证对齐的数学。
    private let monoTest = UserDefaults.standard.bool(forKey: "monoTest")
    /// `-monoReal YES`：在合成画面上跑真的单目深度模型（模拟器里用 CPU，大约一秒几帧）。
    /// 对照点从真实深度里随机取。误差的形状（边缘拖影、整块偏移）跟真机是同一类，
    /// 用来在没有手机的时候调融合和去飞点。
    private let monoReal = UserDefaults.standard.bool(forKey: "monoReal")
    private var estimator: MonoDepthEstimator?
    private let inferQueue = DispatchQueue(label: "outofbody.synthetic.mono", qos: .userInitiated)
    private var inferBusy = false

    private let scene = SyntheticScene.shared

    func start() {
        if monoReal {
            estimator = MonoDepthEstimator.shared
            statusText = estimator == nil ? "合成场景 · 没有深度模型" : "合成场景 · 真模型"
        }
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: 1.0 / 30.0)
        t.setEventHandler { [weak self] in self?.step() }
        t.resume()
        timer = t
    }

    func stop() {
        timer?.cancel()
        timer = nil
    }

    private func cameraPose(at time: Float) -> Mat4 {
        let theta = 0.35 * time
        let radius: Float = 1.5
        let pos = Vec3(
            radius * cos(theta), 1.32 + 0.012 * sin(theta * 9), radius * 0.9 * sin(theta))
        let tangent = simd_normalize(Vec3(-sin(theta), 0, 0.9 * cos(theta)))
        // 镜头左右扫一扫，稍微朝下。
        let yaw = 0.55 * sin(0.45 * time)
        let c = cos(yaw), s = sin(yaw)
        var fwd = Vec3(tangent.x * c - tangent.z * s, 0, tangent.x * s + tangent.z * c)
        fwd = simd_normalize(fwd + Vec3(0, -0.16, 0))
        let right = simd_normalize(simd_cross(fwd, Vec3(0, 1, 0)))
        let up = simd_cross(right, fwd)
        // 竖拿：相机 +x 朝下，+y 朝右，+z 朝用户。
        return Mat4.basis(x: -up, y: right, z: -fwd, origin: pos)
    }

    private func step() {
        tick += 1
        let time = Float(tick) / 30
        let cam = cameraPose(at: time)
        onPose?(PoseSample(
            time: Double(time), cameraToWorld: cam, headToWorld: nil, floorY: 0))
        guard tick % 3 == 0 else { return }

        let boxes = scene.boxes(at: time)

        if monoReal, let estimator {
            if inferBusy { return }
            inferBusy = true
            let seed = UInt32(truncatingIfNeeded: tick) &* 2_654_435_761
            inferQueue.async {
                self.stepReal(estimator, cam: cam, boxes: boxes, seed: seed, time: Double(time))
                self.queue.async { self.inferBusy = false }
            }
            return
        }

        let w = width, h = height
        var depth = [Float](repeating: 0, count: w * h)
        var color = [UInt8](repeating: 255, count: w * h * 4)
        let o = cam.position
        let cx = Float(w) / 2, cy = Float(h) / 2
        let ax = cam.xAxis, ay = cam.yAxis, az = cam.zAxis

        for v in 0..<h {
            for u in 0..<w {
                // 方向不归一化，z 分量恒为 -1，于是求交参数 t 直接就是沿光轴的深度。
                let dx = (Float(u) + 0.5 - cx) / focal
                let dy = -(Float(v) + 0.5 - cy) / focal
                let d = ax * dx + ay * dy - az
                let (t, c) = scene.trace(o, d, boxes)
                let i = v * w + u
                depth[i] = t
                color[i * 4] = UInt8(max(0, min(255, c.x * 255)))
                color[i * 4 + 1] = UInt8(max(0, min(255, c.y * 255)))
                color[i * 4 + 2] = UInt8(max(0, min(255, c.z * 255)))
            }
        }
        if monoTest {
            // 每帧换一组缩放和偏移，跟真模型一样不稳定。
            let a0 = 0.30 + 0.12 * sin(time * 1.7)
            let b0 = 0.08 + 0.05 * cos(time * 1.1)
            var seed = UInt32(truncatingIfNeeded: tick) &* 2_654_435_761
            func rand() -> Float {
                seed = seed &* 1_664_525 &+ 1_013_904_223
                return Float(seed >> 8) / Float(1 << 24)
            }
            let disparity = depth.map { (1 / $0 - b0) / a0 * (1 + 0.01 * (rand() - 0.5)) }
            var samples: [DepthAlign.Sample] = []
            for k in 0..<160 {
                let u = min(w - 1, Int(rand() * Float(w)))
                let v = min(h - 1, Int(rand() * Float(h)))
                var z = depth[v * w + u]
                if k % 6 == 0 { z *= 1.6 }  // 六分之一是错的
                samples.append(DepthAlign.Sample(u: u, v: v, z: z))
            }
            guard let fit = DepthAlign.fit(disparity: disparity, width: w, samples: samples) else {
                statusText = "合成场景 · 单目对齐失败"
                return
            }
            depth = DepthAlign.metricDepth(disparity: disparity, fit: fit, maxRange: 8)
            // 真模型的深度每帧都不太一样：整体缩放飘几个百分点，画面里还有缓慢的扭曲。
            let wobble = 0.04 * sin(Float(tick) * 0.9)
            for vv in 0..<h {
                for uu in 0..<w where depth[vv * w + uu] > 0 {
                    let warp = 0.02 * sin(Float(uu) * 0.04 + Float(vv) * 0.03 + Float(tick) * 0.5)
                    depth[vv * w + uu] *= 1 + wobble + warp
                }
            }
            DepthAlign.removeEdges(&depth, width: w, height: h)
            statusText = String(
                format: "合成场景 · 单目对齐 %d/%d 误差 %.1f%%", fit.inliers, fit.total,
                fit.rms * 100)
        }
        var frame = DepthFrame(
            width: w, height: h, depth: depth, color: color,
            fx: focal, fy: focal, cx: cx, cy: cy, cameraToWorld: cam, maxRange: 8)

        // 同一个视角再渲一张 3 倍分辨率的彩色图，相当于真机上的高清相机画面。
        let scale = 3
        let hw = w * scale, hh = h * scale
        let hf = focal * Float(scale)
        let hi = scene.render(cam: cam, boxes: boxes, width: hw, height: hh, focal: hf)
        frame.imageWidth = hw
        frame.imageHeight = hh
        frame.image = hi
        if monoTest { frame.applyMonoTolerances() }
        onDepth?(frame)
    }

    /// 跟真机的 ARKitSource.runMono 走同一条路：模型 → 对照点对齐 → 去边 → 交给地图。
    private func stepReal(
        _ estimator: MonoDepthEstimator, cam: Mat4, boxes: [SyntheticScene.Box], seed s0: UInt32, time: Double
    ) {
        let scale = 3
        let hw = width * scale, hh = height * scale
        let hf = focal * Float(scale)
        let hi = scene.render(cam: cam, boxes: boxes, width: hw, height: hh, focal: hf)
        guard let pb = MonoDepthEstimator.pixelBuffer(rgba: hi, width: hw, height: hh),
            let out = estimator.run(pb)
        else {
            statusText = "合成场景 · 模型推理失败"
            return
        }
        let w = out.width, h = out.height
        let fx = hf * Float(w) / Float(hw), fy = hf * Float(h) / Float(hh)
        let cx = Float(w) / 2, cy = Float(h) / 2
        var seed = s0
        func rand() -> Float {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            return Float(seed >> 8) / Float(1 << 24)
        }
        let o = cam.position
        // 对照点：ARKit 的特征点只长在有纹理的地方（角、边、花纹），白墙上一个都没有。
        // 所以这里随机挑候选像素，只留下周围有明暗变化的那些。
        var samples: [DepthAlign.Sample] = []
        func luma(_ x: Int, _ y: Int) -> Int {
            let j = (y * hw + x) * 4
            return Int(hi[j]) * 2 + Int(hi[j + 1]) * 5 + Int(hi[j + 2])
        }
        var k = 0
        for _ in 0..<1200 where samples.count < 220 {
            let u = 2 + min(w - 5, Int(rand() * Float(w - 4)))
            let v = 2 + min(h - 5, Int(rand() * Float(h - 4)))
            let ix = (u * hw + hw / 2) / w, iy = (v * hh + hh / 2) / h
            let gx = abs(luma(ix + 3, iy) - luma(ix - 3, iy))
            let gy = abs(luma(ix, iy + 3) - luma(ix, iy - 3))
            if max(gx, gy) < 90 { continue }
            let dx = (Float(u) + 0.5 - cx) / fx
            let dy = -(Float(v) + 0.5 - cy) / fy
            var (z, _) = scene.trace(o, cam.xAxis * dx + cam.yAxis * dy - cam.zAxis, boxes)
            z *= 1 + 0.03 * (rand() - 0.5)
            k += 1
            if k % 8 == 0 { z *= 1.5 }  // 八分之一是错的
            if z > 0.3, z < 8 { samples.append(DepthAlign.Sample(u: u, v: v, z: z)) }
        }
        // ARKit 在真机上认得出地面，墙要有纹理才认得出：这里给地面和两面墙。
        let rl = scene.roomLo, rh = scene.roomHi
        let planes = [
            PlaneInfo(center: Vec3(0, 0, 0), normal: Vec3(0, 1, 0), radius: 5),
            PlaneInfo(center: Vec3(rl.x, rh.y / 2, 0), normal: Vec3(1, 0, 0), radius: rh.z + 0.2),
            PlaneInfo(center: Vec3(0, rh.y / 2, rl.z), normal: Vec3(0, 0, 1), radius: rh.x + 0.1),
        ]
        let capture = MonoCapture(
            time: time, cameraToWorld: cam, width: w, height: h, disparity: out.values,
            fx: fx, fy: fy, cx: cx, cy: cy, samples: samples,
            imageWidth: hw, imageHeight: hh, image: hi, planes: planes)
        onCapture?(capture)
        let result = MonoPipeline.process(capture)
        statusText = "合成场景 · 真模型 · " + result.status
        if let frame = result.frame { onDepth?(frame) }
    }

    /// 真实表面上均匀撒的点（4 cm 一个），评估地图盖住了多少用。
    static func truthSamples() -> [Vec3] {
        var out: [Vec3] = []
        func face(_ o: Vec3, _ a: Vec3, _ b: Vec3) {
            let na = Int(simd_length(a) / 0.04), nb = Int(simd_length(b) / 0.04)
            guard na > 0, nb > 0 else { return }
            for i in 0..<na {
                for j in 0..<nb {
                    let p = o + a * ((Float(i) + 0.5) / Float(na)) + b * ((Float(j) + 0.5) / Float(nb))
                    out.append(p)
                }
            }
        }
        func box(_ lo: Vec3, _ hi: Vec3, bottom: Bool) {
            let d = hi - lo
            let x = Vec3(d.x, 0, 0), y = Vec3(0, d.y, 0), z = Vec3(0, 0, d.z)
            face(lo, x, y)
            face(lo + z, x, y)
            face(lo, z, y)
            face(lo + x, z, y)
            face(lo + y, x, z)
            if bottom { face(lo, x, z) }
        }
        box(Vec3(-3.5, 0, -3), Vec3(3.5, 2.8, 3), bottom: true)
        box(Vec3(-1.9, 0, -2.7), Vec3(-0.6, 0.75, -1.9), bottom: false)
        box(Vec3(2.2, 0, -1.1), Vec3(3.3, 0.8, 1.1), bottom: false)
        box(Vec3(-3.45, 0, 0.9), Vec3(-2.95, 1.9, 2.3), bottom: false)
        box(Vec3(0.7, 0, 2.1), Vec3(1.3, 0.6, 2.7), bottom: false)
        box(Vec3(-0.18, 0, -0.18), Vec3(0.18, 2.8, 0.18), bottom: false)
        return out
    }

    /// 到真实表面的距离（米），评估地图质量用。来回滑的那个箱子扫过的范围不算，返回 nil。
    static func distanceToTruth(_ p: Vec3) -> Float? {
        if p.x > -2.65, p.x < -1.95, p.y < 0.6, p.z > -2.85, p.z < 0.85 { return nil }
        func box(_ lo: Vec3, _ hi: Vec3) -> Float {
            let c = (lo + hi) * 0.5, half = (hi - lo) * 0.5
            let q = simd_abs(p - c) - half
            let outside = simd_length(simd_max(q, Vec3(repeating: 0)))
            let inside = min(max(q.x, max(q.y, q.z)), 0)
            return abs(outside + inside)
        }
        var best = box(Vec3(-3.5, 0, -3), Vec3(3.5, 2.8, 3))
        let boxes: [(Vec3, Vec3)] = [
            (Vec3(-1.9, 0, -2.7), Vec3(-0.6, 0.75, -1.9)),
            (Vec3(2.2, 0, -1.1), Vec3(3.3, 0.8, 1.1)),
            (Vec3(-3.45, 0, 0.9), Vec3(-2.95, 1.9, 2.3)),
            (Vec3(0.7, 0, 2.1), Vec3(1.3, 0.6, 2.7)),
            (Vec3(-0.18, 0, -0.18), Vec3(0.18, 2.8, 0.18)),
        ]
        for b in boxes { best = min(best, box(b.0, b.1)) }
        return best
    }
}
