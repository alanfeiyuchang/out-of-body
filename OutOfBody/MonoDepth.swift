import CoreML
import CoreVideo
import Foundation
import Vision
import simd

/// 单目深度模型给的是「相对的逆深度」：数值越大越近，但差一个未知的缩放和偏移，
/// 而且每张图的缩放和偏移都不一样。真实的逆深度 1/z ≈ a·d + b。
///
/// ARKit 跟踪用的特征点有真实的米制深度，虽然很稀疏。把它们投到图上当作对照，
/// 每帧拟合一次 a、b，就能把整张相对深度图换算成米。
enum DepthAlign {
    /// 对照点从哪来。
    /// feature：ARKit 的特征点，最可信，但只长在有纹理的地方，常常只有二三十个。
    /// map：地图里看过很多次的 splat，多帧平均过，但它本身是靠以前的对齐建出来的。
    /// plane：视线和 ARKit 认出来的平面（地面、墙）的交点。平面很准，
    ///        但这个像素看到的不一定是平面（可能被家具挡着），所以只是「候选」。
    enum Kind: UInt8 {
        case feature, map, plane
    }

    struct Sample {
        var u: Int
        var v: Int
        var z: Float
        var kind: Kind = .feature
    }

    struct Fit {
        var a: Float
        var b: Float
        var inliers: Int
        var total: Int
        /// 内点的相对误差均方根（0.05 = 5%）。
        var rms: Float
        /// 特征点 / 地图点 / 平面候选各有多少个内点。
        var byKind = [0, 0, 0]
        /// 特征点和地图点里内点占多少。
        var trustedRatio: Float = 0
        /// 内点的逆深度读数跨了多大范围（模型输出的单位）。全挤在同一个深度上时，
        /// 缩放和偏移分不开，这一帧只在那个深度附近是对的。
        var spread: Float = 0
    }

    static let inlierTolerance: Float = 0.10

    /// RANSAC：对照点里总有错的（特征点落在遮挡边缘、平面候选其实被家具挡着），
    /// 先随机取两点定一条线、数内点，挑最好的那条，再用它的内点做一次最小二乘。
    ///
    /// 「最好」不是简单地数个数：平面候选可以有几百个，而且可能大半是错的，
    /// 所以特征点算两票、平面候选算半票，并且要求特征点和地图点里至少四成认可这条线。
    /// 后一条防的是「桌面和地面平行」这类情况：把桌面当成地面，平面候选也能自圆其说，
    /// 只有落在桌面上的特征点会反对。
    static func fit(disparity: [Float], width: Int, samples: [Sample]) -> Fit? {
        let n = samples.count
        guard n >= 8 else { return nil }
        var d = [Float](repeating: 0, count: n)
        var y = [Float](repeating: 0, count: n)
        var kind = [Int](repeating: 0, count: n)
        var totals = [0, 0, 0]
        for (i, s) in samples.enumerated() {
            d[i] = disparity[s.v * width + s.u]
            y[i] = 1 / s.z
            kind[i] = Int(s.kind.rawValue)
            totals[kind[i]] += 1
        }
        let trustedTotal = totals[0] + totals[1]

        var seed: UInt32 = 0x9E37_79B9
        func next(_ bound: Int) -> Int {
            seed = seed &* 1_664_525 &+ 1_013_904_223
            return Int(seed >> 8) % bound
        }

        var bestScore: Float = 0
        var bestA: Float = 0
        var bestB: Float = 0
        for _ in 0..<140 {
            let i = next(n)
            let j = next(n)
            let dd = d[i] - d[j]
            if i == j || abs(dd) < 1e-4 { continue }
            let a = (y[i] - y[j]) / dd
            if a <= 0 { continue }
            let b = y[i] - a * d[i]
            var c = [0, 0, 0]
            for k in 0..<n where abs(a * d[k] + b - y[k]) < inlierTolerance * y[k] {
                c[kind[k]] += 1
            }
            if (c[0] + c[1]) * 5 < trustedTotal * 2 || c[0] * 5 < totals[0] * 2 { continue }
            let score = Float(c[0]) * 2 + Float(c[1]) + Float(c[2]) * 0.5
            if score > bestScore {
                bestScore = score
                bestA = a
                bestB = b
            }
        }
        guard bestScore > 0 else { return nil }

        // 在内点上按 1/y² 加权做最小二乘，等价于最小化相对误差。平面候选的权重减半。
        var sw: Float = 0, sd: Float = 0, sy: Float = 0, sdd: Float = 0, sdy: Float = 0
        for k in 0..<n where abs(bestA * d[k] + bestB - y[k]) < inlierTolerance * y[k] {
            let w = (kind[k] == 2 ? 0.5 : 1) / (y[k] * y[k])
            sw += w
            sd += w * d[k]
            sy += w * y[k]
            sdd += w * d[k] * d[k]
            sdy += w * d[k] * y[k]
        }
        let det = sw * sdd - sd * sd
        guard abs(det) > 1e-9 else { return nil }
        let a = (sw * sdy - sd * sy) / det
        let b = (sdd * sy - sd * sdy) / det
        guard a > 0 else { return nil }

        var err: Float = 0
        var c = [0, 0, 0]
        var dLo = Float.infinity, dHi = -Float.infinity
        for k in 0..<n {
            let r = (a * d[k] + b - y[k]) / y[k]
            if abs(r) < inlierTolerance {
                err += r * r
                c[kind[k]] += 1
                dLo = min(dLo, d[k])
                dHi = max(dHi, d[k])
            }
        }
        let count = c[0] + c[1] + c[2]
        guard count >= 8 else { return nil }
        return Fit(
            a: a, b: b, inliers: count, total: n, rms: (err / Float(count)).squareRoot(),
            byKind: c,
            trustedRatio: trustedTotal > 0 ? Float(c[0] + c[1]) / Float(trustedTotal) : 0,
            spread: dHi - dLo)
    }

    /// 整张图只用一组 a、b 对齐，对不上的是「一块一块」的偏差：这件家具整体偏近、那面墙整体偏远。
    /// 对照点上能直接看出这种偏差（真实深度 / 对齐后的深度），在一张粗网格上取平均，
    /// 没有对照点的格子往 1 靠，抹平之后乘回深度图。
    static func localCorrection(
        _ depth: inout [Float], width w: Int, height h: Int, disparity: [Float], fit: Fit,
        samples: [Sample]
    ) {
        let gw = 8, gh = 6
        var sum = [Float](repeating: 0, count: gw * gh)
        var cnt = [Float](repeating: 0, count: gw * gh)
        for s in samples {
            let inv = fit.a * disparity[s.v * w + s.u] + fit.b
            if inv <= 0 { continue }
            let r = s.z * inv  // 真实深度 / 预测深度
            // 差得太多的是错点或者遮挡边缘，不是这种偏差。
            // 平面候选收得更严：差一截的多半是平面前面摆着东西。
            if abs(r - 1) > (s.kind == .plane ? 0.08 : 0.2) { continue }
            let g = (s.v * gh / h) * gw + s.u * gw / w
            sum[g] += r
            cnt[g] += 1
        }
        // 每个格子先验上放 3 个「比值 = 1」的假点：对照点少的格子不敢动太多。
        var cell = [Float](repeating: 1, count: gw * gh)
        for g in 0..<gw * gh { cell[g] = (sum[g] + 3) / (cnt[g] + 3) }
        var smooth = cell
        for gy in 0..<gh {
            for gx in 0..<gw {
                var acc: Float = 0
                var k: Float = 0
                for dy in -1...1 where gy + dy >= 0 && gy + dy < gh {
                    for dx in -1...1 where gx + dx >= 0 && gx + dx < gw {
                        let wgt: Float = (dx == 0 && dy == 0) ? 2 : 1
                        acc += cell[(gy + dy) * gw + gx + dx] * wgt
                        k += wgt
                    }
                }
                smooth[gy * gw + gx] = acc / k
            }
        }
        for v in 0..<h {
            let gyf = (Float(v) + 0.5) * Float(gh) / Float(h) - 0.5
            let y0 = max(0, min(gh - 1, Int(gyf.rounded(.down))))
            let y1 = min(gh - 1, y0 + 1)
            let ty = max(0, min(1, gyf - Float(y0)))
            for u in 0..<w where depth[v * w + u] > 0 {
                let gxf = (Float(u) + 0.5) * Float(gw) / Float(w) - 0.5
                let x0 = max(0, min(gw - 1, Int(gxf.rounded(.down))))
                let x1 = min(gw - 1, x0 + 1)
                let tx = max(0, min(1, gxf - Float(x0)))
                let a = smooth[y0 * gw + x0] * (1 - tx) + smooth[y0 * gw + x1] * tx
                let b = smooth[y1 * gw + x0] * (1 - tx) + smooth[y1 * gw + x1] * tx
                depth[v * w + u] *= a * (1 - ty) + b * ty
            }
        }
    }

    /// 相对逆深度 → 米。超出量程或者算出来是负数的像素记 0（无效）。
    static func metricDepth(disparity: [Float], fit: Fit, maxRange: Float) -> [Float] {
        let minInverse = 1 / maxRange
        return disparity.map { d in
            let inv = fit.a * d + fit.b
            return inv > minInverse ? 1 / inv : 0
        }
    }

    /// 单目模型在前景和背景的交界处会给出一串「半空中」的过渡值，
    /// 反投影出来就是从物体边缘拖向背景的一片飞点。深度跳变大的像素直接作废。
    static let edgeBase = MonoPipeline.number("edgeBase", 0.012)
    static let edgePerMeter = MonoPipeline.number("edgePerMeter", 0.035)

    static func removeEdges(_ depth: inout [Float], width w: Int, height h: Int) {
        let src = depth
        for v in 0..<h {
            for u in 0..<w {
                let i = v * w + u
                let z = src[i]
                if z <= 0 { continue }
                if u == 0 || v == 0 || u == w - 1 || v == h - 1 {
                    depth[i] = 0
                    continue
                }
                let limit = edgeBase + edgePerMeter * z
                let l = src[i - 1], r = src[i + 1], t = src[i - w], b = src[i + w]
                if l <= 0 || r <= 0 || t <= 0 || b <= 0 || abs(l - z) > limit || abs(r - z) > limit
                    || abs(t - z) > limit || abs(b - z) > limit
                {
                    depth[i] = 0
                }
            }
        }
    }
}

/// 单目深度一帧的原始材料：模型吐出来的相对逆深度、当时能用的对照点、相机图和位姿。
/// 真机、合成测试和回放都先凑出这个，再交给 MonoPipeline 变成 DepthFrame。
/// 录制存的也是它，所以对齐和去飞点的算法改了以后，旧的录像可以直接重跑。
struct MonoCapture {
    var time: Double
    var cameraToWorld: Mat4
    var width: Int
    var height: Int
    var disparity: [Float]
    /// 逆深度图分辨率下的内参。
    var fx: Float
    var fy: Float
    var cx: Float
    var cy: Float
    var samples: [DepthAlign.Sample]
    var imageWidth: Int
    var imageHeight: Int
    var image: [UInt8]  // RGBA
    var planes: [PlaneInfo] = []
}

enum MonoPipeline {
    static let planesDisabled = UserDefaults.standard.bool(forKey: "noPlanes")
    static let mapAlign = flag("mapAlign", true)
    static let localAlign = flag("localAlign", true)
    static let planeAlign = flag("planeAlign", true)
    static let minSpread = number("minSpread", 0.2)
    static let sampleKeep = Int(number("sampleKeep", 0))
    /// 统计：处理了多少帧、其中多少帧对齐不上被跳过（回放结束时打出来）。
    static var framesSeen = 0
    static var framesSkipped = 0
    /// 地图里已经看准的点投到这一帧上，当作额外的对照点（由 AppModel 接到 SplatMap 上）。
    static var mapSamples: ((MonoCapture) -> [DepthAlign.Sample])?

    /// 单目深度的误差跟距离成正比，太远的不要。`-monoRange 4` 可覆盖。
    static let maxRange: Float = number("monoRange", 4.5)

    static func number(_ key: String, _ fallback: Float) -> Float {
        let d = UserDefaults.standard
        return d.object(forKey: key) != nil ? d.float(forKey: key) : fallback
    }

    private static func flag(_ key: String, _ fallback: Bool) -> Bool {
        let d = UserDefaults.standard
        return d.object(forKey: key) != nil ? d.bool(forKey: key) : fallback
    }

    /// 整张逆深度图的读数范围（5% ~ 95% 分位之间）。
    private static func disparityRange(_ d: [Float]) -> Float {
        var v: [Float] = []
        v.reserveCapacity(d.count / 7 + 1)
        var i = 0
        while i < d.count {
            v.append(d[i])
            i += 7
        }
        v.sort()
        guard v.count > 20 else { return 0 }
        return v[v.count * 95 / 100] - v[v.count * 5 / 100]
    }

    /// 每隔 10 个像素取一个，看这条视线打在哪块已知平面上（取最近的一块），当作对照点的候选。
    static func planeSamples(_ c: MonoCapture) -> [DepthAlign.Sample] {
        guard !c.planes.isEmpty else { return [] }
        let m = c.cameraToWorld
        let ax = m.xAxis, ay = m.yAxis, az = m.zAxis, o = m.position
        let ks = c.planes.map { simd_dot($0.center - o, $0.normal) }
        var out: [DepthAlign.Sample] = []
        var v = 5
        while v < c.height - 2 {
            let dy = -(Float(v) + 0.5 - c.cy) / c.fy
            var u = 5
            while u < c.width - 2 {
                let dx = (Float(u) + 0.5 - c.cx) / c.fx
                let d = ax * dx + ay * dy - az
                var best: Float = 0
                for (i, p) in c.planes.enumerated() {
                    let dn = simd_dot(d, p.normal)
                    // 贴着平面擦过去的视线，交点差一点点角度就差很远，不要。
                    if abs(dn) < 0.12 * simd_length(d) { continue }
                    let t = ks[i] / dn
                    if t < 0.3 || t > maxRange || (best > 0 && t > best) { continue }
                    if simd_length_squared(o + d * t - p.center) < p.radius * p.radius { best = t }
                }
                if best > 0 { out.append(DepthAlign.Sample(u: u, v: v, z: best, kind: .plane)) }
                u += 10
            }
            v += 10
        }
        return out
    }

    /// 对齐成米 → 去掉边缘飞点 → 配上颜色和容差。对齐不可靠的帧返回 nil，宁可少更新一次。
    static func process(_ c: MonoCapture) -> (frame: DepthFrame?, status: String) {
        let r = align(c)
        framesSeen += 1
        if r.frame == nil { framesSkipped += 1 }
        return r
    }

    private static func align(_ c: MonoCapture) -> (frame: DepthFrame?, status: String) {
        let w = c.width, h = c.height
        // 特征点只长在有纹理的地方，白墙和天花板上没有，那里的深度就没人管。
        // 地图里被看过很多次的点已经是多帧的平均，比单帧准，拿来补上这些空白。
        // 特征点自己至少要有十来个：全靠地图对齐的话，错了没人纠正，会越漂越远。
        // （真机上手机不怎么动、或者对着素净的地方时，特征点常常只有二三十个，
        //   光靠它们过不了下面的门槛，一帧帧都被跳过，所以门槛不能定得太高。）
        var own = c.samples
        // `-sampleKeep 25`：每帧只留这么多个特征点，模拟真机上特征点很少的情况。
        if sampleKeep > 0, own.count > sampleKeep {
            let every = Float(own.count) / Float(sampleKeep)
            own = (0..<sampleKeep).map { own[Int(Float($0) * every)] }
        }
        var samples = own
        if mapAlign, let extra = mapSamples?(c) { samples += extra }
        if planeAlign, !planesDisabled { samples += planeSamples(c) }
        guard let fit = DepthAlign.fit(disparity: c.disparity, width: w, samples: samples) else {
            return (nil, "对齐失败（\(own.count) 点）")
        }
        // 能不能信这一帧：
        // 1. 得有「不是我们自己建出来的」东西撑着：特征点或者平面。全靠地图的话错了没人纠正。
        // 2. 内点够多；不够多的话，特征点和地图点里得有六成以上认可，而且拟合得很紧。
        // 3. 内点得跨开一段深度，不然缩放和偏移分不开。
        let anchors = fit.byKind[0] + fit.byKind[2]
        let strong = fit.inliers >= 25 && fit.rms < 0.07
        let tight = fit.inliers >= 10 && fit.trustedRatio >= 0.6 && fit.rms < 0.05
        let range = disparityRange(c.disparity)
        guard anchors >= 8, strong || tight, fit.spread >= minSpread * range else {
            return (
                nil,
                String(
                    format: "对齐太弱，跳过（%d/%d，特征 %d 地图 %d 平面 %d，跨度 %.0f%%）",
                    fit.inliers, fit.total, fit.byKind[0], fit.byKind[1], fit.byKind[2],
                    fit.spread / max(range, 1e-6) * 100)
            )
        }
        var depth = DepthAlign.metricDepth(disparity: c.disparity, fit: fit, maxRange: maxRange)
        if localAlign {
            DepthAlign.localCorrection(
                &depth, width: w, height: h, disparity: c.disparity, fit: fit, samples: samples)
        }
        DepthAlign.removeEdges(&depth, width: w, height: h)

        var color = [UInt8](repeating: 255, count: w * h * 4)
        let iw = c.imageWidth, ih = c.imageHeight
        c.image.withUnsafeBufferPointer { img in
            for v in 0..<h {
                let iy = min(ih - 1, (v * ih + ih / 2) / h)
                for u in 0..<w {
                    let ix = min(iw - 1, (u * iw + iw / 2) / w)
                    let i = (v * w + u) * 4, j = (iy * iw + ix) * 4
                    color[i] = img[j]
                    color[i + 1] = img[j + 1]
                    color[i + 2] = img[j + 2]
                }
            }
        }
        var df = DepthFrame(
            width: w, height: h, depth: depth, color: color,
            fx: c.fx, fy: c.fy, cx: c.cx, cy: c.cy, cameraToWorld: c.cameraToWorld,
            maxRange: maxRange)
        df.applyMonoTolerances()
        df.imageWidth = iw
        df.imageHeight = ih
        df.image = c.image
        if !planesDisabled { df.planes = c.planes }
        return (
            df,
            String(
                format: "对齐 %d/%d（特征 %d 地图 %d 平面 %d）误差 %.0f%%", fit.inliers, fit.total,
                fit.byKind[0], fit.byKind[1], fit.byKind[2], fit.rms * 100)
        )
    }
}

/// Depth Anything V2 Small（CoreML）。模型文件不在仓库里，用 tools/fetch_model.sh 下载；
/// 没有模型时 init 返回 nil，调用方退回稀疏点。
final class MonoDepthEstimator {
    /// 全 app 共用一份：模型有 50 MB，加载要好几秒，相机那路和预测那路没必要各载一份。
    static let shared = MonoDepthEstimator()

    private let request: VNCoreMLRequest
    private let lock = NSLock()

    init?() {
        guard
            let url = Bundle.main.url(
                forResource: "DepthAnythingV2SmallF16", withExtension: "mlmodelc")
        else { return nil }
        let cfg = MLModelConfiguration()
        #if targetEnvironment(simulator)
            // 模拟器里走 GPU 的话输出全是 0。
            cfg.computeUnits = .cpuOnly
        #else
            cfg.computeUnits = .all
        #endif
        guard let model = try? MLModel(contentsOf: url, configuration: cfg),
            let vn = try? VNCoreMLModel(for: model)
        else { return nil }
        request = VNCoreMLRequest(model: vn)
        // 整张图直接缩到模型输入尺寸，输出和原图是线性对应的，内参按比例缩就行。
        request.imageCropAndScaleOption = .scaleFill
    }

    /// RGBA 字节 → 模型能吃的像素缓冲（BGRA）。合成场景测试用。
    static func pixelBuffer(rgba: [UInt8], width: Int, height: Int) -> CVPixelBuffer? {
        var pb: CVPixelBuffer?
        let attrs = [kCVPixelBufferIOSurfacePropertiesKey: [:] as CFDictionary] as CFDictionary
        guard
            CVPixelBufferCreate(
                kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA, attrs, &pb)
                == kCVReturnSuccess, let pb
        else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let stride = CVPixelBufferGetBytesPerRow(pb)
        for y in 0..<height {
            let row = (base + y * stride).assumingMemoryBound(to: UInt8.self)
            for x in 0..<width {
                let i = (y * width + x) * 4
                row[x * 4] = rgba[i + 2]
                row[x * 4 + 1] = rgba[i + 1]
                row[x * 4 + 2] = rgba[i]
                row[x * 4 + 3] = 255
            }
        }
        return pb
    }

    /// 返回 2x2 平均后的相对逆深度图（模型输出 518×392 → 259×196）。
    func run(_ image: CVPixelBuffer) -> (width: Int, height: Int, values: [Float])? {
        lock.lock()
        defer { lock.unlock() }
        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .up)
        do {
            try handler.perform([request])
        } catch {
            return nil
        }
        guard let obs = request.results?.first as? VNPixelBufferObservation else { return nil }
        let pb = obs.pixelBuffer
        guard CVPixelBufferGetPixelFormatType(pb) == kCVPixelFormatType_OneComponent16Half
        else { return nil }
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let sw = CVPixelBufferGetWidth(pb)
        let sh = CVPixelBufferGetHeight(pb)
        let stride = CVPixelBufferGetBytesPerRow(pb)
        let w = sw / 2
        let h = sh / 2
        var out = [Float](repeating: 0, count: w * h)
        for v in 0..<h {
            let r0 = (base + (v * 2) * stride).assumingMemoryBound(to: Float16.self)
            let r1 = (base + (v * 2 + 1) * stride).assumingMemoryBound(to: Float16.self)
            for u in 0..<w {
                let s = Float(r0[u * 2]) + Float(r0[u * 2 + 1]) + Float(r1[u * 2])
                    + Float(r1[u * 2 + 1])
                out[v * w + u] = s * 0.25
            }
        }
        return (w, h, out)
    }
}
