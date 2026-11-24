import ARKit
import simd

/// 真机的数据来源：ARKit 位姿 + 深度 + 相机颜色。设备支持的话，同时用前摄跟踪用户的头。
///
/// 深度按设备能力三选一：
/// 1. 有 LiDAR：直接用 sceneDepth。
/// 2. 没有 LiDAR 但有单目深度模型：模型出相对深度，用 ARKit 特征点对齐成米（iPhone Duo 走这条）。
/// 3. 都没有：只画稀疏特征点。
final class ARKitSource: NSObject, FrameSource, ARSessionDelegate {
    var onPose: ((PoseSample) -> Void)?
    var onDepth: ((DepthFrame) -> Void)?
    var onSparsePoints: (([SplatMap.Splat]) -> Void)?
    var onThumbnail: ((Int, Int, [UInt8]) -> Void)?
    var onCapture: ((MonoCapture) -> Void)?
    private(set) var statusText = "启动 ARKit…"

    private let session = ARSession()
    private let queue = DispatchQueue(label: "outofbody.arkit", qos: .userInteractive)
    private var lastDepthTime = 0.0
    private var hasDepth = false
    private var hasFace = false

    // 单目深度。estimator 和 monoBusy、monoStatus 只在 `queue` 上读写。
    private var estimator: MonoDepthEstimator?
    private let inferQueue = DispatchQueue(label: "outofbody.mono", qos: .userInitiated)
    private var monoBusy = false
    private var monoRanOnce = false
    private var monoStatus = "加载深度模型…"
    /// 最近见过的特征点（按 ARKit 给的 id 去重）。只用当前帧的点太少，对齐不稳。
    private var featurePool: [UInt64: (p: Vec3, frame: Int)] = [:]
    private var frameCount = 0
    private var faceAnchors = 0
    private var loggedFace = -1

    func start() {
        let cfg = ARWorldTrackingConfiguration()
        // 竖直的平面（墙）也要：单目深度里墙是弯的，有了平面可以直接压平。
        cfg.planeDetection = [.horizontal, .vertical]
        cfg.isAutoFocusEnabled = true
        let vf = cfg.videoFormat
        DebugLog.write(
            "CAM 格式 \(Int(vf.imageResolution.width))x\(Int(vf.imageResolution.height)) "
                + "\(vf.framesPerSecond)fps \(vf.captureDeviceType.rawValue)")
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            cfg.frameSemantics.insert(.sceneDepth)
            hasDepth = true
        }
        if ARWorldTrackingConfiguration.supportsUserFaceTracking {
            cfg.userFaceTrackingEnabled = true
            hasFace = true
        }
        session.delegate = self
        session.delegateQueue = queue
        session.run(cfg)

        if !hasDepth {
            // 模型第一次加载要编译到神经引擎，可能要几秒，别挡住 ARKit。
            inferQueue.async {
                DebugLog.write("MONO 开始加载模型")
                let t0 = CACurrentMediaTime()
                let est = MonoDepthEstimator.shared
                DebugLog.write(String(
                    format: "MONO 模型%@，用了 %.1f 秒", est == nil ? "加载失败" : "加载完成",
                    CACurrentMediaTime() - t0))
                self.queue.async {
                    self.estimator = est
                    self.monoStatus = est == nil ? "" : "单目深度就绪"
                }
            }
        }
    }

    func stop() {
        session.pause()
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        var head: Mat4?
        var floor: Float?
        var planes: [PlaneInfo] = []
        var faces = 0
        for anchor in frame.anchors {
            if let face = anchor as? ARFaceAnchor {
                faces += 1
                if face.isTracked { head = face.transform }
            } else if let plane = anchor as? ARPlaneAnchor {
                let t = plane.transform
                let c = (t * SIMD4(plane.center.x, plane.center.y, plane.center.z, 1)).xyz
                let e = plane.planeExtent
                // 太小的平面多半是误认，不拿来压深度。
                if e.width * e.height > 0.25 {
                    planes.append(PlaneInfo(
                        center: c, normal: simd_normalize(t.yAxis),
                        radius: 0.5 * (e.width * e.width + e.height * e.height).squareRoot()))
                }
                if plane.alignment == .horizontal {
                    if floor == nil || c.y < floor! { floor = c.y }
                }
            }
        }
        faceAnchors = faces
        onPose?(PoseSample(
            time: frame.timestamp, cameraToWorld: frame.camera.transform,
            headToWorld: head, floorY: floor))

        var s: String
        switch frame.camera.trackingState {
        case .normal: s = "跟踪正常"
        case .notAvailable: s = "跟踪不可用"
        case .limited(let reason):
            switch reason {
            case .initializing: s = "初始化中，慢慢移动手机"
            case .excessiveMotion: s = "动得太快"
            case .insufficientFeatures: s = "画面特征太少"
            case .relocalizing: s = "重定位中"
            @unknown default: s = "跟踪受限"
            }
        }
        if hasDepth {
            s += " · LiDAR"
        } else if !monoStatus.isEmpty {
            s += " · " + monoStatus
        } else {
            s += " · 无深度传感器（稀疏点）"
        }
        s += hasFace ? (head != nil ? " · 头部已跟踪" : " · 头部未见") : " · 头部估算"
        statusText = s
        // 头老是「未见」的原因还没查清：记下系统到底有没有给过脸的锚点。
        let faceState = head != nil ? 2 : (faceAnchors > 0 ? 1 : 0)
        if faceState != loggedFace {
            loggedFace = faceState
            DebugLog.write("FACE " + ["没有脸的锚点", "有锚点但没在跟踪", "在跟踪"][faceState])
        }

        frameCount += 1
        if !hasDepth, let cloud = frame.rawFeaturePoints {
            for (p, id) in zip(cloud.points, cloud.identifiers) {
                featurePool[id] = (p, frameCount)
            }
            // 特征点是世界坐标里的点，过几秒也还是对的；留久一点，对照点才够多。
            if frameCount % 30 == 0 {
                featurePool = featurePool.filter { frameCount - $0.value.frame < 300 }
            }
        }

        // 有深度传感器的每秒 10 次；单目的每秒 7 次左右。真机上单目那一路跑满 10 次的话，
        // 一分半钟后就热到降频（推理从 35 ms 涨到 100 ms 以上，画面掉到三四十帧）。
        guard frame.timestamp - lastDepthTime >= (hasDepth ? 0.1 : 0.14) else { return }
        if let scene = frame.sceneDepth {
            lastDepthTime = frame.timestamp
            if let df = Self.makeDepthFrame(frame, scene) { onDepth?(df) }
        } else if let estimator {
            // 上一帧还在推理就跳过。只留住像素缓冲和几个数，不留 ARFrame 本身。
            guard !monoBusy else { return }
            lastDepthTime = frame.timestamp
            monoBusy = true
            let image = frame.capturedImage
            let cam = frame.camera.transform
            let k = frame.camera.intrinsics
            let points = featurePool.values.map { $0.p }
            let first = !monoRanOnce
            monoRanOnce = true
            let time = frame.timestamp
            let exif = frame.exifData
            inferQueue.async {
                if first { DebugLog.write("MONO 第一次推理开始") }
                let status = self.runMono(
                    estimator, image: image, cameraToWorld: cam, intrinsics: k, points: points,
                    planes: planes, time: time, exif: exif)
                if first { DebugLog.write("MONO 第一次推理结束：" + status) }
                self.queue.async {
                    self.monoBusy = false
                    self.monoStatus = status
                }
            }
        } else if !hasDepth, monoStatus.isEmpty {
            lastDepthTime = frame.timestamp
            sparseFallback(frame)
        }
    }

    /// 跑一次单目深度 → 用特征点对齐成米 → 交给地图。返回一行状态文字。
    private func runMono(
        _ estimator: MonoDepthEstimator, image: CVPixelBuffer, cameraToWorld: Mat4,
        intrinsics k: simd_float3x3, points: [Vec3], planes: [PlaneInfo], time: Double,
        exif: [String: Any]
    ) -> String {
        let t0 = CACurrentMediaTime()
        guard let out = estimator.run(image) else { return "深度模型推理失败" }
        let ms = (CACurrentMediaTime() - t0) * 1000
        let w = out.width, h = out.height
        let imgW = CVPixelBufferGetWidthOfPlane(image, 0)
        let imgH = CVPixelBufferGetHeightOfPlane(image, 0)
        let sx = Float(w) / Float(imgW)
        let sy = Float(h) / Float(imgH)
        let fx = k.columns.0.x * sx, fy = k.columns.1.y * sy
        let cx = k.columns.2.x * sx, cy = k.columns.2.y * sy

        guard let hi = Self.halfResImage(image) else { return "读不到相机画面" }
        logCamera(hi, exif: exif)

        let inv = cameraToWorld.inverse
        var samples: [DepthAlign.Sample] = []
        samples.reserveCapacity(points.count)
        for p in points {
            let pc = inv * SIMD4(p.x, p.y, p.z, 1)
            let z = -pc.z
            if z < 0.3 || z > 8 { continue }
            let u = Int(fx * pc.x / z + cx)
            let v = Int(cy - fy * pc.y / z)
            if u < 1 || v < 1 || u >= w - 1 || v >= h - 1 { continue }
            samples.append(DepthAlign.Sample(u: u, v: v, z: z))
        }
        let capture = MonoCapture(
            time: time, cameraToWorld: cameraToWorld, width: w, height: h, disparity: out.values,
            fx: fx, fy: fy, cx: cx, cy: cy, samples: samples,
            imageWidth: hi.width, imageHeight: hi.height, image: hi.rgba, planes: planes)
        onCapture?(capture)
        let result = MonoPipeline.process(capture)
        if let df = result.frame {
            onDepth?(df)
        } else {
            // 这一帧没进地图，人偶手里那块屏幕还是要更新。
            onThumbnail?(hi.width, hi.height, hi.rgba)
        }
        return String(format: "单目 %.0f ms · ", ms) + result.status
    }

    private var lastCameraLog = 0.0
    /// 每 5 秒记一行相机的状态：曝光时间、感光度，和这张图有多清楚（相邻像素明暗差的平均值）。
    /// 2026-11-23 的真机录像里每一帧相机图都是糊的（清晰度 1.7，数据集里的照片是 4~5），
    /// 记下这些才分得清是曝光太长拖糊了、没对上焦，还是镜头脏了。
    private func logCamera(_ hi: (width: Int, height: Int, rgba: [UInt8]), exif: [String: Any]) {
        let now = CACurrentMediaTime()
        guard now - lastCameraLog >= 5 else { return }
        lastCameraLog = now
        var sum = 0, n = 0
        let w = hi.width, h = hi.height
        func luma(_ x: Int, _ y: Int) -> Int { Int(hi.rgba[(y * w + x) * 4 + 1]) }
        for y in stride(from: 2, to: h - 2, by: 4) {
            for x in stride(from: 2, to: w - 2, by: 4) {
                sum += abs(luma(x + 1, y) + luma(x - 1, y) + luma(x, y + 1) + luma(x, y - 1)
                    - 4 * luma(x, y))
                n += 1
            }
        }
        let exposure = (exif["ExposureTime"] as? Double) ?? 0
        let iso = (exif["ISOSpeedRatings"] as? [Any])?.first.map { "\($0)" } ?? "?"
        DebugLog.write(String(
            format: "CAM 清晰度 %.2f · 曝光 1/%.0f 秒 · ISO %@ · 镜头 %@",
            Double(sum) / Double(max(n, 1)), exposure > 0 ? 1 / exposure : 0, iso,
            (exif["LensModel"] as? String) ?? "?"))
    }

    /// 相机图缩一半的 RGBA（1920×1440 → 960×720），给贴图用。
    /// 亮度取 2x2 平均，色度本来就是半分辨率，一格对一格。全用整数算，每帧几毫秒。
    private static func halfResImage(_ image: CVPixelBuffer)
        -> (width: Int, height: Int, rgba: [UInt8])?
    {
        guard CVPixelBufferGetPlaneCount(image) >= 2 else { return nil }
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
            let cBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
        else { return nil }
        let w = CVPixelBufferGetWidthOfPlane(image, 0) / 2
        let h = CVPixelBufferGetHeightOfPlane(image, 0) / 2
        guard CVPixelBufferGetWidthOfPlane(image, 1) >= w,
            CVPixelBufferGetHeightOfPlane(image, 1) >= h
        else { return nil }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(image, 1)
        let yPtr = yBase.assumingMemoryBound(to: UInt8.self)
        let cPtr = cBase.assumingMemoryBound(to: UInt8.self)
        var out = [UInt8](repeating: 255, count: w * h * 4)
        out.withUnsafeMutableBufferPointer { o in
            for y in 0..<h {
                let y0 = yPtr + (2 * y) * yStride
                let y1 = y0 + yStride
                let c = cPtr + y * cStride
                var i = y * w * 4
                for x in 0..<w {
                    let yy = (Int(y0[2 * x]) + Int(y0[2 * x + 1]) + Int(y1[2 * x])
                        + Int(y1[2 * x + 1]) + 2) >> 2
                    let cb = Int(c[2 * x]) - 128
                    let cr = Int(c[2 * x + 1]) - 128
                    o[i] = UInt8(max(0, min(255, yy + (359 * cr) >> 8)))
                    o[i + 1] = UInt8(max(0, min(255, yy - (88 * cb + 183 * cr) >> 8)))
                    o[i + 2] = UInt8(max(0, min(255, yy + (454 * cb) >> 8)))
                    i += 4
                }
            }
        }
        return (w, h, out)
    }

    /// 把相机图（YCbCr 双平面）缩成 width×height 的 RGBA。
    private static func sampleColors(_ image: CVPixelBuffer, width w: Int, height h: Int)
        -> [UInt8]?
    {
        guard CVPixelBufferGetPlaneCount(image) >= 2 else { return nil }
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
            let cBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
        else { return nil }
        let imgW = CVPixelBufferGetWidthOfPlane(image, 0)
        let imgH = CVPixelBufferGetHeightOfPlane(image, 0)
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(image, 1)
        let yPtr = yBase.assumingMemoryBound(to: UInt8.self)
        let cPtr = cBase.assumingMemoryBound(to: UInt8.self)
        var out = [UInt8](repeating: 255, count: w * h * 4)
        for v in 0..<h {
            let iy = min(imgH - 1, (v * imgH + imgH / 2) / h)
            for u in 0..<w {
                let ix = min(imgW - 1, (u * imgW + imgW / 2) / w)
                let yv = Float(yPtr[iy * yStride + ix])
                let ci = (iy / 2) * cStride + (ix / 2) * 2
                let cb = Float(cPtr[ci]) - 128
                let cr = Float(cPtr[ci + 1]) - 128
                let i = (v * w + u) * 4
                out[i] = UInt8(max(0, min(255, yv + 1.402 * cr)))
                out[i + 1] = UInt8(max(0, min(255, yv - 0.344 * cb - 0.714 * cr)))
                out[i + 2] = UInt8(max(0, min(255, yv + 1.772 * cb)))
            }
        }
        return out
    }

    /// 没有 LiDAR 时：把 ARKit 自己跟踪用的特征点当成稀疏的环境，再单独出一张相机小图。
    /// 只是为了第一次打开不是空的，稠密的环境要等单目深度模型接上。
    private func sparseFallback(_ frame: ARFrame) {
        let image = frame.capturedImage
        guard CVPixelBufferGetPlaneCount(image) >= 2 else { return }
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
            let cBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
        else { return }
        let imgW = CVPixelBufferGetWidthOfPlane(image, 0)
        let imgH = CVPixelBufferGetHeightOfPlane(image, 0)
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(image, 1)
        let yPtr = yBase.assumingMemoryBound(to: UInt8.self)
        let cPtr = cBase.assumingMemoryBound(to: UInt8.self)

        func rgb(_ ix: Int, _ iy: Int) -> (UInt8, UInt8, UInt8) {
            let yv = Float(yPtr[iy * yStride + ix])
            let ci = (iy / 2) * cStride + (ix / 2) * 2
            let cb = Float(cPtr[ci]) - 128
            let cr = Float(cPtr[ci + 1]) - 128
            return (
                UInt8(max(0, min(255, yv + 1.402 * cr))),
                UInt8(max(0, min(255, yv - 0.344 * cb - 0.714 * cr))),
                UInt8(max(0, min(255, yv + 1.772 * cb)))
            )
        }

        let w = 256, h = 192
        var thumb = [UInt8](repeating: 255, count: w * h * 4)
        for v in 0..<h {
            let iy = min(imgH - 1, (v * imgH + imgH / 2) / h)
            for u in 0..<w {
                let c = rgb(min(imgW - 1, (u * imgW + imgW / 2) / w), iy)
                let i = (v * w + u) * 4
                thumb[i] = c.0
                thumb[i + 1] = c.1
                thumb[i + 2] = c.2
            }
        }
        onThumbnail?(w, h, thumb)

        guard let cloud = frame.rawFeaturePoints else { return }
        let k = frame.camera.intrinsics
        let inv = frame.camera.transform.inverse
        var pts: [SplatMap.Splat] = []
        pts.reserveCapacity(cloud.points.count)
        for p in cloud.points {
            let pc = inv * SIMD4(p.x, p.y, p.z, 1)
            let z = -pc.z
            if z < 0.2 || z > 8 { continue }
            let ix = Int(k.columns.0.x * pc.x / z + k.columns.2.x)
            let iy = Int(k.columns.2.y - k.columns.1.y * pc.y / z)
            if ix < 0 || iy < 0 || ix >= imgW || iy >= imgH { continue }
            let c = rgb(ix, iy)
            pts.append(SplatMap.Splat(
                x: p.x, y: p.y, z: p.z,
                color: UInt32(c.0) | (UInt32(c.1) << 8) | (UInt32(c.2) << 16)))
        }
        if !pts.isEmpty { onSparsePoints?(pts) }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        statusText = "ARKit 出错：\(error.localizedDescription)"
    }

    /// 把 LiDAR 深度图和对应位置的相机颜色拷成一份自带内存的 DepthFrame。
    private static func makeDepthFrame(_ frame: ARFrame, _ scene: ARDepthData) -> DepthFrame? {
        let depthMap = scene.depthMap
        let image = frame.capturedImage
        guard CVPixelBufferGetPixelFormatType(depthMap) == kCVPixelFormatType_DepthFloat32,
            CVPixelBufferGetPlaneCount(image) >= 2
        else { return nil }

        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        CVPixelBufferLockBaseAddress(image, .readOnly)
        let conf = scene.confidenceMap
        if let conf { CVPixelBufferLockBaseAddress(conf, .readOnly) }
        defer {
            CVPixelBufferUnlockBaseAddress(depthMap, .readOnly)
            CVPixelBufferUnlockBaseAddress(image, .readOnly)
            if let conf { CVPixelBufferUnlockBaseAddress(conf, .readOnly) }
        }

        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)
        let imgW = CVPixelBufferGetWidthOfPlane(image, 0)
        let imgH = CVPixelBufferGetHeightOfPlane(image, 0)
        guard let dBase = CVPixelBufferGetBaseAddress(depthMap),
            let yBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
            let cBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
        else { return nil }
        let dStride = CVPixelBufferGetBytesPerRow(depthMap)
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let cStride = CVPixelBufferGetBytesPerRowOfPlane(image, 1)
        let confBase = conf.flatMap { CVPixelBufferGetBaseAddress($0) }
        let confStride = conf.map { CVPixelBufferGetBytesPerRow($0) } ?? 0

        var depth = [Float](repeating: 0, count: w * h)
        var color = [UInt8](repeating: 255, count: w * h * 4)
        let yPtr = yBase.assumingMemoryBound(to: UInt8.self)
        let cPtr = cBase.assumingMemoryBound(to: UInt8.self)

        for v in 0..<h {
            let dRow = (dBase + v * dStride).assumingMemoryBound(to: Float32.self)
            let confRow = confBase.map { ($0 + v * confStride).assumingMemoryBound(to: UInt8.self) }
            let iy = min(imgH - 1, (v * imgH + imgH / 2) / h)
            for u in 0..<w {
                let i = v * w + u
                // 置信度 0 = 低：边缘和反光面上的飞点，不要。
                if let confRow, confRow[u] < 1 {
                    depth[i] = 0
                } else {
                    let d = dRow[u]
                    depth[i] = d.isFinite ? d : 0
                }
                let ix = min(imgW - 1, (u * imgW + imgW / 2) / w)
                let yv = Float(yPtr[iy * yStride + ix])
                let ci = (iy / 2) * cStride + (ix / 2) * 2
                let cb = Float(cPtr[ci]) - 128
                let cr = Float(cPtr[ci + 1]) - 128
                color[i * 4] = UInt8(max(0, min(255, yv + 1.402 * cr)))
                color[i * 4 + 1] = UInt8(max(0, min(255, yv - 0.344 * cb - 0.714 * cr)))
                color[i * 4 + 2] = UInt8(max(0, min(255, yv + 1.772 * cb)))
            }
        }

        // 内参是按相机图分辨率给的，缩到深度图分辨率。
        let k = frame.camera.intrinsics
        let sx = Float(w) / Float(imgW)
        let sy = Float(h) / Float(imgH)
        return DepthFrame(
            width: w, height: h, depth: depth, color: color,
            fx: k.columns.0.x * sx, fy: k.columns.1.y * sy,
            cx: k.columns.2.x * sx, cy: k.columns.2.y * sy,
            cameraToWorld: frame.camera.transform, maxRange: 5.0)
    }
}
